"""Dependency parser: biaffine attention for HEAD + linear head for DEPREL."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import ParserDataset, _build_vocab
from src.models.base import BorgBaseModel
from src.models.evaluation import average_f1, evaluate_predictions, print_validation_metrics
from src.models.checkpoints import save_training_models


# ---------------------------------------------------------------------------
# Biaffine attention
# ---------------------------------------------------------------------------

class BiaffineAttention(nn.Module):
    """Biaffine scoring: score[b,i,j] = dep[b,i] W head[b,j]."""

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.out_features = out_features
        # Weight tensor shape: (out_features, in_features+1, in_features+1)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features + 1, in_features + 1)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h_dep: torch.Tensor, h_head: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_dep:  (B, N, H)
            h_head: (B, N, H)
        Returns:
            scores: (B, N, N) when out_features==1
                    (B, N, N, out_features) when out_features>1
        """
        batch, seq_len, hidden = h_dep.shape
        ones = torch.ones(batch, seq_len, 1, device=h_dep.device, dtype=h_dep.dtype)
        h_dep = torch.cat([h_dep, ones], dim=-1)    # (B, N, H+1)
        h_head = torch.cat([h_head, ones], dim=-1)  # (B, N, H+1)

        # einsum: b i h, o h k, b j k -> b i j o
        scores = torch.einsum("bih,ohk,bjk->bijo", h_dep, self.weight, h_head)
        if self.out_features == 1:
            return scores.squeeze(-1)  # (B, N, N)
        return scores  # (B, N, N, O)


# ---------------------------------------------------------------------------
# Greedy arc decoding
# ---------------------------------------------------------------------------

def greedy_decode(scores: torch.Tensor, word_positions: Dict[int, int]) -> Dict[int, int]:
    """Decode a dependency tree greedily from raw arc scores.

    Rather than running Chu-Liu-Edmonds, this sorts every candidate
    (dependent, head) arc by score (descending) and walks through them,
    only keeping an arc if the dependent doesn't already have a head and
    adding it wouldn't close a cycle. This is a simple, fast approximation
    of maximum spanning arborescence decoding.

    Args:
        scores: (L, L) raw arc scores for the full sub-word sequence, where
            scores[dep, head] is the score of attaching sub-word position
            ``dep`` to sub-word position ``head``. Position 0 (the CLS
            token) represents the virtual ROOT node.
        word_positions: mapping of word-id -> first sub-word position.

    Returns:
        Mapping of word-id -> head position (0 means ROOT).
    """
    ROOT = 0
    positions = list(word_positions.values())
    nodes = [ROOT] + positions

    # Vectorized score extraction: build the (dep x head) sub-matrix once
    # instead of calling `.item()` for every pair in a Python double loop.
    # Complexity is O(D*H) ~ O(L^2) in the number of words per sentence,
    # which is negligible for typical sentence lengths; this mirrors how
    # much work a full Chu-Liu-Edmonds decoder would do anyway.
    positions_t = torch.tensor(positions, dtype=torch.long, device=scores.device)
    nodes_t = torch.tensor(nodes, dtype=torch.long, device=scores.device)
    sub_scores = scores[positions_t][:, nodes_t]  # (D, H)
    self_loop_mask = positions_t.unsqueeze(1) == nodes_t.unsqueeze(0)  # (D, H)
    sub_scores = sub_scores.masked_fill(self_loop_mask, float("-inf"))

    flat_order = torch.argsort(sub_scores.reshape(-1), descending=True).tolist()
    num_heads = len(nodes)
    flat_scores = sub_scores.reshape(-1).tolist()
    flat_self_loop = self_loop_mask.reshape(-1).tolist()
    candidates = []
    for idx in flat_order:
        if flat_self_loop[idx]:
            continue  # skip masked self-loop entries, wherever they sort to
        dep_i, head_i = divmod(idx, num_heads)
        candidates.append((flat_scores[idx], positions[dep_i], nodes[head_i]))

    # Group by dependent (score-descending) for a uniform, single-pass
    # head-selection routine (used both for the initial greedy pass and
    # the fallback below).
    by_dep: Dict[int, list] = {}
    for score, dep, head in candidates:
        by_dep.setdefault(dep, []).append((score, head))

    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    root_used = False

    def is_allowed(dep: int, head: int, relax_root_cap: bool, relax_cycle_check: bool) -> bool:
        """Shared constraint check used by both the main greedy pass and
        the fallback `pick_head` below, so the root-cap/cycle rules can't
        drift out of sync between the two call sites."""
        if not relax_root_cap and head == ROOT and root_used:
            return False
        if not relax_cycle_check and find(dep) == find(head):
            return False
        return True

    def pick_head(dep: int, relax_root_cap: bool, relax_cycle_check: bool) -> Optional[int]:
        """Pick the best-scoring head for *dep* satisfying the active
        constraints, or ``None`` if no candidate qualifies."""
        for _, head in by_dep.get(dep, []):
            if is_allowed(dep, head, relax_root_cap, relax_cycle_check):
                return head
        return None

    assigned: Dict[int, int] = {}
    # Note: this main pass intentionally does *not* call `pick_head` — it
    # walks the single globally-sorted `candidates` list so that arcs are
    # considered strictly in overall score order across all dependents
    # (the core of the greedy algorithm), whereas `pick_head` scans only
    # one dependent's own candidates and is used solely by the fallback
    # pass below. Both share the same `is_allowed` constraint check
    # (root cap + cycle), so they can't drift out of sync.
    for score, dep, head in candidates:
        if dep in assigned:
            continue
        if not is_allowed(dep, head, relax_root_cap=False, relax_cycle_check=False):
            continue
        assigned[dep] = head
        union(dep, head)
        if head == ROOT:
            root_used = True

    # Fallback for any dependent that never got a head (can happen once
    # cycle-avoidance and the single-root constraint rule out every
    # remaining candidate). Preference order: (1) cycle-free head that
    # also respects the single-root cap, (2) cycle-free head ignoring the
    # single-root cap, (3) best-scoring head regardless of cycles. Stage
    # (3) fully relaxes both constraints, so it is guaranteed to return a
    # candidate (every dependent has at least the ROOT edge in `by_dep`);
    # the assertion below makes that invariant explicit so a violation is
    # caught immediately instead of silently producing a malformed tree.
    # Note: ROOT is 0, so candidates must be compared against ``None``
    # explicitly rather than relying on truthiness.
    for dep in positions:
        if dep in assigned:
            continue
        chosen = pick_head(dep, relax_root_cap=False, relax_cycle_check=False)
        if chosen is None:
            chosen = pick_head(dep, relax_root_cap=True, relax_cycle_check=False)
        if chosen is None:
            chosen = pick_head(dep, relax_root_cap=True, relax_cycle_check=True)
        assert chosen is not None, (
            f"pick_head returned None for dependent {dep} even with both "
            "constraints fully relaxed; every dependent should have at "
            "least a ROOT candidate in `by_dep`"
        )
        assigned[dep] = chosen
        # Only merge components when the chosen arc doesn't already close
        # a cycle (relevant when it was picked under `relax_cycle_check`):
        # `find(dep) == find(chosen)` means they're already in the same
        # component, so skipping the merge here avoids ever reassigning a
        # root's parent pointer based on an arc that doesn't actually
        # connect two distinct trees.
        if find(dep) != find(chosen):
            union(dep, chosen)
        if chosen == ROOT:
            root_used = True

    return {wid: assigned[pos] for wid, pos in word_positions.items()}


# ---------------------------------------------------------------------------
# Parser model
# ---------------------------------------------------------------------------

class ParserModel(BorgBaseModel):
    """Biaffine dependency parser."""

    ARC_HIDDEN = 512
    REL_HIDDEN = 128

    def __init__(
        self,
        config: BorgConfig,
        deprel_vocab: Optional[Dict[str, int]] = None,
    ):
        super().__init__(config, "parser")
        self.deprel_vocab = deprel_vocab or {"<PAD>": 0, "<UNK>": 1}
        self._build_heads()

    def _build_heads(self) -> None:
        H = self.hidden_size
        # Arc MLP
        self.arc_head_mlp = nn.Sequential(nn.Linear(H, self.ARC_HIDDEN), nn.ELU())
        self.arc_dep_mlp = nn.Sequential(nn.Linear(H, self.ARC_HIDDEN), nn.ELU())
        self.arc_biaffine = BiaffineAttention(self.ARC_HIDDEN, out_features=1)

        # Relation MLP
        self.rel_head_mlp = nn.Sequential(nn.Linear(H, self.REL_HIDDEN), nn.ELU())
        self.rel_dep_mlp = nn.Sequential(nn.Linear(H, self.REL_HIDDEN), nn.ELU())
        n_rels = len(self.deprel_vocab)
        self.rel_biaffine = BiaffineAttention(self.REL_HIDDEN, out_features=n_rels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple:
        """Returns (arc_scores, rel_scores)."""
        hidden = self.encode(input_ids, attention_mask)  # (B, L, H)

        h_arc_head = self.arc_head_mlp(hidden)  # (B, L, arc_h)
        h_arc_dep = self.arc_dep_mlp(hidden)
        arc_scores = self.arc_biaffine(h_arc_dep, h_arc_head)  # (B, L, L)

        h_rel_head = self.rel_head_mlp(hidden)  # (B, L, rel_h)
        h_rel_dep = self.rel_dep_mlp(hidden)
        rel_scores = self.rel_biaffine(h_rel_dep, h_rel_head)  # (B, L, L, n_rels)

        return arc_scores, rel_scores

    # ------------------------------------------------------------------
    def _get_extras(self) -> Dict[str, Any]:
        return {
            "deprel_vocab": self.deprel_vocab,
            "arc_head_mlp": self.arc_head_mlp.state_dict(),
            "arc_dep_mlp": self.arc_dep_mlp.state_dict(),
            "arc_biaffine": self.arc_biaffine.state_dict(),
            "rel_head_mlp": self.rel_head_mlp.state_dict(),
            "rel_dep_mlp": self.rel_dep_mlp.state_dict(),
            "rel_biaffine": self.rel_biaffine.state_dict(),
        }

    def _set_extras(self, extras: Dict[str, Any]) -> None:
        self.deprel_vocab = extras["deprel_vocab"]
        self._build_heads()
        self.arc_head_mlp.load_state_dict(extras["arc_head_mlp"])
        self.arc_dep_mlp.load_state_dict(extras["arc_dep_mlp"])
        self.arc_biaffine.load_state_dict(extras["arc_biaffine"])
        self.rel_head_mlp.load_state_dict(extras["rel_head_mlp"])
        self.rel_dep_mlp.load_state_dict(extras["rel_dep_mlp"])
        self.rel_biaffine.load_state_dict(extras["rel_biaffine"])

    # ------------------------------------------------------------------
    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
    ) -> "ParserModel":
        device = config.resolve_device()
        torch.manual_seed(config.seed)

        all_deprels = [t.deprel for s in train_sentences for t in s.regular_tokens()]
        deprel_vocab = _build_vocab(all_deprels)

        model = ParserModel(config, deprel_vocab).to(device)

        train_ds = ParserDataset(
            train_sentences, config.model_name, config.max_seq_length, deprel_vocab
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        arc_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        rel_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        best_score = -1.0

        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0.0

            progress = tqdm(train_loader, desc=f"[Parser] Epoch {epoch + 1}")
            for batch in progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                head_labels = batch["head_labels"].to(device)       # (B, L)
                deprel_labels = batch["deprel_labels"].to(device)   # (B, L)

                arc_scores, rel_scores = model(input_ids, attention_mask)
                # arc_scores: (B, L, L) — rows=dep, cols=head
                B, L, _ = arc_scores.shape

                arc_loss = arc_loss_fn(
                    arc_scores.view(B * L, L),
                    head_labels.view(B * L),
                )

                # For relation loss, gather scores at the gold head position
                # rel_scores: (B, L, L, n_rels)
                head_idx = head_labels.clamp(min=0)  # avoid -100 for gather
                head_expand = head_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, rel_scores.size(-1))
                rel_at_gold = rel_scores.gather(2, head_expand).squeeze(2)  # (B, L, n_rels)
                mask = deprel_labels != -100
                rel_loss = rel_loss_fn(
                    rel_at_gold.view(B * L, -1),
                    deprel_labels.view(B * L),
                )

                loss = arc_loss + rel_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    arc_loss=f"{arc_loss.item():.4f}",
                    rel_loss=f"{rel_loss.item():.4f}",
                    avg_loss=f"{total_loss / max(progress.n, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            avg_loss = total_loss / len(train_loader)

            metrics = evaluate_predictions(dev_sentences, model.predict(dev_sentences))
            print(f"  loss={avg_loss:.4f}")
            print_validation_metrics(metrics, ["UAS", "LAS"])
            score = average_f1(metrics, ["UAS", "LAS"])
            best_score = save_training_models(model, model_path, score, best_score)

        return model

    # ------------------------------------------------------------------
    def predict(self, sentences: List[Sentence]) -> List[Sentence]:
        device = self.config.resolve_device()
        self.eval()
        self.to(device)

        inv_deprel = {v: k for k, v in self.deprel_vocab.items()}
        hf_tok = self.hf_tokenizer
        results: List[Sentence] = []

        for sent in sentences:
            tokens = sent.regular_tokens()
            if not tokens:
                results.append(sent)
                continue
            forms = [t.form for t in tokens]

            encoding = hf_tok(
                forms,
                is_split_into_words=True,
                max_length=self.config.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                arc_scores, rel_scores = self(input_ids, attention_mask)

            arc_scores_sq = arc_scores.squeeze(0)  # (L, L)
            rel_scores_sq = rel_scores.squeeze(0)  # (L, L, n_rels)

            # Map sub-word positions back to word positions
            word_positions: Dict[int, int] = {}
            for i, wid in enumerate(word_ids):
                if wid is not None and wid not in word_positions:
                    word_positions[wid] = i  # first subword of word wid

            inv_word_positions = {pos: wid for wid, pos in word_positions.items()}

            # Greedy decoding guarantees a (near) well-formed tree, unlike
            # naive per-token argmax which can produce cycles.
            head_positions = greedy_decode(arc_scores_sq, word_positions)

            word_heads: Dict[int, int] = {}
            word_deprels: Dict[int, str] = {}
            for wid, pos in word_positions.items():
                pred_head_pos = head_positions[wid]
                # Position 0 (CLS) is the virtual ROOT -> CoNLL-U head 0.
                if pred_head_pos == 0:
                    word_heads[wid] = 0
                else:
                    # `greedy_decode` is guaranteed to only return ROOT (0)
                    # or one of the positions in `word_positions`, so this
                    # lookup should always succeed; a `None` here would
                    # indicate a decoder bug rather than expected input.
                    pred_head_wid = inv_word_positions.get(pred_head_pos)
                    assert pred_head_wid is not None, (
                        f"greedy_decode returned an unknown head position "
                        f"{pred_head_pos} for word {wid}"
                    )
                    word_heads[wid] = pred_head_wid + 1
                # Get relation
                rel_logits = rel_scores_sq[pos, pred_head_pos]  # (n_rels,)
                rel_id = rel_logits.argmax(-1).item()
                word_deprels[wid] = inv_deprel.get(rel_id, "_")  # type: ignore[arg-type]

            new_sent = Sentence(comments=sent.comments)
            for tok in sent.tokens:
                if tok.is_multiword() or tok.is_empty():
                    new_sent.tokens.append(tok)
                    continue
                tid = tok.id - 1
                new_tok = Token(
                    id=tok.id, form=tok.form, lemma=tok.lemma,
                    upos=tok.upos, xpos=tok.xpos, feats=tok.feats,
                    head=word_heads.get(tid, 0),
                    deprel=word_deprels.get(tid, "_"),
                    deps=tok.deps, misc=tok.misc,
                )
                new_sent.tokens.append(new_tok)
            results.append(new_sent)

        return results
