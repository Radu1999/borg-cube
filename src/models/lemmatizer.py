"""Lemmatizer model: edit-script classification."""
from __future__ import annotations
from dataclasses import replace
import os

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import (
    LemmatizerDataset,
    _build_vocab
)
from src.models.base import BorgBaseModel
from src.models.inference import sentence_batches
from src.models.evaluation import (
    average_f1,
    evaluate_predictions,
    print_validation_metrics,
)
from src.models.checkpoints import TrainingState


class LemmatizerModel(BorgBaseModel):
    """Classifies each token into a lemma edit-script."""

    def __init__(
        self,
        config: BorgConfig,
        upos_vocab: Optional[Dict[str, int]] = None,
        input_vocab: Optional[Dict[str, int]] = None,
        output_vocab: Optional[Dict[str, int]] = None,
    ):
        super().__init__(config, "lemmatizer")

        self.upos_vocab = upos_vocab or {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        self.output_vocab = output_vocab or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3
        }
        self.input_vocab = input_vocab or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3
        }

        self._build_heads()

    def _build_heads(self) -> None:
        upos_emb_dim = 32
        char_emb_dim = 64
        lstm_size = 200   

        self.upos_embedding = nn.Embedding(
            len(self.upos_vocab),
            upos_emb_dim,
            padding_idx=0,
        )
        
        self.input_embedding = nn.Embedding(
            len(self.input_vocab),
            char_emb_dim,
            padding_idx=0,
        )

        self.output_embedding = nn.Embedding(
            len(self.output_vocab),
            char_emb_dim,
            padding_idx=0,
        )
        
        self.output_logit = nn.Linear(
            lstm_size * 2,
            len(self.output_vocab)
        )

        self.input_encoder=nn.LSTM(
            input_size=upos_emb_dim+char_emb_dim+self.encoder.config.hidden_size,
            hidden_size=lstm_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.output_decoder=nn.LSTM(
            input_size=lstm_size * 2+char_emb_dim,
            hidden_size=lstm_size * 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.attention=nn.MultiheadAttention(
            embed_dim=lstm_size*2,
            num_heads=1,
            batch_first=True,
            kdim=lstm_size*2,
            vdim=lstm_size*2   
        )
        

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        upos_ids: torch.Tensor,
        input_char_ids: torch.Tensor,
        output_char_ids: Optional[torch.Tensor] = None,
        max_output_length: int = 200,
    ) -> torch.Tensor:
        hidden = self.encode(
            input_ids,
            attention_mask,
        )  # (B, L, H)

        upos_emb = self.upos_embedding(
            upos_ids
        )  # (B, L, E)

        concat = torch.cat(
            [hidden, upos_emb],
            dim=-1,
        )  # (B, L, H+E)

        concat=concat[:,1:-1,:]
        concat=concat.reshape((concat.shape[0]*concat.shape[1], -1))
        input_char_ids=input_char_ids.reshape((input_char_ids.shape[0]*input_char_ids.shape[1], input_char_ids.shape[2]))
        output_char_ids=output_char_ids.reshape((output_char_ids.shape[0]*output_char_ids.shape[1], output_char_ids.shape[2])) if output_char_ids is not None else None
        
        input_char_emb = self.input_embedding(input_char_ids)
        concat=concat.unsqueeze(1).repeat(1, input_char_emb.shape[1], 1)
        input_embedding= torch.cat([concat, input_char_emb], dim=-1)
        encoder_states, _=self.input_encoder(input_embedding)
        
        output_strings=["" for _ in range(input_char_ids.shape[0])]
        finished = [False for _ in output_strings]
        inverse_output_vocab = {value: key for key, value in self.output_vocab.items()}
        output_logits=[]
        last_output_embeddings=torch.zeros((input_char_ids.shape[0], 1, input_char_emb.shape[2]+encoder_states.shape[2]), device=encoder_states.device)
        decoder_hidden=None
        if output_char_ids is not None:
            output_char_ids=torch.clamp(output_char_ids, min=0)
            max_output_length=output_char_ids.shape[1]
            
        for step in range(max_output_length):
            decoder_output, decoder_hidden=self.output_decoder(last_output_embeddings, decoder_hidden)
            logits=self.output_logit(decoder_output)
            output_logits.append(logits)
            last_output_char_ids=logits.argmax(-1)
            if output_char_ids is not None:
                output_char_emb=self.output_embedding(output_char_ids[:,step:step+1])
            else:   
                output_char_emb=self.output_embedding(last_output_char_ids)
                for index, char_id in enumerate(last_output_char_ids[:, 0].tolist()):
                    if char_id == self.output_vocab["<EOS>"]:
                        finished[index] = True
                    elif not finished[index] and char_id not in (
                        self.output_vocab["<PAD>"],
                        self.output_vocab["<SOS>"],
                    ):
                        output_strings[index] += inverse_output_vocab.get(char_id, "")
                if all(finished):
                    break
            attention_output, _=self.attention(
                query=decoder_output,
                key=encoder_states,
                value=encoder_states,
            )
            last_output_embeddings=torch.cat([output_char_emb, attention_output], dim=-1)
        # from ipdb import set_trace; set_trace()
        return {
            "logits": torch.cat(output_logits, dim=1),
            "strings": output_strings
        }

    # ------------------------------------------------------------------
    def _get_extras(self) -> Dict[str, Any]:
        return {
            "upos_vocab": self.upos_vocab,
            "input_vocab": self.input_vocab,
            "output_vocab": self.output_vocab,
            "upos_embedding": self.upos_embedding.state_dict(),
            "input_embedding": self.input_embedding.state_dict(),
            "output_embedding": self.output_embedding.state_dict(),
            "input_encoder": self.input_encoder.state_dict(),
            "output_decoder": self.output_decoder.state_dict(),
            "output_logit": self.output_logit.state_dict(),
            "attention": self.attention.state_dict(),
        }

    def _set_extras(
        self,
        extras: Dict[str, Any],
    ) -> None:
        self.upos_vocab = extras["upos_vocab"]
        self.input_vocab = extras["input_vocab"]
        self.output_vocab = extras["output_vocab"]

        self._build_heads()

        for name in (
            "upos_embedding",
            "input_embedding",
            "output_embedding",
            "input_encoder",
            "output_decoder",
            "output_logit",
            "attention",
        ):
            getattr(self, name).load_state_dict(extras[name])

    # ------------------------------------------------------------------
    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
        *,
        resume: bool = False,
    ) -> "LemmatizerModel":
        device = config.resolve_device()
        device_type = torch.device(device).type

        torch.manual_seed(config.seed)

        all_upos = [
            t.upos
            for s in train_sentences
            for t in s.regular_tokens()
        ]
        all_input_chars = [
            c for s in train_sentences for t in s.regular_tokens() for c in t.form
        ]
        all_output_chars = [
            c for s in train_sentences for t in s.regular_tokens() for c in t.lemma 
        ]


        upos_vocab = _build_vocab(all_upos)
        input_vocab = _build_vocab(all_input_chars, initial_vocab={"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3})
        output_vocab = _build_vocab(all_output_chars, initial_vocab={"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3})

        if resume:
            model = LemmatizerModel.load(os.path.join(model_path, "last"), config).to(device)
            upos_vocab = model.upos_vocab
            input_vocab = model.input_vocab
            output_vocab = model.output_vocab  
        else:
            model = LemmatizerModel(
                config,
                upos_vocab,
                input_vocab,
                output_vocab,
            ).to(device)

        train_ds = LemmatizerDataset(
            train_sentences,
            config.model_name,
            config.max_seq_length,
            upos_vocab,
            input_vocab,
            output_vocab,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        total_steps = (
            len(train_loader)
            * config.num_epochs
        )

        warmup_steps = int(
            total_steps
            * config.warmup_ratio
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100
        )

        with TrainingState(
            model, model_path, optimizer, scheduler, train_loader, resume=resume,
        ) as state:
            for epoch in range(state.epoch, config.num_epochs):
                model.train()
                progress = state.progress(epoch, "Lemmatizer")
                for batch in progress:
                    input_ids = batch[
                        "input_ids"
                    ].to(device)

                    attention_mask = batch[
                        "attention_mask"
                    ].to(device)

                    upos_ids = batch[
                        "upos_ids"
                    ].to(device)

                    input_char_ids = batch[
                        "input_char_ids"
                    ].to(device)

                    output_char_ids = batch[
                        "output_char_ids"
                    ].to(device)

                    optimizer.zero_grad()

                    with torch.autocast(
                        device_type=device_type,
                        dtype=config.dtype,
                        enabled=device_type == "cuda",
                    ):
                        logits = model(
                            input_ids,
                            attention_mask,
                            upos_ids,
                            input_char_ids,
                            output_char_ids
                        )['logits']
                        loss = loss_fn(
                            logits.view(
                                -1,
                                logits.shape[-1]
                            ),
                            output_char_ids.view(-1),
                        )

                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                    optimizer.step()
                    scheduler.step()

                    state.step(loss.item())

                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg_loss=(f"{state.total_loss / state.batch:.4f}"),
                        lr=(
                            f"{scheduler.get_last_lr()[0]:.2e}"
                        ),
                    )

                avg_loss = state.total_loss / len(train_loader)

                metrics = evaluate_predictions(
                    dev_sentences,
                    model.predict(dev_sentences),
                )

                print(f"  loss={avg_loss:.4f}")

                print_validation_metrics(
                    metrics,
                    ["LEMMA"],
                )

                score = average_f1(
                    metrics,
                    ["LEMMA"],
                )

                state.finish_epoch(score)

        return model

    # ------------------------------------------------------------------
    def predict(
        self,
        sentences: List[Sentence],
        *,
        show_progress: bool = True,
    ) -> List[Sentence]:


        device = self.config.resolve_device()
        device_type = torch.device(device).type

        self.eval()
        self.to(device)
        results = list(sentences)
        for batch in sentence_batches(self, sentences, show_progress=show_progress):
            if batch.input_char_ids is None:
                raise RuntimeError("Lemmatizer inference requires input character IDs")
            upos_ids = torch.zeros_like(batch.input_ids)
            for row, word_ids in enumerate(batch.word_ids):
                token_upos = [
                    self.upos_vocab.get(token.upos, self.upos_vocab["<UNK>"])
                    for token in batch.tokens[row]
                ]
                upos_ids[row, :len(word_ids)] = torch.tensor(
                    [token_upos[wid] if wid is not None else 0 for wid in word_ids],
                    dtype=torch.long,
                )

            with torch.inference_mode(), torch.autocast(
                device_type=device_type,
                dtype=self.config.dtype,
                enabled=device_type == "cuda",
            ):
                lemmas = self(
                    batch.input_ids.to(device),
                    batch.attention_mask.to(device),
                    upos_ids.to(device),
                    batch.input_char_ids.to(device),
                )["strings"]

            for row, index in enumerate(batch.indices):
                sent = sentences[index]
                sequence_width = batch.input_ids.shape[1] - 2
                word_lemmas = {
                    word_id: lemmas[row * sequence_width + position - 1]
                    for word_id, position in batch.word_positions[row].items()
                }
                new_sent = Sentence(comments=sent.comments)
                word_id = 0
                for tok in sent.tokens:
                    if tok.is_multiword() or tok.is_empty():
                        new_sent.tokens.append(tok)
                        continue
                    new_sent.tokens.append(
                        replace(tok, lemma=word_lemmas.get(word_id, tok.lemma))
                    )
                    word_id += 1
                results[index] = new_sent

        return results
