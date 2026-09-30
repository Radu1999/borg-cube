"""CoNLL-U reader and writer.

Supports:
- Regular tokens (integer IDs)
- Multi-word tokens (range IDs like "1-2")
- Empty nodes (decimal IDs like "1.1")
- FEATS as dict
- Sentence-level comments
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class Token:
    id: Union[int, str]       # int for regular; "1-2" for MWT; "1.1" for empty
    form: str
    lemma: str = "_"
    upos: str = "_"
    xpos: str = "_"
    feats: Union[Dict[str, str], str] = "_"  # parsed dict or "_"
    head: Optional[int] = None               # None when not set
    deprel: str = "_"
    deps: str = "_"
    misc: str = "_"
    space_after: bool = True

    # Convenience helpers
    def is_multiword(self) -> bool:
        return isinstance(self.id, str) and "-" in str(self.id)

    def is_empty(self) -> bool:
        return isinstance(self.id, str) and "." in str(self.id)

    def feats_str(self) -> str:
        if isinstance(self.feats, dict):
            if not self.feats:
                return "_"
            return "|".join(f"{k}={v}" for k, v in sorted(self.feats.items()))
        return self.feats if self.feats else "_"

    def __str__(self) -> str:
        head_str = str(self.head) if self.head is not None else "_"

        # Handle SpaceAfter in misc column: only write SpaceAfter=No
        misc_val = self.misc if self.misc != "_" else ""
        if not self.space_after:
            space_attr = "SpaceAfter=No"
            if misc_val:
                misc_final = f"{misc_val},{space_attr}" if "," not in misc_val else f"{misc_val},{space_attr}"
            else:
                misc_final = space_attr
        else:
            misc_final = misc_val

        return "\t".join([
            str(self.id),
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feats_str(),
            head_str,
            self.deprel,
            self.deps,
            misc_final,
        ])


@dataclass
class Sentence:
    tokens: List[Token] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)

    def regular_tokens(self) -> List[Token]:
        """Return only regular (non-MWT, non-empty) tokens."""
        return [t for t in self.tokens if not t.is_multiword() and not t.is_empty()]

    def __repr__(self) -> str:
        words = " ".join(t.form for t in self.regular_tokens())
        return f"<Sentence: {words!r}>"

    def __str__(self) -> str:
        lines = list(self.comments)
        lines.extend(str(t) for t in self.tokens)
        lines.append("")
        return "\n".join(lines)

    def to_conllu(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_feats(feats_str: str) -> Union[Dict[str, str], str]:
    if feats_str in ("_", ""):
        return "_"
    feats: Dict[str, str] = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            feats[k] = v
        else:
            feats[part] = ""
    return feats


def _parse_id(id_str: str) -> Union[int, str]:
    """Return int for simple IDs, str for ranges and decimals."""
    if re.match(r"^\d+$", id_str):
        return int(id_str)
    return id_str


def _parse_head(head_str: str) -> Optional[int]:
    if head_str == "_":
        return None
    try:
        return int(head_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_conllu(path: str) -> List[Sentence]:
    """Read a CoNLL-U file and return a list of Sentence objects."""
    sentences: List[Sentence] = []
    current_tokens: List[Token] = []
    current_comments: List[str] = []

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                current_comments.append(line)
            elif line == "":
                if current_tokens:
                    sentences.append(Sentence(tokens=current_tokens, comments=current_comments))
                current_tokens = []
                current_comments = []
            else:
                parts = line.split("\t")
                if len(parts) != 10:
                    continue  # skip malformed lines

                misc = parts[9]
                space_after = True
                if "SpaceAfter=No" in misc:
                    space_after = False
                elif "SpaceAfter=Yes" in misc:
                    space_after = True

                tok = Token(
                    id=_parse_id(parts[0]),
                    form=parts[1],
                    lemma=parts[2],
                    upos=parts[3],
                    xpos=parts[4],
                    feats=_parse_feats(parts[5]),
                    head=_parse_head(parts[6]),
                    deprel=parts[7],
                    deps=parts[8],
                    misc=misc,
                    space_after=space_after,
                )
                current_tokens.append(tok)

    # Handle file without trailing newline
    if current_tokens:
        sentences.append(Sentence(tokens=current_tokens, comments=current_comments))

    return sentences


def write_conllu(sentences: List[Sentence], path: str) -> None:
    """Write a list of Sentence objects to a CoNLL-U file."""
    with open(path, "w", encoding="utf-8") as fh:
        for sent in sentences:
            for comment in sent.comments:
                fh.write(comment + "\n")
            for token in sent.tokens:
                fh.write(str(token) + "\n")
            fh.write("\n")
