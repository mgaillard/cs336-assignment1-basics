"""
Llama 2 SentencePiece tokenizer (32,000 tokens) used to tokenize the TinyStories datasets.

The tokenizer model is vendored in cs336_basics/llama2/ (see LICENSE and NOTICE there).
Each story is encoded as `<s> story </s>` (BOS = 1, EOS = 2). In the raw text files, stories are
separated by `<|endoftext|>`, which is not part of the Llama 2 vocabulary.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import sentencepiece as spm

TOKENIZER_MODEL_PATH = Path(__file__).resolve().parent / "llama2" / "tokenizer.model"
STORY_SEPARATOR = "<|endoftext|>"
# Token ids are < 32,000, so uint16 is enough to store them.
TOKEN_DTYPE = np.uint16


def load_tokenizer(model_path: str | Path = TOKENIZER_MODEL_PATH) -> spm.SentencePieceProcessor:
    """Load the Llama 2 SentencePiece tokenizer."""
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    assert tokenizer.vocab_size() <= np.iinfo(TOKEN_DTYPE).max + 1, "vocabulary does not fit in TOKEN_DTYPE"
    return tokenizer


def split_stories(text: str) -> list[str]:
    """Split raw text on `<|endoftext|>`, strip each story and drop empty ones."""
    stories = (story.strip() for story in text.split(STORY_SEPARATOR))
    return [story for story in stories if story]


def encode_stories(tokenizer: spm.SentencePieceProcessor, stories: list[str]) -> np.ndarray:
    """Encode stories as one flat array `<s> story_1 </s> <s> story_2 </s> ...` of TOKEN_DTYPE."""
    if not stories:
        return np.empty(0, dtype=TOKEN_DTYPE)
    encoded = tokenizer.encode(stories, add_bos=True, add_eos=True, num_threads=-1)
    return np.concatenate([np.asarray(ids, dtype=TOKEN_DTYPE) for ids in encoded])


def decode_stories(tokenizer: spm.SentencePieceProcessor, tokens: Iterable[int]) -> list[str]:
    """Decode a flat array of token ids into stories, splitting on EOS (BOS/EOS are dropped)."""
    tokens = np.asarray(tokens).astype(np.int64)
    eos_positions = np.flatnonzero(tokens == tokenizer.eos_id())
    segments = np.split(tokens, eos_positions + 1)
    segments = [segment.tolist() for segment in segments if segment.size > 0]
    return [story for story in tokenizer.decode(segments) if story]
