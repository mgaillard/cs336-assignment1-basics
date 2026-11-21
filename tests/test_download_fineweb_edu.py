import numpy as np
import pytest

from cs336_basics.download_fineweb_edu import (
    TokenWriter,
    is_validation_document,
    split_of,
    tokenize_splits,
)
from cs336_basics.llama2_tokenizer import TOKEN_DTYPE, decode_stories, load_tokenizer

VALID_FRACTION = 0.05
DOCS = [
    {"id": f"<urn:uuid:{i}>", "text": f"Document number {i} talks about science.\nIt has two lines."}
    for i in range(2000)
]


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def collect(tokenizer, docs, budgets, batch_size=7):
    out: dict[str, list[np.ndarray]] = {split: [] for split in budgets}
    for split, tokens in tokenize_splits(tokenizer, docs, budgets, VALID_FRACTION, batch_size):
        out[split].append(tokens)
    return out


def test_split_is_deterministic_and_roughly_sized():
    splits = [split_of(d["id"], VALID_FRACTION) for d in DOCS]
    assert splits == [split_of(d["id"], VALID_FRACTION) for d in DOCS]
    assert 0.02 * len(DOCS) < splits.count("valid") < 0.08 * len(DOCS)
    assert all((s == "valid") == is_validation_document(d["id"], VALID_FRACTION) for s, d in zip(splits, DOCS))


def test_documents_are_routed_to_their_split_with_bos_eos(tokenizer):
    docs = DOCS[:300] + [{"id": "empty", "text": "  \n"}]
    out = collect(tokenizer, docs, {"train": 10**9, "valid": 10**9})

    for split, chunks in out.items():
        expected = [d["text"] for d in DOCS[:300] if split_of(d["id"], VALID_FRACTION) == split]
        assert len(expected) > 0
        assert all(
            c.dtype == TOKEN_DTYPE and c[0] == tokenizer.bos_id() and c[-1] == tokenizer.eos_id() for c in chunks
        )
        assert decode_stories(tokenizer, np.concatenate(chunks)) == expected


def test_each_split_respects_its_own_budget(tokenizer):
    full = collect(tokenizer, DOCS[:300], {"train": 10**9, "valid": 10**9})
    valid_lengths = [len(c) for c in full["valid"]]
    budget = sum(valid_lengths[:2]) + 1  # room for 2 validation documents, not 3

    out = collect(tokenizer, DOCS[:300], {"train": 500, "valid": budget})

    assert 0 < sum(len(c) for c in out["train"]) <= 500
    assert [len(c) for c in out["valid"]] == valid_lengths[:2]


def test_only_requested_splits_are_produced(tokenizer):
    out = collect(tokenizer, DOCS[:300], {"valid": 10**9})
    assert set(out) == {"valid"}
    expected = [d["text"] for d in DOCS[:300] if split_of(d["id"], VALID_FRACTION) == "valid"]
    assert decode_stories(tokenizer, np.concatenate(out["valid"])) == expected


def test_token_writer_roundtrip(tmp_path):
    output_file = tmp_path / "out" / "tokens.npy"
    writer = TokenWriter(output_file)
    writer.write(np.arange(5, dtype=TOKEN_DTYPE))
    writer.write(np.arange(5, 12, dtype=TOKEN_DTYPE))

    assert writer.finalize() == 12

    loaded = np.load(output_file, mmap_mode="r")
    assert loaded.dtype == TOKEN_DTYPE
    assert loaded.tolist() == list(range(12))
    assert [p.name for p in output_file.parent.iterdir()] == ["tokens.npy"]
