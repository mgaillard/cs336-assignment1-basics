import numpy as np
import pytest
import torch

from cs336_basics.dataset import MemoryMappedDataset
from cs336_basics.llama2_tokenizer import TOKEN_DTYPE, decode_stories, encode_stories, load_tokenizer, split_stories
from cs336_basics.tokenizer import decode_file, encode_file

TEXT = "Once upon a time, there was a cat.\nThe end.\n<|endoftext|>\n\n\nTom saw 3 dogs 😀!\n<|endoftext|>\n"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def test_encode_file_wraps_each_story_in_bos_eos(tokenizer, tmp_path):
    text_file = tmp_path / "stories.txt"
    text_file.write_text(TEXT, encoding="utf-8")

    tokens = encode_file(tokenizer, text_file, chunk_size=16)

    stories = split_stories(TEXT)
    assert stories == ["Once upon a time, there was a cat.\nThe end.", "Tom saw 3 dogs 😀!"]
    expected = [1, *tokenizer.encode(stories[0]), 2, 1, *tokenizer.encode(stories[1]), 2]
    assert tokens.dtype == TOKEN_DTYPE
    assert tokens.tolist() == expected
    assert tokenizer.unk_id() not in expected

    tokens_file = tmp_path / "tokens.npy"
    np.save(tokens_file, tokens)
    assert decode_stories(tokenizer, tokens) == stories
    assert split_stories(decode_file(tokenizer, tokens_file)) == stories


def test_encode_stories_empty(tokenizer):
    tokens = encode_stories(tokenizer, split_stories("\n<|endoftext|>\n  \n"))
    assert tokens.dtype == TOKEN_DTYPE and tokens.size == 0


def test_dataset_reads_uint16(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.array([1, 500, 31999, 2, 1, 7], dtype=np.uint16))

    dataset = MemoryMappedDataset(path, context_length=3)
    inputs, targets = dataset[1]
    assert inputs.dtype == torch.long
    assert inputs.tolist() == [500, 31999, 2] and targets.tolist() == [31999, 2, 1]
