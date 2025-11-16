from collections.abc import Iterable, Iterator
import regex as re

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or {}
        self.regex_splitter = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        pass

    def _encode_pretoken(self, input_pretoken: bytes) -> list[int]:
        """
        Encode a single pre-token (string) into a sequence of token IDs using BPE merges.
        """
        current_tokens: list[bytes] = [t.to_bytes() for t in input_pretoken]  # Start with each byte as its own token
        # Recursively apply BPE merges to get final tokens
        while True:
            # Search for the highest-priority merge that can be applied
            best_priority_pair = -1
            best_priority_rank = -1
            for i in range(len(current_tokens) - 1):
                # Concatenate current_tokens[i] and current_tokens[i + 1] to bytes
                current_pair = (current_tokens[i], current_tokens[i + 1])
                # Search for current_pair in self.merges
                for merge_rank, merge_pair in enumerate(self.merges):
                    if current_pair == merge_pair:
                        if best_priority_rank == -1 or merge_rank < best_priority_rank:
                            best_priority_pair = current_pair
                            best_priority_rank = merge_rank
                        break  # No need to check further merges because merges are sorted by priority

            if best_priority_rank == -1:
                break  # No more merges can be applied

            # Apply the best merge found
            new_tokens: list[bytes] = []
            i = 0
            current_tokens_len = len(current_tokens)
            while i < current_tokens_len: # We need to use a while loop instead of a for loop because we may skip tokens
                if i < len(current_tokens) - 1 and (current_tokens[i], current_tokens[i + 1]) == best_priority_pair:
                    # Merge the pair
                    new_tokens.append(current_tokens[i] + current_tokens[i + 1])
                    i += 2  # Skip the next token as it has been merged
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
            
            # Update current_tokens with the newly merged tokens
            current_tokens = new_tokens

        # convert final tokens to token IDs using self.vocab
        token_ids: list[int] = []
        for token in current_tokens:
            # Find the token ID in the vocabulary
            for token_id, vocab_token in self.vocab.items():
                if token == vocab_token:
                    token_ids.append(token_id)
                    break
            else:
                # Token not found in vocabulary, handle unknown token case
                raise ValueError(f"Token {token} not found in vocabulary")

        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        tokens: list[int] = []

        # TODO: remove special tokens from text if any are present

        # The regex splitter will split the text into pre-tokens (words, punctuation, spaces, etc.)
        for m in re.finditer(self.regex_splitter, text):
            # Convert the pre-token to UTF-8 bytes
            pre_token_str = m.group(0)
            pre_token_utf8_encoded = pre_token_str.encode("utf-8")
            # Encode the pre-token using BPE and add the tokens to the output list
            tokens.extend(self._encode_pretoken(pre_token_utf8_encoded))
        
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings into a flat iterator of token IDs.
        """
        pass
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a string.
        """
        byte_array = bytearray()
        for token_id in ids:
            if token_id in self.vocab:
                byte_array.extend(self.vocab[token_id])
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")
        return byte_array.decode("utf-8", errors="ignore")
