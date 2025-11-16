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
        # The regex pattern for pre-tokenization
        self.regex_splitter = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # The inverse vocabulary for faster encoding
        self.inv_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        # A hashmap for the merge table for faster lookup
        self.merge_table: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if self.special_tokens:
            # Sort the special tokens by length in descending order to avoid sub-token issues
            self.special_tokens.sort(key=lambda x: len(x), reverse=True)
            for special_token in self.special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in self.inv_vocab:
                    index = len(vocab) # Warning: This assumes that the vocab is a dense mapping from 0 to N-1
                    self.vocab[index] = byte_encoded_special_token
                    self.inv_vocab = index

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        pass

    def _convert_bytes_to_token(self, byte_seq: bytes) -> int:
        """
        Convert a byte sequence to its corresponding token ID using the vocabulary.
        """
        if byte_seq in self.inv_vocab:
            return self.inv_vocab[byte_seq]
        else:
            raise ValueError(f"Byte sequence {byte_seq} not found in vocabulary")
        
    def _find_next_special_token(self, text: str, start_pos: int, end_pos: int) -> tuple[int, str]:
        """
        Find the next special token in the text between start_pos and end_pos.
        Returns the index of the next special token, or -1 if none found.
        """
        if not self.special_tokens:
            return -1
        
        next_special_token: str = ""
        next_special_token_pos = -1
        for special_token in self.special_tokens:
            index = text.find(special_token, start_pos, end_pos)
            if index != -1:
                if next_special_token_pos == -1 or index < next_special_token_pos:
                    next_special_token = special_token
                    next_special_token_pos = index
        
        return next_special_token_pos, next_special_token

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
                if current_pair in self.merge_table:
                    merge_rank = self.merge_table[current_pair]
                    if best_priority_rank == -1 or merge_rank < best_priority_rank:
                            best_priority_pair = current_pair
                            best_priority_rank = merge_rank

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
            token_ids.append(self._convert_bytes_to_token(token))

        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        tokens: list[int] = []

        pos = 0
        while pos < len(text):
            # By default, we intend to process until the end of the text
            endpos = len(text)

            # But we attempt to find the first special token in the text, and if found, we will process until there
            if self.special_tokens:
               index, special_token_str = self._find_next_special_token(text, pos, endpos)
               if index != -1:
                   if index == pos:
                        # Special token found at the current position
                        special_token_utf8_encoded = special_token_str.encode("utf-8")
                        # We add the special token ID directly to the output tokens
                        tokens.append(self._convert_bytes_to_token(special_token_utf8_encoded))
                        # We move the position forward by the length of the special token
                        pos += len(special_token_str)
                        continue
                   else:
                       # Special token found later in the text, process until the next special token
                       endpos = index

            # The regex splitter will split the text into pre-tokens (words, punctuation, spaces, etc.)
            for m in re.finditer(self.regex_splitter, text, pos=pos, endpos=endpos):
                # This is the pre-token found by the regex
                pre_token_str = m.group(0)
                # Convert the pre-token to UTF-8 bytes
                pre_token_utf8_encoded = pre_token_str.encode("utf-8")
                # Encode the pre-token using BPE and add the tokens to the output list
                tokens.extend(self._encode_pretoken(pre_token_utf8_encoded))

            pos = endpos
            
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings into a flat iterator of token IDs.
        """
        for text in iterable:
            yield from self.encode(text)
    
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
