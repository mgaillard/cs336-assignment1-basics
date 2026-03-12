from tqdm import tqdm

import torch
from torch import nn

from cs336_basics.normalization import create_rms_norm
from cs336_basics.softmax import softmax
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.type_definitions import RMSNormType

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            eps: float = 1e-5,
            max_seq_len: int | None = None,
            theta: float | None = None,
            rms_normalization: RMSNormType = "pre-norm",
            device: torch.device=None,
            dtype:torch.dtype=None) -> None:
        """
        Construct the TransformerBlock module.
        Parameters:
        - vocab_size: int Size of the vocabulary. Necessary for determining the dimensionality of the token embedding matrix.
        - num_layers: int Number of Transformer blocks to stack.
        - d_model: int Dimensionality of the Transformer block inputs.
        - num_heads: int Number of heads to use in multi-head self-attention.
        - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        - eps: float = 1e-5 Epsilon value for numerical stability
        - max_seq_len: int Maximum sequence length for RoPE. If None, RoPE is not used.
        """
        super().__init__()

        # Parameters of the model
        self.num_layers = num_layers

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Blocks
        self.blocks = nn.ModuleDict()
        for i in range(num_layers):
            self.blocks[f"block_{i}"] = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                eps=eps,
                max_seq_len=max_seq_len,
                theta=theta,
                rms_normalization=rms_normalization,
                device=device,
                dtype=dtype,
            )

        # Final RMSNorm
        self.final_norm = create_rms_norm(rms_normalization, [d_model], eps=eps, device=device, dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Applies the Transformer Language Model to the input token indices.
        Parameters:
        - in_indices: torch.Tensor Input tensor of shape (batch_size, sequence_length) containing token indices.
        Returns:
        - torch.Tensor Output tensor of shape (batch_size, sequence_length, vocab_size) containing the predicted
          next-word distributions for each token.
        """
        batch_size, seq_length = in_indices.shape
        token_positions = torch.arange(seq_length, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)

        embedding = self.embedding(in_indices)

        z = embedding
        for block in range(self.num_layers):
            z = self.blocks[f"block_{block}"](z, token_positions)

        z_norm = self.final_norm(z)

        logits = self.output_proj(z_norm)

        return logits

    def generate(
        self,
        prompt: torch.Tensor,
        eos_token_id: int,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_steps: int = 32,
    ) -> torch.Tensor:
        """
        Generates a sequence of tokens autoregressively given an initial prompt.
        Uses nucleus sampling (top-p) and temperature to sample from the predicted next-word distributions.
        Used code from thepowerfuldeez/cs336_solutions.
        
        Parameters:
        - prompt: torch.Tensor Input integer tensor of shape (batch_size, prompt_length) containing token indices for the initial prompt.
        - eos_token_id: int The special token ID that indicates the end of the generated sequence.
        - top_p: float = 1.0 The cumulative probability threshold for nucleus sampling. Must be in the range (0, 1].
        - temperature: float = 1.0 The temperature for scaling the predicted next-word distributions. Must be > 0.
        - max_steps: int = 32 The maximum number of tokens to generate after the prompt.
        """
        input_seq = prompt

        with torch.inference_mode():
            for _ in tqdm(range(max_steps)):
                logits = self.forward(input_seq)
                if temperature == 0.0:
                    out = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                else:
                    probs = softmax(logits, dim=-1, temperature=temperature)[:, -1, :]
                    # nucleous sampling
                    if top_p < 1.0:
                        sorted_values, sorted_idx = probs.sort(-1, descending=True)
                        mask = sorted_values.cumsum(-1) <= top_p
                        mask[:, 0] = True
                        orig_mask = mask.gather(-1, sorted_idx.argsort(-1))
                        for i in range(len(probs)):
                            probs[i].masked_fill_(~orig_mask[i], 0.0)
                            probs[i] /= probs[i].sum(-1)
                    out = torch.multinomial(probs, 1)
                input_seq = torch.cat([input_seq, out], dim=-1)
                # Stop if all sequences in the batch have generated an EOS token
                if (out[-1:] == eos_token_id).all(dim=-1).item():
                    break

        return input_seq
