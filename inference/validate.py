#!/usr/bin/env python3
"""Check the C++ inference binaries against the Python reference (tiktoken + PyTorch float32 on CPU).

  1. tokenizer: token ids of many UTF-8 strings equal tiktoken's, invalid UTF-8 is rejected;
  2. logits:    the logits following a prompt match the PyTorch model (max abs difference);
  3. greedy:    greedy generation produces the same token ids as the PyTorch model.

Run from the inference/ folder after `make` and `make convert`:
    python validate.py --checkpoint ../outputs/<run>/checkpoint_best_model.pt
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tiktoken
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_basics.config_schema import ModelConfig  # noqa: E402
from cs336_basics.transformer_lm import TransformerLM  # noqa: E402

HERE = Path(__file__).resolve().parent
EOS = 50256

TOKENIZER_CASES = [
    "Once upon a time",
    "Hello, world!",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\n\r\n  mixed",
    "He said it's not what we'd've thought; they're sure I'll go.",
    "'s 'S 're 'RE ' ' ''",
    "numbers 12345 and 3.14159, 1,000,000 x2 a1b2",
    "punctuation!!! ...??? --- ((())) []{} <|endoftext|> <|fim|>",
    "naïve café résumé façade Ångström",
    "日本語のテキスト 中文文本 한국어 텍스트",
    "emoji 😀🎉 and \u2603 snowman, \u00a0nbsp, \u2003em space, \u3000ideographic space",
    "Привет, мир! Ελληνικά العربية עברית",
    "a" * 300,
    "word " * 120,
    "\n\n\n\n",
    " ",
    "   ",
    "x\n\n  y",
    "end with spaces   ",
    "\u0301combining marks e\u0301 a\u0308",
    "ÿ\u0080\u009f\u00ad control-ish",
    "１２３ fullwidth Ａ Ｂ ² ³ ½ Ⅷ",
    "https://example.com/path?query=1&b=2#frag",
    "def f(x):\n    return x**2  # square\n",
]

FUZZ_ALPHABET = list("abcXYZ 0123456789.,!?'\"-_\n\t") + [" ", " ", " ", "\n"] + list("éüñ中日한😀\u00a0\u2003\u3000²½'s'")


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, **kwargs)


def check_tokenizer(binary: Path, encoding, models: Path) -> int:
    rng = random.Random(0)
    cases = list(TOKENIZER_CASES)
    for _ in range(150):
        cases.append("".join(rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(1, 40))))
    corpus = HERE.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    if corpus.exists():
        with open(corpus, encoding="utf-8") as f:
            text = f.read(400_000)
        cases += [text[i : i + 1500] for i in range(0, 60_000, 6_000)]

    failures = 0
    for case in cases:
        proc = run([str(binary), "--tokenizer", str(models / "tokenizer.bin"), "--print-tokens", "--prompt", case])
        got = [int(t) for t in proc.stdout.split()]
        expected = encoding.encode(case, disallowed_special=())
        if proc.returncode != 0 or got != expected:
            failures += 1
            print(f"  TOKENIZER MISMATCH for {case[:60]!r}\n    expected {expected[:20]}\n    got      {got[:20]} {proc.stderr[:200]!r}")
    # Invalid UTF-8 must be rejected.
    for bad in (b"abc\xff", b"\xc0\xaf", b"\xed\xa0\x80", b"ok \xe2\x82"):
        proc = run([str(binary), "--tokenizer", str(models / "tokenizer.bin"), "--print-tokens", "--prompt", os.fsdecode(bad)])
        if proc.returncode == 0:
            failures += 1
            print(f"  invalid UTF-8 {bad!r} was accepted")
    print(f"  tokenizer: {len(cases)} strings + 4 invalid inputs, {failures} failures")
    return failures


def load_reference(checkpoint: Path, config_path: Path) -> TransformerLM:
    raw = yaml.safe_load(config_path.read_text())
    config = ModelConfig(**(raw["model"] if "model" in raw else raw))
    model = TransformerLM(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict({k.removeprefix("_orig_mod."): v.float() for k, v in state.items()})
    return model.eval()


@torch.inference_mode()
def reference_greedy(model: TransformerLM, prompt: list, steps: int) -> list:
    seq = torch.tensor([prompt])
    steps = min(steps, model.config.max_seq_len - len(prompt))
    for _ in range(steps):
        token = model(seq)[:, -1, :].argmax(-1, keepdim=True)
        seq = torch.cat([seq, token], dim=-1)
        if token.item() == EOS:
            break
    return seq[0].tolist()


def check_model(binary: Path, encoding, model: TransformerLM, models: Path, steps: int, tolerance: float) -> int:
    prompts = ["Once", "Once upon a time, there was a little girl named Lily.", "The dog and the cat", "Tom's big red ball"]
    failures = 0
    base = [str(binary), "--model", str(models / "model.bin"), "--tokenizer", str(models / "tokenizer.bin")]
    for prompt in prompts:
        ids = encoding.encode(prompt)
        with torch.inference_mode():
            expected = model(torch.tensor([ids]))[0, -1].numpy()
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "logits.bin"
            proc = run(base + ["--prompt", prompt, "--dump-logits", str(dump)])
            if proc.returncode != 0:
                print(f"  {binary} failed: {proc.stderr.decode()[:300]}")
                failures += 1
                continue
            got = np.fromfile(dump, dtype="<f4")
        diff = float(np.abs(got - expected).max())
        ok = diff < tolerance and got.argmax() == expected.argmax()
        failures += not ok
        print(f"  logits {'OK ' if ok else 'FAIL'} max|diff|={diff:.2e} argmax={int(got.argmax())} for {prompt[:40]!r}")

        proc = run(base + ["--prompt", prompt, "--max_steps", str(steps), "--temperature", "0", "--show-ids"])
        lines = [l for l in proc.stderr.decode().splitlines() if l.startswith("ids: ")]
        got_ids = [int(t) for t in lines[0][5:].split()] if lines else []
        want_ids = reference_greedy(model, ids, steps)
        ok = got_ids == want_ids
        failures += not ok
        first_diff = next((i for i, (a, b) in enumerate(zip(got_ids, want_ids)) if a != b), min(len(got_ids), len(want_ids)))
        print(f"  greedy {'OK ' if ok else 'FAIL'} {len(want_ids) - len(ids)} tokens" + ("" if ok else f", first difference at {first_diff}"))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE.parent / "configs" / "gpt_small.yaml")
    parser.add_argument("--models", type=Path, default=HERE / "models")
    parser.add_argument("--binaries", nargs="*", type=Path, default=[HERE / "build" / t / "cs336_infer" for t in ("x86_sse", "x64_avx2")])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    encoding = tiktoken.get_encoding("gpt2")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = load_reference(args.checkpoint, args.config)
    total = 0
    for binary in args.binaries:
        if not binary.exists():
            print(f"skipping {binary} (not built)")
            continue
        print(f"== {binary}")
        total += check_tokenizer(binary, encoding, args.models)
        total += check_model(binary, encoding, model, args.models, args.steps, args.tolerance)
    print("ALL CHECKS PASSED" if total == 0 else f"{total} CHECK(S) FAILED")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
