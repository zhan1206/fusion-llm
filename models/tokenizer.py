"""
Fusion Tokenizer - Unified tokenizer management

Handles the gap between GPT2 (vocab_size=50257) and Fusion's target vocab (100K).
Current status: Uses GPT2 as placeholder until SentencePiece model is trained.

Usage:
    from models.tokenizer import get_tokenizer
    tokenizer = get_tokenizer("gpt2")  # placeholder
    tokenizer = get_tokenizer("fusion", vocab_size=100000)  # future: SentencePiece

Author: Zhu Zizhan
Project: Fusion-LLM
License: Apache 2.0
"""

import json
import os
from pathlib import Path
from typing import Optional

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizer = None


# Fusion special tokens
FUSION_SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|start|>",
    "eos_token": "<|end|>",
    "think_tokens": ["<|think_depth_0|>", "<|think_depth_1|>", "<|think_depth_2|>", "<|think_depth_3|>"],
}


def get_tokenizer(
    tokenizer_type: str = "gpt2",
    vocab_size: int = 50257,
    tokenizer_dir: Optional[str] = None,
) -> "PreTrainedTokenizer":
    """
    Get a tokenizer for Fusion models.

    Args:
        tokenizer_type: "gpt2" (placeholder) or "fusion" (SentencePiece, if available)
        vocab_size: Target vocabulary size
        tokenizer_dir: Directory containing tokenizer files (for "fusion" type)

    Returns:
        A HuggingFace PreTrainedTokenizer instance

    Notes:
        - "gpt2" mode: Uses GPT2 BPE tokenizer (vocab_size=50257). This is a
          PLACEHOLDER. The model config should set vocab_size=50257 when using this.
        - "fusion" mode: Loads a SentencePiece tokenizer from tokenizer_dir.
          Requires tokenizer.model file to exist. Falls back to GPT2 if not found.
    """
    if AutoTokenizer is None:
        raise ImportError("transformers is required: pip install transformers")

    if tokenizer_type == "fusion":
        sp_model_path = None
        if tokenizer_dir:
            sp_model_path = Path(tokenizer_dir) / "tokenizer.model"
        else:
            # Try project root
            for candidate in ["tokenizers", ".", "data"]:
                p = Path(candidate) / "tokenizer.model"
                if p.exists():
                    sp_model_path = p
                    break

        if sp_model_path and sp_model_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(
                str(sp_model_path.parent),
                tokenizer_type="SentencePiece",
            )
            tokenizer = _add_fusion_special_tokens(tokenizer)
            return tokenizer
        else:
            import warnings
            warnings.warn(
                "Fusion SentencePiece tokenizer not found. "
                "Falling back to GPT2 tokenizer. "
                "Set vocab_size=50257 in model config to match.",
                UserWarning,
            )
            tokenizer_type = "gpt2"
            vocab_size = 50257

    if tokenizer_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer = _add_fusion_special_tokens(tokenizer)

        # Verify vocab size consistency
        actual_vocab = len(tokenizer)
        if actual_vocab != vocab_size:
            import warnings
            warnings.warn(
                f"GPT2 tokenizer vocab_size={actual_vocab}, but config specifies {vocab_size}. "
                f"Using actual tokenizer size ({actual_vocab}). "
                f"Update model config vocab_size to match.",
                UserWarning,
            )
        return tokenizer

    raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


def _add_fusion_special_tokens(tokenizer: "PreTrainedTokenizer") -> "PreTrainedTokenizer":
    """Add Fusion-specific special tokens to any tokenizer."""
    special_tokens_dict = {
        "pad_token": FUSION_SPECIAL_TOKENS["pad_token"],
        "additional_special_tokens": FUSION_SPECIAL_TOKENS["think_tokens"],
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def get_effective_vocab_size(tokenizer_type: str = "gpt2", requested_vocab: int = 100000) -> int:
    """
    Return the effective vocab size that should be used in model config.
    This ensures model embedding size matches the actual tokenizer.
    """
    if tokenizer_type == "gpt2":
        return 50257 + len(FUSION_SPECIAL_TOKENS["think_tokens"]) + 1  # ~50262
    if tokenizer_type == "fusion":
        return requested_vocab
    return requested_vocab


if __name__ == "__main__":
    print("[Fusion Tokenizer] Testing tokenizer creation...")
    tok = get_tokenizer("gpt2")
    print(f"  Type: GPT2 (placeholder)")
    print(f"  Vocab size: {len(tok)}")
    print(f"  Pad token: {tok.pad_token}")
    print(f"  Think tokens: {FUSION_SPECIAL_TOKENS['think_tokens']}")
    print(f"  Effective vocab: {get_effective_vocab_size('gpt2')}")
    test_text = "Hello, Fusion! <|think_depth_2|>"
    encoded = tok.encode(test_text)
    decoded = tok.decode(encoded)
    print(f"  Encode/decode test: '{test_text}' -> {encoded} -> '{decoded}'")
