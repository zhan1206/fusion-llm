#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer for Fusion models.

This creates the tokenizer.model file needed by Fusion's 100K vocabulary.
Training data should be a plain text file with one sentence per line.

Usage:
    python scripts/train_tokenizer.py --input data/tokenizer_train.txt --vocab_size 100000 --output tokenizers/

Requirements:
    pip install sentencepiece

Author: Zhu Zizhan
Project: Fusion-LLM
License: Apache 2.0
"""

import argparse
import os
import sys


def train_tokenizer(input_path: str, vocab_size: int, output_dir: str, model_type: str = "unigram"):
    """Train a SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("[ERROR] sentencepiece not installed. Run: pip install sentencepiece")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"[ERROR] Training data not found: {input_path}")
        print("Create a plain text file with one sentence per line.")
        print("For bilingual (zh+en) tokenizer, mix Chinese and English text.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "tokenizer")

    print(f"[Tokenizer] Training SentencePiece model...")
    print(f"  Input: {input_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")
    print(f"  Output: {output_dir}/")

    # Special tokens for Fusion
    control_symbols = [
        "<|pad|>", "<|start|>", "<|end|>",
        "<|think_depth_0|>", "<|think_depth_1|>",
        "<|think_depth_2|>", "<|think_depth_3|>",
    ]

    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,  # High coverage for CJK
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
        control_symbols=control_symbols,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        byte_fallback=True,  # Important for multilingual
        split_by_unicode_script=True,
        allow_whitespace_only_pieces=True,
    )

    model_path = os.path.join(output_dir, "tokenizer.model")
    vocab_path = os.path.join(output_dir, "tokenizer.vocab")

    print(f"\n[Done] Tokenizer trained successfully!")
    print(f"  Model: {model_path}")
    print(f"  Vocab: {vocab_path}")

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    print(f"  Actual vocab size: {sp.get_piece_size()}")

    # Test
    test_zh = "Fusion是一个开源大语言模型"
    test_en = "Fusion is an open-source language model"
    print(f"\n  Test encode (zh): {test_zh}")
    print(f"    -> {sp.encode(test_zh)}")
    print(f"  Test encode (en): {test_en}")
    print(f"    -> {sp.encode(test_en)}")


def create_sample_training_data(output_path: str, num_lines: int = 100000):
    """Create sample training data for tokenizer (for testing only)."""
    import random

    print(f"[Sample Data] Creating sample training data: {output_path}")
    samples_zh = [
        "人工智能是计算机科学的一个重要分支",
        "深度学习使用多层神经网络来模拟人脑",
        "自然语言处理帮助计算机理解人类语言",
        "机器学习使计算机能够从数据中学习",
        "大语言模型在文本生成任务中表现出色",
        "Transformer架构彻底改变了自然语言处理领域",
        "注意力机制是现代深度学习的核心组件",
        "预训练语言模型通过大规模语料学习语言知识",
        "微调技术使模型适应特定下游任务",
        "开源模型推动了人工智能技术的普及",
    ]
    samples_en = [
        "Artificial intelligence is a branch of computer science",
        "Deep learning uses multi-layer neural networks",
        "Natural language processing helps computers understand language",
        "Machine learning enables computers to learn from data",
        "Large language models excel at text generation tasks",
        "The Transformer architecture revolutionized NLP",
        "Attention mechanisms are core components of modern deep learning",
        "Pre-trained models learn language knowledge from large corpora",
        "Fine-tuning adapts models to specific downstream tasks",
        "Open-source models promote the democratization of AI",
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        for _ in range(num_lines):
            if random.random() > 0.5:
                f.write(random.choice(samples_zh) + "\n")
            else:
                f.write(random.choice(samples_en) + "\n")

    print(f"  Generated {num_lines} lines")


def main():
    parser = argparse.ArgumentParser(description="Train Fusion SentencePiece tokenizer")
    parser.add_argument("--input", type=str, default="data/tokenizer_train.txt",
                        help="Path to training data (one sentence per line)")
    parser.add_argument("--vocab_size", type=int, default=100000,
                        help="Vocabulary size (default: 100000)")
    parser.add_argument("--output", type=str, default="tokenizers/",
                        help="Output directory")
    parser.add_argument("--model_type", type=str, default="unigram",
                        choices=["unigram", "bpe"], help="Model type")
    parser.add_argument("--create_sample_data", action="store_true",
                        help="Create sample training data for testing")
    args = parser.parse_args()

    if args.create_sample_data:
        create_sample_training_data(args.input)
        return

    train_tokenizer(args.input, args.vocab_size, args.output, args.model_type)


if __name__ == "__main__":
    main()
