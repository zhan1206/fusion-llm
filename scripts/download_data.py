"""
下载预训练数据脚本

提供公开数据集的下载和预处理，用于 Fusion-LLM 预训练。
支持的数据源:
  - wikitext-103: Wikipedia 文本 (500MB)
  - openwebtext: OpenWebText 子集 (自定义大小)
  - custom: 用户自定义文本文件

使用方式:
    python scripts/download_data.py --source wikitext --output data/pretrain/
    python scripts/download_data.py --source custom --input my_corpus.txt --output data/pretrain/
"""
import argparse
import os
import sys
from pathlib import Path


def download_wikitext(output_dir, subset="wikitext-103-raw-v1"):
    """Download WikiText-103 dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package required. Install with: pip install datasets")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DOWNLOAD] Loading {subset}...")
    dataset = load_dataset("wikitext", subset)
    
    for split in dataset:
        text = "\n".join(dataset[split]["text"])
        out_file = output_dir / f"wikitext_{split}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  {split}: {len(text):,} chars -> {out_file}")
    
    return True


def download_openwebtext(output_dir, num_shards=1, max_samples=None):
    """Download OpenWebText subset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package required. Install with: pip install datasets")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DOWNLOAD] Loading OpenWebText (shards={num_shards})...")
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    
    count = 0
    out_file = output_dir / "openwebtext.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            f.write(item["text"] + "\n\n")
            count += 1
            if count % 10000 == 0:
                print(f"  Downloaded {count:,} documents...")
    
    print(f"  Total: {count:,} documents -> {out_file}")
    return True


def prepare_custom(input_path, output_dir):
    """Prepare custom text data for training."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return False
    
    # Copy and clean text
    import re
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    # Basic cleaning: remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    out_file = output_dir / "custom_data.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"  Custom data: {len(text):,} chars -> {out_file}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare training data")
    parser.add_argument("--source", choices=["wikitext", "openwebtext", "custom"],
                        required=True, help="Data source")
    parser.add_argument("--output", default="data/pretrain", help="Output directory")
    parser.add_argument("--input", help="Input file (for custom source)")
    parser.add_argument("--max-samples", type=int, help="Max samples (for openwebtext)")
    args = parser.parse_args()
    
    if args.source == "wikitext":
        download_wikitext(args.output)
    elif args.source == "openwebtext":
        download_openwebtext(args.output, max_samples=args.max_samples)
    elif args.source == "custom":
        if not args.input:
            print("[ERROR] --input required for custom source")
            sys.exit(1)
        prepare_custom(args.input, args.output)
