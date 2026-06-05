"""
创建伪真实训练数据（用于实际模型训练）
使用简单的英文句子作为训练数据
"""
import os
import torch
from pathlib import Path

def create_training_data():
    """创建训练数据"""
    print("[DATA] 创建训练数据...")
    
    # 简单的英文句子（用于训练）
    sentences = [
        "The cat sits on the mat.",
        "A dog runs in the park.",
        "Birds fly in the sky.",
        "Fish swim in the sea.",
        "Children play in the garden.",
        "The sun is shining brightly.",
        "It is raining heavily today.",
        "Snow falls in winter.",
        "Flowers bloom in spring.",
        "Leaves fall in autumn.",
        "I love reading books.",
        "She writes a letter.",
        "He cooks dinner for us.",
        "We watch a movie together.",
        "They sing a beautiful song.",
        "The car moves fast on the road.",
        "A plane flies in the air.",
        "Ships sail on the ocean.",
        "Trains travel across the country.",
        "Bicycles are good for health.",
        "Apple is a delicious fruit.",
        "Water is essential for life.",
        "The house has a big garden.",
        "Music brings joy to people.",
        "Learning is a lifelong journey.",
    ]
    
    # 重复句子以增加数据量
    all_sentences = sentences * 10  # 250 个句子
    
    # 保存到文件
    data_path = Path("data/training_data.txt")
    data_path.parent.mkdir(exist_ok=True)
    
    with open(data_path, "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(sentence + "\n")
    
    print(f"   保存路径: {data_path}")
    print(f"   句子数量: {len(all_sentences)}")
    print("   训练数据创建成功")
    print()
    
    return data_path


def create_tokenizer_from_data(data_path):
    """从训练数据创建 tokenizer"""
    print("[TOKENIZER] 创建 tokenizer...")
    
    # 简单字符级 tokenizer（用于演示）
    # 在实际应用中，应该使用 SentencePiece 或 BPE
    
    # 读取所有文本
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 创建字符词汇表
    chars = sorted(list(set(text)))
    vocab_size = len(chars) + 3  # +3 for [PAD], [UNK], [CLS]
    
    # 保存词汇表
    vocab_path = Path("tokenizers/char_vocab.txt")
    vocab_path.parent.mkdir(exist_ok=True)
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("[PAD]\n")
        f.write("[UNK]\n")
        f.write("[CLS]\n")
        for char in chars:
            f.write(char + "\n")
    
    print(f"   词汇表大小: {vocab_size}")
    print(f"   保存路径: {vocab_path}")
    print("   Tokenizer 创建成功")
    print()
    
    return vocab_path, vocab_size


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 创建训练数据")
    print("=" * 60)
    print()
    
    # 1. 创建训练数据
    data_path = create_training_data()
    
    # 2. 创建 tokenizer
    vocab_path, vocab_size = create_tokenizer_from_data(data_path)
    
    print("[DONE] 训练数据准备完成")
    print(f"   训练数据: {data_path}")
    print(f"   Tokenizer: {vocab_path}")
    print(f"   词汇表大小: {vocab_size}")
