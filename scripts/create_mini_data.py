"""
创建 Fusion Mini 训练数据

生成极简的训练数据（字符级），用于验证完整训练流程。

使用方法：
    python tests/create_mini_data.py
    
    # 会生成 data/mini_data.json

作者：朱子瞻
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import json
import random
from pathlib import Path


def create_mini_dataset(output_path: str, num_samples: int = 100):
    """
    创建 mini 训练数据集
    
    参数：
        output_path: 输出文件路径
        num_samples: 样本数量
    """
    print("[数据] 创建 mini 训练数据集...")
    print(f"   输出路径：{output_path}")
    print(f"   样本数量：{num_samples}")
    
    data = []
    
    # 预定义一些简单的中文和英文句子
    chinese_samples = [
        ("你好", "你好！我是 Fusion Mini 模型。"),
        ("什么是人工智能", "人工智能是计算机科学的一个分支，致力于创建智能机器。"),
        ("解释机器学习", "机器学习是人工智能的子领域，使计算机能够从数据中学习。"),
        ("深度学习是什么", "深度学习是机器学习的一个分支，使用多层神经网络模拟人脑。"),
        ("什么是自然语言处理", "自然语言处理是AI的一个分支，帮助计算机理解人类语言。"),
        ("Python 有什么特点", "Python 是一种简单易学、功能强大的编程语言。"),
        ("如何学习编程", "学习编程需要理论与实践相结合，多写代码多思考。"),
        ("什么是大数据", "大数据是指规模巨大、类型多样的数据集合。"),
        ("云计算的优势", "云计算提供弹性扩展、成本节约、易于维护等优势。"),
        ("区块链的原理", "区块链是一种分布式账本技术，确保数据不可篡改。"),
    ]
    
    english_samples = [
        ("Hello", "Hello! I am Fusion Mini model."),
        ("What is AI", "AI stands for Artificial Intelligence."),
        ("Explain machine learning", "Machine learning is a subset of AI."),
        ("What is deep learning", "Deep learning uses neural networks with many layers."),
        ("What is NLP", "NLP helps computers understand human language."),
        ("Python features", "Python is simple, powerful, and versatile."),
        ("How to learn coding", "Practice coding regularly and build projects."),
        ("What is big data", "Big data refers to extremely large datasets."),
        ("Benefits of cloud computing", "Cloud computing offers scalability and cost savings."),
        ("How blockchain works", "Blockchain is a distributed ledger technology."),
    ]
    
    # 生成样本
    for i in range(num_samples):
        # 随机选择中文或英文
        if random.random() > 0.5:
            prompt, response = random.choice(chinese_samples)
        else:
            prompt, response = random.choice(english_samples)
        
        # Assign think_rank based on content depth
        if any(kw in prompt for kw in ["Prove", "Derive", "Analyze", "\u8bc1\u660e", "\u63a8\u5bfc", "\u5206\u6790"]):
            think_rank = 3
        elif any(kw in prompt for kw in ["Explain", "How", "Why", "\u89e3\u91ca", "\u5982\u4f55", "\u4e3a\u4ec0\u4e48"]):
            think_rank = 2
        elif any(kw in prompt for kw in ["Write", "Implement", "\u5199", "\u5b9e\u73b0"]):
            think_rank = 1
        else:
            think_rank = 0

        data.append({
            "prompt": prompt,
            "response": response,
            "think_rank": think_rank,
        })
    
    # 保存为 JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("[完成] 数据集创建成功！")
    print(f"   文件路径：{output_path}")
    print(f"   样本数量：{len(data)}")
    
    # 显示几个示例
    print("\n[示例] 数据示例：")
    for i, item in enumerate(data[:3]):
        print(f"   [{i+1}] Prompt: {item['prompt']}")
        print(f"       Response: {item['response'][:50]}...")
        print()


def main():
    print("=" * 60)
    print("创建 Fusion Mini 训练数据")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # 生成训练数据
    output_path = output_dir / "mini_data.json"
    create_mini_dataset(output_path, num_samples=100)
    
    print(f"\n[完成] 数据创建完成！")
    print(f"\n下一步：")
    print(f"   1. 检查数据文件：{output_path}")
    print(f"   2. 开始训练：python train/train_mini.py")
    print(f"   3. 或者运行完整测试：python tests/run_tests.py")


if __name__ == "__main__":
    main()
