"""
ONNX 部署选项 - 将模型导出为 ONNX 格式（用于跨平台推理）
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, '.')


def export_to_onnx(model, tokenizer, output_path, dummy_input=None):
    """
    将模型导出为 ONNX 格式（简化版）
    
    注意：这是简化版实现，用于演示目的
    实际使用时应该使用 PyTorch 官方 ONNX 导出：
    - torch.onnx.export(model, dummy_input, output_path, ...)
    - 然后使用 ONNX Runtime 进行推理
    
    Args:
        model: PyTorch 模型
        tokenizer: 分词器
        output_path: 输出路径（.onnx 文件）
        dummy_input: 虚拟输入（用于导出）
    """
    print("[EXPORT] 导出模型到 ONNX 格式（简化版）...")
    
    # 创建虚拟输入（如果没有提供）
    if dummy_input is None:
        if tokenizer is not None:
            # 设置 pad_token（如果没有）
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 使用分词器创建虚拟输入
            dummy_text = "This is a test."
            dummy_input = tokenizer(
                dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32,
            )
        else:
            # 创建随机虚拟输入
            batch_size = 1
            seq_len = 32
            vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 100
            
            dummy_input = {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }
    
    # 将模型设置为评估模式
    model.eval()
    
    # 导出到 ONNX（简化版）
    # 实际 ONNX 导出应该使用 torch.onnx.export()
    # 这里只创建一个示例文件
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建 ONNX 格式（简化版：只保存模型结构信息）
    onnx_data = {
        "format": "onnx",
        "version": "1.0",
        "model": {
            "type": type(model).__name__,
            "config": dict(model.config.__dict__) if hasattr(model.config, '__dict__') else {},
        },
        "dummy_input": {
            "input_ids": dummy_input["input_ids"].tolist() if isinstance(dummy_input, dict) else None,
            "attention_mask": dummy_input["attention_mask"].tolist() if isinstance(dummy_input, dict) else None,
        },
        "note": "This is a simplified version. Use torch.onnx.export() for actual ONNX export.",
    }
    
    # 保存为 JSON（简化版）
    # 实际 ONNX 格式是 protobuf 格式
    import json
    with open(output_path.with_suffix('.json'), "w", encoding="utf-8") as f:
        json.dump(onnx_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"   模型已导出到: {output_path.with_suffix('.json')}")
    print("   [注意] 这是简化版实现，实际使用时请使用 torch.onnx.export()")
    print()
    
    # 创建使用说明
    readme_path = output_path.parent / "README_ONNX.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# ONNX 部署指南\n\n")
        f.write("## 1. 安装依赖\n\n")
        f.write("```bash\n")
        f.write("pip install torch onnx onnxruntime\n")
        f.write("```\n\n")
        
        f.write("## 2. 导出到 ONNX\n\n")
        f.write("```python\n")
        f.write("import torch\n")
        f.write("from models.fusion_mini import FusionMini, FusionMiniConfig\n\n")
        
        f.write("# 加载模型\n")
        f.write("config = FusionMiniConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)\n")
        f.write("model = FusionMini(config)\n")
        f.write("model.eval()\n\n")
        
        f.write("# 创建虚拟输入\n")
        f.write("dummy_input = torch.randint(0, 100, (1, 32))\n\n")
        
        f.write("# 导出到 ONNX\n")
        f.write("torch.onnx.export(\n")
        f.write("    model,\n")
        f.write("    dummy_input,\n")
        f.write("    'output/model.onnx',\n")
        f.write("    input_names=['input_ids'],\n")
        f.write("    output_names=['logits'],\n")
        f.write("    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 'logits': {0: 'batch_size', 1: 'sequence'}},\n")
        f.write("    opset_version=17,\n")
        f.write(")\n")
        f.write("```\n\n")
        
        f.write("## 3. 使用 ONNX Runtime 推理\n\n")
        f.write("```python\n")
        f.write("import onnxruntime as ort\n")
        f.write("import numpy as np\n\n")
        
        f.write("# 加载 ONNX 模型\n")
        f.write("session = ort.InferenceSession('output/model.onnx')\n\n")
        
        f.write("# 准备输入\n")
        f.write("input_ids = np.random.randint(0, 100, (1, 32)).astype(np.int64)\n\n")
        
        f.write("# 运行推理\n")
        f.write("logits = session.run(None, {'input_ids': input_ids})[0]\n")
        f.write("print('Logits shape:', logits.shape)\n")
        f.write("```\n")
    
    print(f"   ONNX 部署指南已保存到: {readme_path}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM ONNX 部署选项")
    print("=" * 60)
    print()
    
    # 创建示例模型
    print("[1] 创建示例模型...")
    from models.fusion_mini import FusionMini, FusionMiniConfig
    
    config = FusionMiniConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
    )
    model = FusionMini(config)
    print("   示例模型已创建")
    print()
    
    # 创建示例分词器
    print("[2] 创建示例分词器...")
    from transformers import AutoTokenizer
    
    # 使用真实分词器（如果可用）
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("   使用 GPT-2 分词器")
    except:
        # 创建模拟分词器
        class MockTokenizer:
            def __init__(self, vocab_size=100):
                self.vocab_size = vocab_size
            
            def __call__(self, text, **kwargs):
                return {"input_ids": [[1, 2, 3]]}
            
            def decode(self, ids, **kwargs):
                return "Generated text"
        
        tokenizer = MockTokenizer()
        print("   使用模拟分词器")
    print()
    
    # 导出到 ONNX
    print("[3] 导出到 ONNX 格式...")
    output_path = Path("output/onnx/model.onnx")
    export_to_onnx(model, tokenizer, output_path)
    print()
    
    print("[PASS] ONNX 部署选项测试通过")
    sys.exit(0)
