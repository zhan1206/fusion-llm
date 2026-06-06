"""
模型量化工具

完整的模型量化流程：
1. 动态量化（DyQuant）
2. 量化感知训练（QAT）
3. 量化后评估
4. 性能对比（量化 vs 原始）
"""

import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig, prepare_qat, convert

sys.path.insert(0, '.')

from evaluation.metrics import ModelEvaluator, EvaluationMetrics
from inference.dyquant import DyQuantConverter, QATTrainer, QuantConfig


class QuantizationTool:
    """模型量化工具（完整流程）"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer = None,
        device: str = "cpu",
        model_path: Optional[str] = None,
    ):
        """
        初始化量化工具
        
        参数：
            model: 要量化的模型
            tokenizer: tokenizer（可选）
            device: 设备
            model_path: 模型路径（用于 DyQuant）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_path = model_path or "fusion-mini"
        self.original_model = None
        self.quantized_model = None
        self.qat_trainer = None
        self.converter = None
    
    def backup_original_model(self):
        """备份原始模型"""
        print("[QuantTool] 备份原始模型...")
        import copy
        self.original_model = copy.deepcopy(self.model)
        print("[QuantTool] 备份完成")
    
    def dynamic_quantize(
        self,
        bits: int = 8,
        mode: str = "symmetric",
    ) -> nn.Module:
        """
        动态量化（post-training quantization）
        
        参数：
            bits: 量化位数（4/8）
            mode: 量化模式（symmetric/asymmetric）
            
        返回：
            量化后的模型
        """
        print(f"[QuantTool] 开始动态量化（{bits}-bit, {mode}）...")
        
        # 使用正确的 QuantConfig API
        config = QuantConfig(
            model_path=self.model_path,
            bits=bits,
            mixed_precision=False,
        )
        
        self.converter = DyQuantConverter(config)
        self.converter.model = self.model  # 注入已加载的模型
        self.quantized_model = self.converter.convert()
        
        print(f"[QuantTool] 动态量化完成")
        return self.quantized_model
    
    def prepare_qat(
        self,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        train_data: Optional[str] = None,
    ) -> "QATTrainer":
        """
        准备量化感知训练（QAT）
        
        参数：
            learning_rate: 学习率
            num_epochs: 训练轮数（注意：QATTrainer 没有 num_epochs 参数）
            train_data: 训练数据路径
            
        返回：
            QATTrainer 对象
        """
        print(f"[QuantTool] 准备量化感知训练（QAT）...")
        print(f"   学习率: {learning_rate}")
        print(f"   训练轮数: {num_epochs}")
        
        # 使用正确的 QuantConfig API
        config = QuantConfig(
            model_path=self.model_path,
            bits=4,
            mixed_precision=True,
        )
        
        # QATTrainer 签名：(config, train_data, learning_rate, warmup_steps)
        self.qat_trainer = QATTrainer(
            config=config,
            train_data=train_data,
            learning_rate=learning_rate,
            warmup_steps=100,
        )
        
        # 注入已加载的模型，避免重复加载
        self.qat_trainer.model = self.model
        
        self.qat_trainer.prepare()
        print(f"[QuantTool] QAT 准备完成")
        
        # 保存 num_epochs 供后续 train() 使用
        self._qat_epochs = num_epochs
        
        return self.qat_trainer
    
    def run_qat_training(self, epochs: Optional[int] = None):
        """运行 QAT 训练"""
        if self.qat_trainer is None:
            raise ValueError("请先调用 prepare_qat()")
        
        epochs = epochs or getattr(self, '_qat_epochs', 3)
        # QATTrainer.train() 签名：(epochs, lr, batch_size, max_len)
        self.qat_trainer.train(epochs=epochs)
        self.quantized_model = self.qat_trainer.qat_model
    
    def evaluate_quantized(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> EvaluationMetrics:
        """
        评估量化后的模型
        
        参数：
            texts: 评估文本
            max_length: 最大长度
            
        返回：
            评估指标
        """
        if self.quantized_model is None:
            raise ValueError("请先量化模型")
        
        print(f"[QuantTool] 评估量化后的模型...")
        
        evaluator = ModelEvaluator(
            model=self.quantized_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        
        metrics = evaluator.evaluate(texts, max_length)
        print(f"[QuantTool] 评估完成")
        print(metrics)
        
        return metrics
    
    def compare_models(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> Dict[str, EvaluationMetrics]:
        """
        对比原始模型和量化模型的性能
        
        参数：
            texts: 评估文本
            max_length: 最大长度
            
        返回：
            包含原始模型和量化模型评估指标的字典
        """
        if self.original_model is None:
            self.backup_original_model()
        
        if self.quantized_model is None:
            raise ValueError("请先量化模型")
        
        print(f"[QuantTool] 对比原始模型和量化模型...")
        print()
        
        # 1. 评估原始模型
        print("[1] 评估原始模型...")
        original_evaluator = ModelEvaluator(
            model=self.original_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        original_metrics = original_evaluator.evaluate(texts, max_length)
        print(f"   原始模型 Perplexity: {original_metrics.perplexity:.4f}")
        print(f"   原始模型 Loss: {original_metrics.loss:.4f}")
        print()
        
        # 2. 评估量化模型
        print("[2] 评估量化模型...")
        quantized_evaluator = ModelEvaluator(
            model=self.quantized_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        quantized_metrics = quantized_evaluator.evaluate(texts, max_length)
        print(f"   量化模型 Perplexity: {quantized_metrics.perplexity:.4f}")
        print(f"   量化模型 Loss: {quantized_metrics.loss:.4f}")
        print()
        
        # 3. 计算差异
        print("[3] 性能差异...")
        perplexity_diff = quantized_metrics.perplexity - original_metrics.perplexity
        loss_diff = quantized_metrics.loss - original_metrics.loss
        
        print(f"   Perplexity 差异: {perplexity_diff:+.4f}")
        print(f"   Loss 差异: {loss_diff:+.4f}")
        print()
        
        # 4. 模型大小对比
        original_size = self._get_model_size(self.original_model)
        quantized_size = self._get_model_size(self.quantized_model)
        size_reduction = (1 - quantized_size / original_size) * 100
        
        print("[4] 模型大小对比...")
        print(f"   原始模型大小: {original_size / 1e6:.2f} MB")
        print(f"   量化模型大小: {quantized_size / 1e6:.2f} MB")
        print(f"   大小减少: {size_reduction:.2f}%")
        print()
        
        print("[QuantTool] 对比完成")
        
        return {
            "original": original_metrics,
            "quantized": quantized_metrics,
        }
    
    def _get_model_size(self, model: nn.Module) -> int:
        """获取模型大小（字节）"""
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        return total_size
    
    def save_quantized(
        self,
        path: str,
        format: str = "safetensors",
    ):
        """
        保存量化后的模型
        
        参数：
            path: 保存路径
            format: 格式（safetensors/pytorch）
        """
        if self.quantized_model is None:
            raise ValueError("请先量化模型")
        
        print(f"[QuantTool] 保存量化模型到 {path}...")
        
        if format == "safetensors":
            try:
                import safetensors.torch
                safetensors.torch.save_file(self.quantized_model.state_dict(), path)
            except ImportError:
                print(f"[QuantTool] 警告：safetensors 未安装，使用 PyTorch 格式")
                format = "pytorch"
        
        if format == "pytorch":
            torch.save(self.quantized_model.state_dict(), path)
        
        print(f"[QuantTool] 保存完成")


def quantize_model(
    model: nn.Module,
    method: str = "dynamic",
    bits: int = 8,
    **kwargs,
) -> nn.Module:
    """
    便捷函数：量化模型
    
    参数：
        model: 要量化的模型
        method: 量化方法（dynamic/qat）
        bits: 量化位数
        **kwargs: 其他参数
        
    返回：
        量化后的模型
    """
    tool = QuantizationTool(model, **kwargs)
    
    if method == "dynamic":
        return tool.dynamic_quantize(bits=bits)
    elif method == "qat":
        trainer = tool.prepare_qat(**kwargs)
        epochs = kwargs.get('num_epochs', 3)
        trainer.train(epochs=epochs)
        return tool.quantized_model
    else:
        raise ValueError(f"不支持的量化方法: {method}")


if __name__ == "__main__":
    print("[QuantTool] 模型量化工具")
    print()
    print("功能：")
    print("  1. 动态量化（post-training）")
    print("  2. 量化感知训练（QAT）")
    print("  3. 量化后评估")
    print("  4. 性能对比（量化 vs 原始）")
    print()
    print("用法：")
    print("  from evaluation.quantization_tool import QuantizationTool")
    print("  tool = QuantizationTool(model, model_path='fusion-mini')")
    print("  quantized_model = tool.dynamic_quantize(bits=8)")
    print("  metrics = tool.compare_models(texts)")
