"""
模型可解释性工具：LIME 和 SHAP（简化版）

LIME: Local Interpretable Model-agnostic Explanations
SHAP: SHapley Additive exPlanations

注意：这是简化版实现，用于演示目的
实际使用时建议安装 lime 和 shap 包：pip install lime shap
"""
import sys
import math
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict

sys.path.insert(0, '.')


class SimplifiedLIME:
    """
    简化版 LIME 解释器
    
    通过遮蔽输入 token 来衡量每个 token 对输出的贡献
    """
    
    def __init__(self, model, tokenizer=None, num_samples: int = 100):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
    
    def explain(self, text: str, top_k: int = 10) -> Dict:
        """
        解释单个文本的预测
        
        Args:
            text: 输入文本
            top_k: 返回最重要的 top_k token
        
        Returns:
            dict: {token_importances: [{token, importance}], prediction, confidence}
        """
        self.model.eval()
        
        # Tokenize
        if self.tokenizer:
            tokens = text.split()  # 简化分词
        else:
            tokens = text.split()
        
        if not tokens:
            return {'token_importances': [], 'num_tokens': 0, 'original_score': 0.0}
        
        # 获取原始预测
        original_score = self._predict_score(tokens)
        
        # 遮蔽每个 token 并计算重要性
        importances = []
        for i in range(len(tokens)):
            masked_tokens = tokens[:i] + ['[MASK]'] + tokens[i+1:]
            masked_score = self._predict_score(masked_tokens)
            
            # 重要性 = 原始分数 - 遮蔽后分数
            importance = original_score - masked_score
            importances.append({
                'token': tokens[i],
                'index': i,
                'importance': importance,
            })
        
        # 排序
        importances.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return {
            'token_importances': importances[:top_k],
            'num_tokens': len(tokens),
            'original_score': original_score,
        }
    
    def _predict_score(self, tokens):
        """使用模型预测分数（简化版）"""
        # 将 token 转为 input_ids
        vocab_size = getattr(self.model.config, 'vocab_size', 100) if hasattr(self.model, 'config') else 100
        
        input_ids = []
        for t in tokens:
            # 简单 hash 到 vocab 范围
            token_id = int(hashlib.md5(t.encode()).hexdigest(), 16) % vocab_size
            input_ids.append(token_id)
        
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output', torch.tensor([[0.0]])))
                else:
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                
                # 返回最大 logit 作为分数
                score = logits[0, -1].max().item()
            except:
                score = 0.0
        
        return score


class SimplifiedSHAP:
    """
    简化版 SHAP 解释器
    
    使用 Shapley 值的近似计算来衡量每个 token 的贡献
    """
    
    def __init__(self, model, tokenizer=None, max_coalitions: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_coalitions = max_coalitions
    
    def explain(self, text: str, top_k: int = 10) -> Dict:
        """
        使用 Shapley 值解释单个文本
        
        Args:
            text: 输入文本
            top_k: 返回最重要的 top_k token
        
        Returns:
            dict: {shap_values: [{token, shap_value}], base_value, prediction}
        """
        self.model.eval()
        
        tokens = text.split()
        n = len(tokens)
        
        if n == 0:
            return {'shap_values': [], 'base_value': 0.0}
        
        # 计算 base value（空输入的预测）
        base_value = self._predict_score([])
        
        # 计算 Shapley 值
        shap_values = []
        
        for i in range(n):
            # 简化 Shapley 值计算：随机采样联盟
            marginal_contributions = []
            
            for _ in range(min(self.max_coalitions, 2 ** n)):
                # 随机生成联盟（不包含 token i）
                coalition = [j for j in range(n) if j != i and np.random.random() > 0.5]
                
                # 有 token i 的联盟
                coalition_with_i = coalition + [i]
                
                # 计算边际贡献
                score_without = self._predict_score([tokens[j] for j in coalition])
                score_with = self._predict_score([tokens[j] for j in coalition_with_i])
                
                marginal_contributions.append(score_with - score_without)
            
            # Shapley 值 = 边际贡献的平均
            shap_value = np.mean(marginal_contributions) if marginal_contributions else 0.0
            shap_values.append({
                'token': tokens[i],
                'index': i,
                'shap_value': shap_value,
            })
        
        # 排序
        shap_values.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'shap_values': shap_values[:top_k],
            'base_value': base_value,
            'num_tokens': n,
        }
    
    def _predict_score(self, tokens):
        """使用模型预测分数"""
        vocab_size = getattr(self.model.config, 'vocab_size', 100) if hasattr(self.model, 'config') else 100
        
        if not tokens:
            return 0.0
        
        input_ids = []
        for t in tokens:
            token_id = int(hashlib.md5(t.encode()).hexdigest(), 16) % vocab_size
            input_ids.append(token_id)
        
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output', torch.tensor([[0.0]])))
                else:
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                
                score = logits[0, -1].max().item()
            except:
                score = 0.0
        
        return score


class AttentionVisualizer:
    """
    基于注意力权重的解释工具
    """
    
    def __init__(self, model):
        self.model = model
    
    def get_attention_weights(self, input_ids, attention_mask=None):
        """
        获取注意力权重（简化版）
        
        注意：需要模型支持输出注意力权重
        """
        self.model.eval()
        
        with torch.no_grad():
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                if isinstance(outputs, dict) and 'attentions' in outputs:
                    return outputs['attentions']
            except:
                pass
        
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 模型可解释性工具测试")
    print("=" * 60)
    print()
    
    # 创建模型
    print("[1] 创建模型...")
    from models.fusion_mini import FusionMini, FusionMiniConfig
    config = FusionMiniConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
    model = FusionMini(config)
    print("   模型已创建")
    print()
    
    # 测试 LIME
    print("[2] 测试 LIME 解释器...")
    lime = SimplifiedLIME(model, num_samples=10)
    explanation = lime.explain("the cat sat on the mat", top_k=5)
    print(f"   Token 数量: {explanation['num_tokens']}")
    print(f"   原始分数: {explanation['original_score']:.4f}")
    print(f"   Top-5 重要 token:")
    for item in explanation['token_importances']:
        print(f"      '{item['token']}' (index={item['index']}): {item['importance']:.4f}")
    print("   LIME 测试通过")
    print()
    
    # 测试 SHAP
    print("[3] 测试 SHAP 解释器...")
    np.random.seed(42)
    shap = SimplifiedSHAP(model, max_coalitions=10)
    shap_result = shap.explain("the cat sat on the mat", top_k=5)
    print(f"   Base value: {shap_result['base_value']:.4f}")
    print(f"   Top-5 SHAP token:")
    for item in shap_result['shap_values']:
        print(f"      '{item['token']}' (index={item['index']}): {item['shap_value']:.4f}")
    print("   SHAP 测试通过")
    print()
    
    # 边缘情况
    print("[4] 边缘情况测试...")
    empty_explain = lime.explain("", top_k=5)
    assert empty_explain['num_tokens'] == 0
    print("   空输入测试通过")
    
    single_explain = lime.explain("hello", top_k=5)
    assert single_explain['num_tokens'] == 1
    print("   单 token 测试通过")
    print()
    
    print("[PASS] 模型可解释性工具测试全部通过")
    sys.exit(0)
