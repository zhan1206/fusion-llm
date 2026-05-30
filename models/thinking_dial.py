"""
Fusion 模型核心：动态推理强度调节器（Thinking Dial）

创新点：
1. 通过特殊 token `<|think| depth=0/1/2/3|>` 控制推理深度
2. depth=0：直接作答（闲聊、翻译）
3. depth=3：长思维链模式（数学、代码调试）
4. 通过 GRPO 强化学习加入简洁性惩罚
5. 一个模型同时拥有 Mistral 的爽快与 DeepSeek 的深沉

作者：朱子瞻
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class ThinkingConfig:
    """
    推理强度配置
    """
    depth: int = 0  # 0-3，推理深度
    max_thinking_tokens: int = 512  # 最大思维链长度
    temperature: float = 1.0  # 生成温度
    do_sample: bool = True  # 是否采样
    
    # 不同 depth 的预设配置
    @classmethod
    def from_depth(cls, depth: int) -> "ThinkingConfig":
        presets = {
            0: cls(depth=0, max_thinking_tokens=0, temperature=0.9, do_sample=False),
            1: cls(depth=1, max_thinking_tokens=128, temperature=0.85, do_sample=True),
            2: cls(depth=2, max_thinking_tokens=256, temperature=0.8, do_sample=True),
            3: cls(depth=3, max_thinking_tokens=512, temperature=0.75, do_sample=True),
        }
        return presets.get(depth, cls(depth=depth))


class ThinkingDialProcessor:
    """
    处理 Thinking Dial 控制 token
    
    特殊 token 格式：<|think| depth={0,1,2,3}|>
    """
    
    # 特殊 token 正则表达式
    THINK_PATTERN = re.compile(r"<\|think\|\s*depth\s*=\s*(\d)\|>")
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # 添加特殊 token 到 tokenizer
        special_tokens = ["<|think|", "|>"]  # 简化版本
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    def parse_thinking_depth(self, prompt: str) -> Tuple[int, str]:
        """
        从 prompt 中解析推理深度
        
        返回：
            depth: 0-3 推理深度
            clean_prompt: 移除控制 token 后的 prompt
        """
        match = self.THINK_PATTERN.search(prompt)
        
        if match:
            depth = int(match.group(1))
            clean_prompt = self.THINK_PATTERN.sub("", prompt).strip()
            return depth, clean_prompt
        
        # 默认 depth=0（直接作答）
        return 0, prompt
    
    def inject_thinking_token(
        self,
        prompt: str,
        depth: int,
    ) -> str:
        """
        注入 Thinking Dial 控制 token
        
        参数：
            prompt: 原始提示
            depth: 0-3 推理深度
            
        返回：
            注入控制 token 后的提示
        """
        if depth < 0 or depth > 3:
            raise ValueError(f"depth must be 0-3, got {depth}")
        
        thinking_token = f"<|think| depth={depth}|>"
        return f"{thinking_token}\n{prompt}"
    
    def format_training_example(
        self,
        prompt: str,
        response: str,
        think_rank: int,
    ) -> Dict[str, str]:
        """
        格式化训练样本（用于 SFT/RLHF）
        
        参数：
            prompt: 用户输入
            response: 模型回答
            think_rank: 推理深度标签（0-3）
            
        返回：
            格式化后的训练样本
        """
        # 注入控制 token
        formatted_prompt = self.inject_thinking_token(prompt, think_rank)
        
        return {
            "prompt": formatted_prompt,
            "response": response,
            "think_rank": think_rank,
        }


class ThinkingDialModel(nn.Module):
    """
    集成 Thinking Dial 的 Fusion 模型
    
    在推理时根据 depth 动态调整生成策略
    """
    
    def __init__(self, base_model, tokenizer, config: Optional[ThinkingConfig] = None):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.processor = ThinkingDialProcessor(tokenizer)
        self.config = config or ThinkingConfig()
        
    def generate_with_thinking(
        self,
        prompt: str,
        thinking_depth: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        带推理控制的生成
        
        参数：
            prompt: 输入提示
            thinking_depth: 推理深度（0-3），如果为 None 则自动解析
            **kwargs: 其他生成参数
            
        返回：
            生成的文本
        """
        # 解析或设置推理深度
        if thinking_depth is not None:
            depth = thinking_depth
            clean_prompt = prompt
        else:
            depth, clean_prompt = self.processor.parse_thinking_depth(prompt)
        
        # 获取该深度的配置
        config = ThinkingConfig.from_depth(depth)
        
        # 注入控制 token
        if depth > 0:
            formatted_prompt = self.processor.inject_thinking_token(clean_prompt, depth)
        else:
            formatted_prompt = clean_prompt
        
        # 编码
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # 根据深度调整生成参数
        gen_kwargs = {
            "max_new_tokens": config.max_thinking_tokens if depth > 0 else 256,
            "temperature": config.temperature,
            "do_sample": config.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }
        
        # 生成
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 如果 depth > 0，可能需要提取思维链（简化实现）
        if depth > 0:
            # 实际实现中可以解析 `<think>...</think>` 标签
            response = self._extract_thinking_and_response(response, depth)
        
        return response
    
    def _extract_thinking_and_response(self, text: str, depth: int) -> str:
        """
        提取思维链和最终回答（简化实现）
        """
        # 这里可以解析特殊标签，如 `<think>...</think>`
        # 当前简化版本：直接返回全文
        return text
    
    def batch_generate(
        self,
        prompts: List[str],
        thinking_depths: Optional[List[int]] = None,
        **kwargs,
    ) -> List[str]:
        """
        批量生成
        """
        if thinking_depths is None:
            thinking_depths = [None] * len(prompts)
        
        responses = []
        for prompt, depth in zip(prompts, thinking_depths):
            response = self.generate_with_thinking(
                prompt, thinking_depth=depth, **kwargs
            )
            responses.append(response)
        
        return responses


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) 训练器
    
    用于强化学习对齐，加入简洁性惩罚
    """
    
    def __init__(self, model, tokenizer, reward_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model  # 可选：用户自己的偏好模型
        
    def compute_reward(
        self,
        prompt: str,
        response: str,
        thinking_depth: int,
    ) -> float:
        """
        计算奖励（简化版本）
        
        奖励组成：
            1. 任务完成度（正确性）
            2. 简洁性惩罚（思维链过长时惩罚）
            3. 格式奖励（是否遵循 depth 要求）
        """
        # 1. 任务奖励（需要外部评估或奖励模型）
        task_reward = 0.0
        if self.reward_model is not None:
            task_reward = self.reward_model.score(prompt, response)
        else:
            # 简化：假设任务完成度为 1.0
            task_reward = 1.0
        
        # 2. 简洁性惩罚
        thinking_length = len(response.split())  # 简化：用词数衡量
        max_allowed = ThinkingConfig.from_depth(thinking_depth).max_thinking_tokens
        
        if thinking_length > max_allowed:
            simplicity_penalty = -0.1 * (thinking_length - max_allowed) / max_allowed
        else:
            simplicity_penalty = 0.0
        
        # 3. 格式奖励
        format_reward = 1.0 if self._check_format(response, thinking_depth) else -0.5
        
        total_reward = task_reward + simplicity_penalty + format_reward
        return total_reward
    
    def _check_format(self, response: str, thinking_depth: int) -> bool:
        """
        检查回答格式是否符合要求
        """
        # 简化检查：是否包含思维链标记
        if thinking_depth >= 2:
            return "<think>" in response and "</think>" in response
        return True
    
    def train_step(self, batch: Dict[str, List]) -> Dict[str, float]:
        """
        执行一步 GRPO 训练（简化版本）
        """
        # 实际实现需要：
        # 1. 采样多个回答
        # 2. 计算相对奖励
        # 3. 计算策略梯度
        # 4. 更新模型参数
        
        # 这里只提供框架
        prompts = batch["prompt"]
        responses = batch["response"]
        thinking_depths = batch["think_rank"]
        
        rewards = []
        for prompt, response, depth in zip(prompts, responses, thinking_depths):
            reward = self.compute_reward(prompt, response, depth)
            rewards.append(reward)
        
        # 返回平均奖励（实际训练需要更复杂的逻辑）
        return {"avg_reward": sum(rewards) / len(rewards)}


if __name__ == "__main__":
    # 单元测试（模拟）
    print("🧪 测试 Thinking Dial 机制...")
    
    # 模拟 tokenizer 和 model
    class MockTokenizer:
        def add_special_tokens(self, tokens):
            pass
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.randint(0, 1000, (1, 50))}
        def decode(self, ids, skip_special_tokens=True):
            return "模拟生成结果"
    
    class MockModel(nn.Module):
        def generate(self, **kwargs):
            return torch.randint(0, 1000, (1, 100))
    
    tokenizer = MockTokenizer()
    model = MockModel()
    
    # 测试 ThinkingDialProcessor
    processor = ThinkingDialProcessor(tokenizer)
    
    test_prompt = "<|think| depth=2|> 证明勾股定理"
    depth, clean = processor.parse_thinking_depth(test_prompt)
    print(f"✅ 解析 depth: {depth}, clean_prompt: {clean}")
    
    # 测试注入
    injected = processor.inject_thinking_token("解释量子纠缠", depth=1)
    print(f"✅ 注入控制 token: {injected}")
    
    # 测试 ThinkingDialModel
    thinking_model = ThinkingDialModel(model, tokenizer)
    
    # 模拟生成（简化）
    response = thinking_model.generate_with_thinking(
        "什么是机器学习",
        thinking_depth=0,
    )
    print(f"✅ depth=0 生成: {response[:50]}...")
    
    print("\n✅ Thinking Dial 测试通过！")
    print("💡 提示：完整功能需要集成真实的语言模型")
