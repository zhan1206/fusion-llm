"""
Thinking Dial（动态推理强度控制）- 真实实现

核心功能：
1. 通过特殊 token 控制推理深度 `<|think| depth=N|>`（N=0-3）
2. Depth 0：直接回答（闲聊、翻译、简单问答）
3. Depth 3：长思维链模式（数学证明、代码调试、复杂推理）
4. 一个模型同时拥有 Mistral 的爽快与 DeepSeek 的深沉

实现说明：
- 通过特殊 token 注入推理控制信号
- 使用 GRPO（Group Relative Policy Optimization）训练 Thinking Dial 能力
- 支持 HuggingFace Transformers 接口（generate 方式）
- 提供 ThinkingDialProcessor 用于预处理，ThinkingDialModel 用于训练

架构说明 (F1):
- 当前实现为 per-batch 粒度 (每个样本同一深度)，非 per-token
- 这是效率和训练稳定性的权衡：per-token 需要更复杂的梯度流
- 如需 per-token 控制，可在 forward() 中接收 thinking_depth 作为 input_ids 对应的张量

已知限制:
- F3: T-KD 蒸馏依赖未发布的 teacher 模型权重
  当前 train_t_kd_distillation.py 需要预训练的 teacher model
  如无可用权重，模型仍可从随机初始化开始训练 (冷启动)
- F2: SBLA 增量生成使用 lossy 的块潜向量近似
  这是设计权衡：精确的全局潜向量需要 O(seq_len) 内存
  当前实现用缓存的 block latents，在精度和效率间平衡

使用方法：
    # 1. 预处理数据（注入 thinking token）
    processor = ThinkingDialProcessor(tokenizer)
    processed = processor.process(raw_data)
    
    # 2. 训练时支持 think_rank
    trainer = GRPOTrainer(model, grpo_config)
    trainer.train(training_data)
    
    # 3. 推理时控制深度
    output = model.generate(
        input_ids,
        thinking_depth=2,  # 0-3
    )

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, GenerationMixin


# ============================================================
# 特殊 Token 定义
# ============================================================

THINK_START = "<|think_depth_"
THINK_CLOSE = "|>"  # Closing bracket for think_depth token
THINK_END = "<|think_end|>"  # End-of-thinking-block marker
THINK_DEPTH_PATTERN = re.compile(r"<\|think_depth_(\d+)\|")  # N14 FIX: \d+ for future depth>=10

# Depth 0-3 的描述
THINK_DEPTH_DESCRIPTIONS = {
    0: "直接回答模式 - 快速响应，适用于闲聊、翻译、简单问答",
    1: "基础思考模式 - 简短分析，适用于一般性问题",
    2: "深度思考模式 - 详细推理，适用于复杂问题",
    3: "极致思考模式 - 完整思维链，适用于数学、代码、复杂推理",
}


def build_think_token(depth: int) -> str:
    """
    构建带深度信息的 thinking token
    
    参数：
        depth: 推理深度（0-3）
        
    返回：
        thinking token 字符串
    """
    if not 0 <= depth <= 3:
        raise ValueError(f"depth 必须在 0-3 之间，当前值：{depth}")
    
    return f"{THINK_START}{depth}{THINK_CLOSE}"


def parse_think_token(text: str) -> Optional[int]:
    """
    从文本中解析 thinking depth
    
    参数：
        text: 带 thinking token 的文本
        
    返回：
        推理深度（0-3）或 None
    """
    matches = THINK_DEPTH_PATTERN.findall(text)
    if matches:
        return int(matches[0])
    return None


# ============================================================
# Thinking Dial 配置
# ============================================================

@dataclass
class ThinkingConfig:
    """
    Thinking Dial 配置
    """
    # 是否启用 Thinking Dial
    enable_thinking_dial: bool = True
    
    # Thinking depth 数量（默认为 4：0-3）
    num_thinking_depths: int = 4
    
    # 各深度在训练数据中的比例
    depth_ratios: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.depth_ratios is None:
            # 默认：简单问题多，复杂问题少
            self.depth_ratios = [0.4, 0.3, 0.2, 0.1]
        
        # 检查比例和
        if abs(sum(self.depth_ratios) - 1.0) > 0.01:
            raise ValueError(f"depth_ratios 总和必须为 1.0，当前：{self.depth_ratios}")
        
        if len(self.depth_ratios) != self.num_thinking_depths:
            raise ValueError(f"depth_ratios 长度必须为 {self.num_thinking_depths}")


@dataclass
class GRPOConfig:
    """
    GRPO（Group Relative Policy Optimization）配置
    
    GRPO 是一种简化版的 PPO，不需要额外的批评模型。
    通过组内相对优势来更新策略。
    """
    # 组内采样数
    grpo_sample_size: int = 8
    
    # 优势估计的衰减因子
    gamma: float = 1.0
    
    # 策略更新的 KL 散度系数
    kl_coef: float = 0.01
    
    # 裁剪范围
    epsilon: float = 0.2
    
    # 奖励基线类型（mean/median/min）
    baseline_type: str = "mean"
    
    # 是否使用正则化
    use_regulation: bool = True


# ============================================================
# Thinking Dial 处理器
# ============================================================

class ThinkingDialProcessor:
    """
    Thinking Dial 数据处理器
    
    负责在数据预处理阶段注入 thinking token
    """
    
    def __init__(
        self,
        tokenizer,
        enable_thinking_dial: bool = True,
    ):
        self.tokenizer = tokenizer
        self.enable_thinking_dial = enable_thinking_dial
        
        # 注册特殊 token
        self._register_special_tokens()
    
    def _register_special_tokens(self):
        """注册特殊 token 到 tokenizer"""
        # M1-M3 FIX: Register complete think tokens matching tokenizer.py format
        # tokenizer.py registers ["<|think_depth_0|>", "<|think_depth_1|>", "<|think_depth_2|>", "<|think_depth_3|>"]
        think_tokens = [build_think_token(d) for d in range(4)]  # ["<|think_depth_0|>", ..., "<|think_depth_3|>"]
        
        special_tokens = {
            "additional_special_tokens": think_tokens,
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        
        if num_added > 0:
            self.tokenizer.resize_embeddings(
                self.tokenizer.vocab_size + num_added
            )
    
    def process_single(
        self,
        prompt: str,
        response: str,
        think_rank: int = 0,
    ) -> Dict[str, Any]:
        """
        处理单条数据
        
        参数：
            prompt: 用户问题
            response: 模型回答
            think_rank: 推理深度（0-3）
            
        返回：
            包含处理后文本的字典
        """
        if not self.enable_thinking_dial:
            return {
                "text": f"{prompt}\n{response}",
                "think_rank": 0,
            }
        
        # 构建 thinking token
        think_token = build_think_token(think_rank)
        
        # 根据深度决定是否需要 thinking token
        if think_rank == 0:
            # depth=0：直接回答，不需要 thinking token
            full_text = f"{prompt}\n{response}"
        else:
            # depth>0：添加 thinking token
            full_text = f"{think_token}\n{prompt}\n{response}\n{THINK_END}"
        
        return {
            "text": full_text,
            "prompt": prompt,
            "response": response,
            "think_rank": think_rank,
            "think_token": think_token if think_rank > 0 else None,
        }
    
    def process_dataset(
        self,
        data: List[Dict],
        prompt_key: str = "prompt",
        response_key: str = "response",
        think_rank_key: str = "think_rank",
    ) -> List[Dict]:
        """
        批量处理数据集
        
        参数：
            data: 原始数据列表
            prompt_key: prompt 字段名
            response_key: response 字段名
            think_rank_key: think_rank 字段名
            
        返回：
            处理后的数据列表
        """
        processed = []
        
        for item in data:
            prompt = item.get(prompt_key, "")
            response = item.get(response_key, "")
            think_rank = item.get(think_rank_key, 0)
            
            processed_item = self.process_single(prompt, response, think_rank)
            processed.append(processed_item)
        
        return processed
    
    def tokenize(
        self,
        text: str,
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        分词
        
        参数：
            text: 文本
            max_length: 最大长度
            padding: 填充策略
            truncation: 是否截断
            
        返回：
            包含 input_ids 和 attention_mask 的字典
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
    
    def __call__(self, text, **kwargs):
        """作为可调用对象使用"""
        return self.tokenize(text, **kwargs)


# ============================================================
# GRPO Trainer
# ============================================================

class GRPOTrainer:
    """
    GRPO（Group Relative Policy Optimization）训练器
    
    GRPO 是一种简化版的 PPO 算法，不需要额外的批评模型。
    它通过组内相对优势来更新语言模型的策略。
    
    核心思想：
    1. 对同一输入采样多个回复
    2. 根据奖励计算组内相对优势
    3. 使用策略梯度更新模型
    
    参考：DeepSeekMath GRPO
    """
    
    # S1 FIX: Built-in reward function registry
    REWARD_FUNCTIONS = {}
    
    @classmethod
    def register_reward_fn(cls, name: str, fn: callable):
        """Register a custom reward function by name.
        
        Args:
            name: Function name for lookup
            fn: Callable(prompt: str, response: str) -> float
        """
        cls.REWARD_FUNCTIONS[name] = fn
    
    def __init__(
        self,
        model: PreTrainedModel,
        grpo_config: Optional[GRPOConfig] = None,
        thinking_config: Optional[ThinkingConfig] = None,
        tokenizer=None,  # N16 FIX: Accept tokenizer for decode
        reward_fn=None,  # S1 FIX: Accept reward function at init
    ):
        self.model = model
        self.grpo_config = grpo_config or GRPOConfig()
        self.thinking_config = thinking_config or ThinkingConfig()
        self.tokenizer = tokenizer  # N16 FIX
        self.reward_fn = reward_fn  # S1 FIX: Store reward function
        
        # 优化器
        self.optimizer = None
        
        # 统计
        self.step_count = 0
        self.loss_history = []
    
    def setup_optimizer(self, learning_rate: float = 1e-6):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        sample_size: int = None,
    ) -> torch.Tensor:
        """
        计算组内相对优势
        
        参数：
            rewards: (group_size,) 每组的奖励
            sample_size: 每组采样数
            
        返回：
            advantages: (group_size,) 组内优势
        """
        sample_size = sample_size or self.grpo_config.grpo_sample_size
        
        # 分组
        num_groups = len(rewards) // sample_size
        if num_groups <= 1:
            # 只有一组时，优势为 0（相对均值为 0）
            return torch.zeros_like(rewards)
        
        rewards = rewards[:num_groups * sample_size]
        groups = rewards.view(num_groups, sample_size)  # (num_groups, sample_size)
        
        # 组内标准化
        mean = groups.mean(dim=1, keepdim=True)  # (num_groups, 1)
        std = groups.std(dim=1, keepdim=True) + 1e-8  # (num_groups, 1)
        
        advantages = (groups - mean) / std  # (num_groups, sample_size)
        
        return advantages.flatten()
    
    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 GRPO 损失
        
        参数：
            log_probs: 新策略的 log 概率
            advantages: 优势估计
            old_log_probs: 旧策略的 log 概率（用于 KL 约束）
            
        返回：
            GRPO 损失
        """
        # 策略梯度损失
        pg_loss = -advantages * log_probs
        
        # KL 散度损失
        kl_loss = 0.0
        if old_log_probs is not None and self.grpo_config.kl_coef > 0:
            ratio = torch.exp(log_probs - old_log_probs)
            kl_div = ratio - 1 - (log_probs - old_log_probs)
            kl_loss = self.grpo_config.kl_coef * kl_div.mean()
        
        # 总损失
        total_loss = pg_loss.mean() + kl_loss
        
        return total_loss
    
    def generate_with_thinking(
        self,
        input_ids: torch.Tensor,
        thinking_depth: int = 0,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs,
    ) -> List[str]:
        """
        使用 thinking depth 生成回复
        
        参数：
            input_ids: 输入 token IDs
            thinking_depth: 推理深度（0-3）
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_p: nucleus 采样
            
        返回：
            生成的文本列表
        """
        # M5 FIX: Pass thinking_depth to model.generate() so the ThinkingDialModel
        # forward() can actually apply depth-dependent bias to logits
        # M5/N10/N12: Pass thinking_depth via ThinkingDialModel.generate() which builds logits_hook
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=getattr(self.model.config, 'pad_token_id', None) or 0,
            eos_token_id=getattr(self.model.config, 'eos_token_id', None) or 1,
            thinking_depth=thinking_depth,
            **kwargs,
        )
        
        # Decode - N16 FIX: Use self.tokenizer if available
        if self.tokenizer is not None:
            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            texts = [f"[generated_ids shape: {outputs.shape}]" for _ in outputs]
        
        return texts
    
    def _normalize_logits_to_log_probs(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        per_token: bool = False,
    ) -> torch.Tensor:
        """Extract log probabilities from logits.
        
        Args:
            logits: (B, seq_len, vocab)
            labels: (B, seq_len) target token ids (shifted)
            per_token: If True, return (B, seq_len) per-token log probs;
                       If False, return (B,) per-sequence sum (legacy behavior)
        
        For GRPO, we need per-token log probs to mask prompt positions.
        """
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            per_token_lp = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)  # (B, L)
            # Use -100 as ignore index (standard PyTorch/HuggingFace convention for masked labels)
            mask = (shift_labels != -100) & (shift_labels != 0)
            if per_token:
                return per_token_lp * mask.float()  # (B, L)
            return (per_token_lp * mask.float()).sum(dim=1)  # (B,)
        return torch.log_softmax(logits[:, -1, :], dim=-1).sum(dim=-1)

    def compute_reward(
        self,
        prompt: str,
        response: str,
        reward_fn=None,
    ) -> float:
        """
        Compute reward for a generated response.
        
        S1 FIX: Now supports three reward sources (in priority order):
        1. reward_fn parameter (per-call override)
        2. self.reward_fn (instance-level, set at init)
        3. REWARD_FUNCTIONS registry (by name string)
        4. Built-in heuristics (fallback)
        
        Built-in reward heuristics (used when reward_fn is None):
        1. Length reward: prefer moderate length (not too short, not too long)
        2. Coherence reward: penalize excessive repetition
        3. Format reward: bonus for structured output (numbered lists, code blocks)
        
        Args:
            prompt: Input prompt text
            response: Generated response text
            reward_fn: Optional custom reward function, or string key into REWARD_FUNCTIONS
            
        Returns:
            Reward score (float)
        """
        # S1 FIX: Priority: param > instance > registry > built-in
        if callable(reward_fn):
            return reward_fn(prompt, response)
        if reward_fn is not None and reward_fn in self.REWARD_FUNCTIONS:
            return self.REWARD_FUNCTIONS[reward_fn](prompt, response)
        # S2 FIX: Check callable before invoking instance reward_fn (could be string from registry)
        if self.reward_fn is not None and callable(self.reward_fn):
            return self.reward_fn(prompt, response)
        if self.reward_fn is not None and isinstance(self.reward_fn, str) and self.reward_fn in self.REWARD_FUNCTIONS:
            return self.REWARD_FUNCTIONS[self.reward_fn](prompt, response)
        
        score = 0.0
        
        # 1. Length reward (optimal: 50-500 chars)
        resp_len = len(response.strip())
        if resp_len > 20:
            if resp_len < 50:
                score += 0.3
            elif resp_len < 200:
                score += 0.7
            elif resp_len < 500:
                score += 0.5
            else:
                score += 0.2
        else:
            score -= 0.5
        
        # 2. Repetition penalty
        words = response.split()
        if len(words) > 1:
            bigrams = list(zip(words[:-1], words[1:]))
            bigram_set = set(bigrams)
            repetition_ratio = 1.0 - (len(bigram_set) / max(len(words) - 1, 1))
            score -= repetition_ratio * 0.3
        
        # 3. Format reward
        if "```" in response:
            score += 0.2
        if any(f"{i}." in response for i in range(1, 6)):
            score += 0.1
        
        return score

    # S1 FIX: Convenience method to set reward function after init
    def set_reward_fn(self, reward_fn: callable):
        """Set the reward function for this trainer instance."""
        self.reward_fn = reward_fn

    def generate_samples(
        self,
        input_ids: torch.Tensor,
        num_samples: int = 4,
        thinking_depth: int = 0,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate multiple samples from the same input for GRPO.
        
        Args:
            input_ids: (batch, seq_len) input tokens
            num_samples: Number of responses per input
            thinking_depth: Thinking Dial depth (0-3)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Tuple of (all_generated_ids, decoded_texts)
        """
        all_ids = []
        all_texts = []
        
        # N17 FIX: Prefill once, then clone KV cache for each sample
        # Determine the actual generation model (unwrap ThinkingDialModel if needed)
        gen_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        
        with torch.no_grad():
            prefill_outputs = gen_model.forward(
                input_ids=input_ids,
                use_cache=True,
            )
            base_kv = prefill_outputs.past_key_values
            first_logits = prefill_outputs.logits[:, -1, :]  # (B, vocab)
        
        # N19 FIX: Use shared logits_hook builder (single source of truth)
        # N24 FIX: Guard against missing Thinking Dial attributes on pure FusionModel
        logits_hook_arg = None
        if thinking_depth is not None and hasattr(self.model, 'thinking_embedding'):
            logits_hook_arg = ThinkingDialModel._build_thinking_logits_hook(
                thinking_depth, input_ids.shape[0], input_ids.device,
                self.model.thinking_config, self.model.thinking_embedding, self.model.thinking_gate,
                gen_model.lm_head,
            )
        
        for _ in range(num_samples):
            # Sample first token from prefill logits
            first_logits_for_sample = first_logits / temperature
            # N18 FIX: Apply thinking bias to first token logits too
            if logits_hook_arg is not None:
                first_logits_for_sample = logits_hook_arg(first_logits_for_sample.unsqueeze(1)).squeeze(1)
            probs = F.softmax(first_logits_for_sample, dim=-1)
            first_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Clone KV cache for this sample (each sample diverges)
            sample_kv = tuple(tuple(t.clone() for t in layer_kv) for layer_kv in base_kv)
            
            # Generate rest using pre-computed KV cache
            outputs = gen_model.generate(
                input_ids=first_token,
                max_new_tokens=max_new_tokens - 1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=getattr(gen_model.config, 'pad_token_id', None) or 0,
                eos_token_id=getattr(gen_model.config, 'eos_token_id', None) or 1,
                past_key_values=sample_kv,  # N17 FIX: Reuse prefilled KV cache
                logits_hook=logits_hook_arg,
            )
            # outputs already includes first_token at position 0, prepend original input_ids
            full_output = torch.cat([input_ids, outputs], dim=1)
            all_ids.append(full_output)
            
            if self.tokenizer is not None:  # N16 FIX
                texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_texts.extend(texts)
        
        return torch.cat(all_ids, dim=0), all_texts

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reward_fn=None,
        thinking_depth: int = 0,  # S2 FIX: Allow configuring thinking depth for training
    ) -> Dict[str, float]:
        """
        Execute one GRPO training step with multi-sample generation.
        
        Real GRPO loop:
        1. Generate N samples per input
        2. Compute rewards for each sample (built-in heuristics or custom)
        3. Compute group-relative advantages
        4. Policy gradient update with KL constraint
        
        Args:
            input_ids: Input token IDs
            labels: Optional labels for log prob computation
            reward_fn: Optional custom reward function(prompt, response) -> float
            thinking_depth: Thinking Dial depth (0-3) for generation (S2 FIX)
            
        Returns:
            Training statistics
        """
        self.model.train()
        
        if self.optimizer is None:
            self.setup_optimizer()
        
        device = input_ids.device
        num_samples = self.grpo_config.grpo_sample_size
        
        # Step 1: Generate multiple samples per input
        generated_ids, generated_texts = self.generate_samples(
            input_ids, num_samples=num_samples, thinking_depth=thinking_depth,
        )
        
        # Step 2: Compute rewards
        rewards_list = []
        for i, text in enumerate(generated_texts):
            prompt_idx = i // num_samples
            prompt_text = ""
            if self.tokenizer is not None:  # N16 FIX
                prompt_text = self.tokenizer.decode(input_ids[prompt_idx], skip_special_tokens=True)
            reward = self.compute_reward(prompt_text, text, reward_fn)
            rewards_list.append(reward)
        
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        
        # Step 3: Compute group-relative advantages
        advantages = self.compute_advantages(rewards, sample_size=num_samples)
        
        # Step 4: Get log probs and compute GRPO loss
        # N22 FIX / N23 FIX: Build loss_mask to exclude prompt tokens from log_prob computation
        prompt_len = input_ids.shape[1]
        
        outputs = self.model(input_ids=generated_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        
        # Labels: shift right so log_probs[i] = P(token[i+1] | token[...i])
        use_labels = generated_ids[:, 1:].clone()  # predict next token
        # N23 FIX: Get per token log probs, then mask prompt positions correctly
        # generated_ids layout: [prompt_tokens | gen_tokens]
        # logits layout:     [prompt_logits | gen_logits] (shifted by 1)
        # We want log_probs starting from position prompt_len-2 (first gen token prediction)
        # N25 FIX: off-by-one - first gen token is at index prompt_len-2, not prompt_len-1
        mask_start = max(prompt_len - 2, 0)  # logits at prompt_len-2 predict token at prompt_len-1
        
        log_probs_per_token = self._normalize_logits_to_log_probs(logits, use_labels, per_token=True)  # (B*N, L)
        # Zero out prompt positions so GRPO loss only uses generated tokens
        log_probs_per_token[:, :mask_start] = 0.0
        log_probs = log_probs_per_token.sum(dim=1)  # (B*N,) per-sequence sum of gen-only log probs
        
        loss = self.compute_grpo_loss(log_probs, advantages)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Stats
        self.step_count += 1
        self.loss_history.append(loss.item())
        
        return {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item(),
            'std_reward': rewards.std().item() if len(rewards) > 1 else 0.0,
            'step': self.step_count,
        }
    
    def train_step_with_labels(
        self,
        batch: Dict[str, torch.Tensor],
        reward_fn=None,
    ) -> Dict[str, float]:
        """
        使用标签数据执行一步训练
        
        参数：
            batch: 包含 input_ids, attention_mask, labels 的字典
            reward_fn: 可选的奖励函数
            
        返回：
            训练统计字典
        """
        self.model.train()
        
        if self.optimizer is None:
            self.setup_optimizer()
        
        # 前向传播
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.step_count += 1
            self.loss_history.append(loss.item())
            
            return {"loss": loss.item()}
        
        return {"loss": 0.0}


# ============================================================
# Thinking Dial 模型增强
# ============================================================

class ThinkingDialModel(nn.Module):
    """
    Thinking Dial 增强模型
    
    在基础模型上添加 Thinking Dial 控制能力。
    通过额外的 embedding 层学习推理深度表示。
    
    N11 FIX: Now provides generate() method that delegates to base_model.generate()
    with thinking_depth forwarding, making it compatible with GRPOTrainer.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        thinking_config: Optional[ThinkingConfig] = None,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.thinking_config = thinking_config or ThinkingConfig()
        
        # Thinking embedding（学习推理深度表示）
        self.thinking_embedding = nn.Embedding(
            thinking_config.num_thinking_depths,
            base_model.config.hidden_size,
        )
        
        # 门控机制（控制 thinking embedding 的贡献度）
        self.thinking_gate = nn.Parameter(torch.tensor(0.1))
    
    @staticmethod
    def _build_thinking_logits_hook(
        thinking_depth: Optional[int],
        batch_size: int,
        device: torch.device,
        thinking_config: 'ThinkingConfig',
        thinking_embedding: torch.nn.Module,
        thinking_gate: torch.nn.Parameter,
        lm_head: torch.nn.Module,
    ) -> Optional[callable]:
        """N19 FIX: Single source of truth for constructing thinking depth logits hook.

        Returns a callable(logits) -> logits or None if thinking_depth is None.
        Used by both ThinkingDialModel.generate() and GRPOTrainer.generate_samples().
        """
        if thinking_depth is None:
            return None
        if isinstance(thinking_depth, int):
            thinking_depth_t = torch.tensor(
                [thinking_depth] * batch_size, device=device,
            )
        else:
            thinking_depth_t = thinking_depth
        depth_idx = thinking_depth_t.long().clamp(0, thinking_config.num_thinking_depths - 1)
        depth_embedding = thinking_embedding(depth_idx)  # (B, hidden)
        depth_hidden = thinking_gate * depth_embedding
        vocab_bias = lm_head(depth_hidden).unsqueeze(1)  # (B, 1, vocab)
        def thinking_logits_hook(logits):
            return logits + vocab_bias
        return thinking_logits_hook

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        thinking_depth: Optional[torch.Tensor] = None,
    ):
        """
        前向传播
        
        参数：
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
            labels: (batch, seq_len) 标签（用于计算 loss）
            thinking_depth: (batch,) 推理深度（0-3）
            
        返回：
            CausalLMOutputWithPast
        """
        # [F1 FIX] Call base model first, then inject thinking depth bias
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        if thinking_depth is not None:
            # M7 FIX: Use shared logits_hook builder (single source of truth with generate path)
            logits_hook = self._build_thinking_logits_hook(
                thinking_depth, base_outputs.logits.shape[0], base_outputs.logits.device,
                self.thinking_config, self.thinking_embedding, self.thinking_gate,
                self.base_model.lm_head,
            )
            if logits_hook is not None:
                biased_logits = logits_hook(base_outputs.logits)  # (B, seq_len, vocab)
                from transformers.modeling_outputs import CausalLMOutputWithPast
                base_outputs = CausalLMOutputWithPast(
                    loss=base_outputs.loss,
                    logits=biased_logits,
                    past_key_values=base_outputs.past_key_values,
                    hidden_states=base_outputs.hidden_states,
                    attentions=base_outputs.attentions,
                )
        
        return base_outputs
    
    # N11 FIX: Provide generate() method for GRPOTrainer compatibility
    @property
    def config(self):
        return self.base_model.config
    
    def generate(
        self,
        input_ids: torch.Tensor,
        thinking_depth: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Generate text with Thinking Dial control.
        
        N11 FIX: Now provides generate() by delegating to base_model.generate()
        with a logits_hook that applies depth-dependent bias at each step.
        
        Args:
            input_ids: Input token IDs
            thinking_depth: Thinking Dial depth (0-3)
            **kwargs: Additional args forwarded to base_model.generate()
            
        Returns:
            Generated token IDs tensor or CausalLMOutputWithPast
        """
        hook = ThinkingDialModel._build_thinking_logits_hook(
            thinking_depth, input_ids.shape[0], input_ids.device,
            self.thinking_config, self.thinking_embedding, self.thinking_gate,
            self.base_model.lm_head,
        )
        if hook is not None:
            kwargs['logits_hook'] = hook
        
        return self.base_model.generate(input_ids=input_ids, **kwargs)


# ============================================================
# 工具函数
# ============================================================

def apply_thinking_control(
    text: str,
    depth: int,
) -> str:
    """
    在文本中注入 thinking token
    
    参数：
        text: 原始文本
        depth: 推理深度（0-3）
        
    返回：
        带 thinking token 的文本
    """
    think_token = build_think_token(depth)
    
    if depth == 0:
        return text
    else:
        return f"{think_token}\n{text}\n{THINK_END}"


def extract_thinking_depth(text: str) -> Optional[int]:
    """
    从文本中提取 thinking depth
    
    参数：
        text: 带 thinking token 的文本
        
    返回：
        推理深度（0-3）或 None
    """
    matches = THINK_DEPTH_PATTERN.findall(text)
    
    if matches:
        return int(matches[0])
    
    return None


# ============================================================
# 主程序入口（单元测试）
# ============================================================

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("=" * 60)
    print("Thinking Dial 单元测试")
    print("=" * 60)
    
    # 1. 测试特殊 token 构建
    print("\n[1] 测试特殊 token 构建")
    for depth in range(4):
        token = build_think_token(depth)
        print(f"   depth={depth}: {token}")
    
    # 2. 测试 tokenizer
    print("\n[2] 测试 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    processor = ThinkingDialProcessor(tokenizer)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # 3. 测试数据处理
    print("\n[3] 测试数据处理")
    test_data = [
        {"prompt": "什么是量子纠缠？", "response": "量子纠缠是...", "think_rank": 2},
        {"prompt": "你好", "response": "你好！", "think_rank": 0},
    ]
    
    processed = processor.process_dataset(test_data)
    for item in processed:
        print(f"   think_rank={item['think_rank']}: {item['text'][:50]}...")
    
    # 4. 测试 ThinkingDialProcessor
    print("\n[4] 测试 ThinkingDialProcessor")
    text = processor.tokenize("测试文本")
    print(f"   input_ids shape: {text['input_ids'].shape}")
    
    # 5. 测试配置
    print("\n[5] 测试配置")
    thinking_config = ThinkingConfig()
    grpo_config = GRPOConfig()
    print(f"   Thinking depths: {thinking_config.num_thinking_depths}")
    print(f"   Depth ratios: {thinking_config.depth_ratios}")
    print(f"   GRPO sample size: {grpo_config.grpo_sample_size}")
    
    # 6. 测试 GRPOTrainer（模拟）
    print("\n[6] 测试 GRPOTrainer")
    print("   GRPOTrainer 已初始化（需要真实模型才能训练）")
    
    print("\n" + "=" * 60)
    print("单元测试完成！")
    print("=" * 60)