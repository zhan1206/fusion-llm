"""
Fusion Mini - 可运行的最小化模型

这是一个简化但**完整可运行**的 Fusion 模型实现，用于验证整个流程。

包含：
1. 标准 Transformer 架构（暂时不用 SBLA）
2. 基础 Thinking Dial 控制（通过 token 注入）
3. 完整的训练、推理接口

使用方法：
    from models.fusion_mini import FusionMini, FusionMiniConfig
    
    # 创建 mini 模型
    config = FusionMiniConfig(
        vocab_size=10000,      # 小词表
        hidden_size=128,        # 小隐层
        num_hidden_layers=4,    # 少层数
        num_attention_heads=4,  # 少注意力头
    )
    
    model = FusionMini(config)
    
    # 测试前向传播
    input_ids = torch.randint(0, 10000, (2, 64))
    outputs = model.forward(input_ids=input_ids, labels=input_ids)
    print(f"Loss: {outputs['loss'].item()}")
    
    # 推理
    generated = model.generate(input_ids[:, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional, Tuple
import math
import json
from pathlib import Path

# 导入 SBLA 注意力
from .sbla_attention import SBLAttention
from .fusion_model import RMSNorm


class FusionMiniConfig(PretrainedConfig):
    """
    Fusion Mini 配置
    
    极简配置，用于快速验证流程
    """
    
    model_type = "fusion_mini"
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 128,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 512,
        hidden_act: str = "silu",
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # Thinking Dial 参数
        enable_thinking_dial: bool = True,
        num_thinking_depths: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # Thinking Dial
        self.enable_thinking_dial = enable_thinking_dial
        self.num_thinking_depths = num_thinking_depths
        
    @classmethod
    def from_pretrained(cls, config_path: str, **kwargs):
        """
        从配置文件加载
        """
        config_file = Path(config_path) / "config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            return cls(**config_dict)
        
        raise FileNotFoundError(f"配置文件未找到：{config_file}")


class FusionMiniEmbeddings(nn.Module):
    """
    Fusion Mini 词嵌入
    """
    
    def __init__(self, config: FusionMiniConfig):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
        )
        
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        
        self.LayerNorm = RMSNorm(
            config.hidden_size,
            eps=1e-6,
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        参数：
            input_ids: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        word_embeds = self.word_embeddings(input_ids)
        
        # 位置编码
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        position_embeds = self.position_embeddings(position_ids)
        
        # 合并
        embeddings = word_embeds + position_embeds
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class FusionMiniAttention(nn.Module):
    """
    Fusion Mini 注意力层（标准多头注意力）
    """
    
    def __init__(self, config: FusionMiniConfig):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数：
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, 1, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 线性投影
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context = torch.matmul(attention_probs, v)
        
        # 重塑回原始形状
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        
        # 输出线性层
        output = self.out(context)
        
        return output


class FusionMiniLayer(nn.Module):
    """
    Fusion Mini Transformer 层
    
    Unified with FusionModel: uses RMSNorm + SwiGLU FFN
    """
    
    def __init__(self, config: FusionMiniConfig):
        super().__init__()
        
        # Input RMSNorm (pre-norm, same as FusionModel)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-6)
        
        # SBLA Attention
        self.sbla_attention = SBLAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            block_size=64,
            latent_dim=config.hidden_size // 8,
            dropout=0.1,
        )
        
        # Post-attention RMSNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-6)
        
        # SwiGLU FFN (same as FusionModel)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm + SBLA Attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.dropout(self.sbla_attention(hidden_states, attention_mask))
        
        # Pre-norm + SwiGLU FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = residual + self.dropout(self.down_proj(gate * up))
        
        return hidden_states


class FusionMini(PreTrainedModel):
    """
    Fusion Mini 完整模型
    
    极简实现，用于验证完整流程
    """
    
    config_class = FusionMiniConfig
    
    def __init__(self, config: FusionMiniConfig):
        super().__init__(config)
        
        self.config = config
        
        # 1. Embeddings
        self.embeddings = FusionMiniEmbeddings(config)
        
        # 2. Transformer 层
        self.layers = nn.ModuleList([
            FusionMiniLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # 3. Layer Norm（最后一层后）
        self.ln_f = RMSNorm(config.hidden_size, eps=1e-6)
        
        # 4. LM Head
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """
        初始化权重
        """
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        权重初始化
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        参数：
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len)（用于训练）
            use_cache: 是否使用 KV 缓存（推理时）
            return_dict: 是否返回字典格式
            
        返回：
            (loss), logits, ...
        """
        # 1. Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # 2. 处理 attention_mask
        if attention_mask is not None:
            # 转换为 (batch, 1, 1, seq_len) 格式
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 3. Transformer 层
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
        
        # 4. 最后一层 Layer Norm
        hidden_states = self.ln_f(hidden_states)
        
        # 5. LM Head
        logits = self.lm_head(hidden_states)
        
        # 6. 计算损失（如果有 labels）
        loss = None
        if labels is not None:
            # 移位：预测下一个 token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        if return_dict:
            return {"loss": loss, "logits": logits}
        
        return (loss, logits)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs,
    ):
        """
        生成文本（简化版本）
        
        参数：
            input_ids: (batch, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: nucleus sampling
            do_sample: 是否采样
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        self.eval()
        
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(
                input_ids=generated,
                use_cache=False,
                return_dict=True,
            )
            
            logits = outputs["logits"]
            
            # 取最后一个 token 的 logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 散回原始顺序
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # 采样或贪婪解码
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 追加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否生成 EOS
            if kwargs.get("eos_token_id") is not None:
                if (next_token == kwargs["eos_token_id"]).all():
                    break
            
            # 更新 input_ids（简化：实际应使用 KV 缓存）
            input_ids = generated
        
        return generated


if __name__ == "__main__":
    # 单元测试
    print("🧪 测试 Fusion Mini 模型...")
    
    # 创建配置
    config = FusionMiniConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )
    
    print(f"✅ 配置创建成功")
    print(f"   词表大小：{config.vocab_size}")
    print(f"   隐层大小：{config.hidden_size}")
    print(f"   层数：{config.num_hidden_layers}")
    
    # 创建模型
    model = FusionMini(config)
    
    print(f"\n✅ 模型创建成功")
    print(f"   参数量：{sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,  # 自监督
        return_dict=True,
    )
    
    print(f"\n✅ 前向传播测试通过")
    print(f"   Loss: {outputs['loss'].item():.4f}")
    print(f"   Logits 形状: {outputs['logits'].shape}")
    
    # 测试生成
    generated = model.generate(
        input_ids=input_ids[:, :10],  # 只用前 10 个 token
        max_new_tokens=20,
    )
    
    print(f"\n✅ 生成测试通过")
    print(f"   生成形状: {generated.shape}")
    
    print("\n🎉 Fusion Mini 测试完成！")
    print("\n💡 下一步：")
    print("   1. 使用真实数据训练这个 mini 模型")
    print("   2. 验证训练流程")
    print("   3. 然后实现 SBLA 和 Thinking Dial")
