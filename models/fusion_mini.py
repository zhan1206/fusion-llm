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
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple
import math
import json
from pathlib import Path

# H4-H6: Use try/except for relative imports with sys.path fallback
try:
    from .sbla_attention import SBLAttention
    from .fusion_model import RMSNorm
except ImportError:
    from models.sbla_attention import SBLAttention
    from models.fusion_model import RMSNorm


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


# H1-H3: Register FusionMiniConfig with AutoConfig
try:
    from transformers import AutoConfig
    AutoConfig.register("fusion_mini", FusionMiniConfig)
except Exception:
    pass  # Already registered or AutoConfig unavailable


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


class FusionMiniLayer(nn.Module):
    """
    Fusion Mini Transformer 层
    
    Unified with FusionModel: uses RMSNorm + SwiGLU FFN
    """
    
    def __init__(self, config: FusionMiniConfig):
        super().__init__()
        
        # Input RMSNorm (pre-norm, same as FusionModel)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-6)
        
        # Q/K/V projections for SBLA (avoids Q/K/V in SBLAttention)
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
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
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + SBLA Attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Compute Q/K/V projections and reshape to 4D for SBLA
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.sbla_attention.num_heads
        head_dim = self.sbla_attention.head_dim
        num_kv_heads = self.sbla_attention.num_key_value_heads
        kv_head_dim = self.sbla_attention.kv_head_dim
        
        Q = self.query(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        K = self.key(hidden_states).view(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
        V = self.value(hidden_states).view(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
        
        # SBLA attention with forward_with_qkv (avoids Q/K/V projection in SBLAttention)
        attn_output, present_key_value = self.sbla_attention.forward_with_qkv(
            Q, K, V, attention_mask,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # Pre-norm + SwiGLU FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = residual + self.dropout(self.down_proj(gate * up))
        
        return hidden_states, present_key_value


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
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # 1. Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # 2. Transformer layers
        present_key_values = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                present_key_values = present_key_values + (cache,)
        
        # 4. Final Layer Norm
        hidden_states = self.ln_f(hidden_states)
        
        # 5. LM Head
        logits = self.lm_head(hidden_states)
        
        # 6. Compute loss (if labels provided)
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        # C5: Return CausalLMOutputWithPast instead of plain dict
        if not return_dict:
            output = (logits,) + (present_key_values,) if present_key_values is not None else (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=None,
            attentions=None,
        )
    
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
        生成文本（使用 KV Cache 加速）
        """
        generated = input_ids.clone()
        past_key_values = None
        
        self.eval()
        
        for _ in range(max_new_tokens):
            # Use KV cache: only pass last token after first step
            if past_key_values is not None:
                current_input = generated[:, -1:]
                current_mask = torch.ones(generated.shape[0], 1, device=generated.device)
            else:
                current_input = generated
                current_mask = torch.ones_like(generated, dtype=torch.float)
            
            outputs = self.forward(
                input_ids=current_input,
                attention_mask=current_mask,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
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
            
            # Check EOS
            if kwargs.get("eos_token_id") is not None:
                if (next_token == kwargs["eos_token_id"]).all():
                    break
        
        return generated


if __name__ == "__main__":
    # 单元测试
    print("[INFO] 测试 Fusion Mini 模型...")
    
    # 创建配置
    config = FusionMiniConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )
    
    print(f"[OK] 配置创建成功")
    print(f"   词表大小：{config.vocab_size}")
    print(f"   隐层大小：{config.hidden_size}")
    print(f"   层数：{config.num_hidden_layers}")
    
    # 创建模型
    model = FusionMini(config)
    
    print(f"\n[OK] 模型创建成功")
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
    
    print(f"\n[OK] 前向传播测试通过")
    print(f"   Loss: {outputs.loss.item():.4f}")
    print(f"   Logits shape: {outputs.logits.shape}")
    
    # 测试生成
    generated = model.generate(
        input_ids=input_ids[:, :10],  # 只用前 10 个 token
        max_new_tokens=20,
    )
    
    print(f"\n[OK] 生成测试通过")
    print(f"   生成形状: {generated.shape}")
    
    print("\n[DONE] Fusion Mini 测试完成！")
    print("\n[TIP] 下一步：")
    print("   1. 使用真实数据训练这个 mini 模型")
    print("   2. 验证训练流程")
    print("   3. 然后实现 SBLA 和 Thinking Dial")
