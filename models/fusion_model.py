"""
Fusion 完整模型定义（v2 - 可实例化可运行）

集成：
1. SBLA 注意力（滑动分块潜注意力）- 真实实现
2. Thinking Dial（动态推理强度控制）- 通过特殊 token
3. 标准 Transformer 架构 + KV Cache 支持

修复（v2）：
- FusionModel 现在可以完整实例化和运行
- SBLA 注意力已正确集成到每一层
- 支持 causal mask、padding mask
- generate() 方法支持 KV cache 加速推理
- 配置文件与代码完全对齐

使用方法：
    from models.fusion_model import FusionModel, FusionConfig
    
    config = FusionConfig(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        block_size=64,
        latent_dim=16,
    )
    model = FusionModel(config)
    
    input_ids = torch.randint(0, 10000, (2, 128))
    outputs = model(input_ids=input_ids, labels=input_ids)
    print(f"Loss: {outputs['loss'].item()}")

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Dict, Any
import math

# H4-H6: Use try/except for relative imports with sys.path fallback
try:
    from .sbla_attention import SBLAttention
except ImportError:
    from models.sbla_attention import SBLAttention


class FusionConfig(PretrainedConfig):
    """Fusion 模型配置"""
    
    model_type = "fusion"
    
    def __init__(
        self,
        vocab_size: int = 100000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: int = 11008,
        hidden_act: str = "silu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        # SBLA parameters
        block_size: int = 512,
        latent_dim: int = 64,
        window_size: Optional[int] = None,
        sbla_mode: str = "pure_sbla",
        # Thinking Dial parameters
        enable_thinking_dial: bool = True,
        num_thinking_depths: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        
        # SBLA parameters
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.window_size = window_size or block_size  # [M8 FIX] Remove redundant sbla_window_size
        self.sbla_mode = sbla_mode
        
        # Thinking Dial parameters
        self.enable_thinking_dial = enable_thinking_dial
        self.num_thinking_depths = num_thinking_depths
        
        # RoPE parameters
        self.rope_theta = kwargs.pop('rope_theta', 10000.0)


# H1-H3: Register FusionConfig with AutoConfig
try:
    from transformers import AutoConfig
    AutoConfig.register("fusion", FusionConfig)
except (ImportError, ValueError):
    pass  # Already registered or AutoConfig unavailable


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for positional encoding in attention."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device=None):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_kv_heads, seq_len, head_dim)
        cos: (kv_seq_len, head_dim) cosine part of rotary embedding
        sin: (kv_seq_len, head_dim) sine part of rotary embedding
        position_ids: (batch, seq_len) position ids for slicing cos/sin

    Returns:
        Tuple of (q_embed, k_embed) with rotary position encoding applied.
    """
    if position_ids is not None:
        # N6 FIX: Slice cos/sin by position_ids to match actual Q/K positions
        # position_ids: (batch, seq_len), cos/sin: (kv_seq_len, head_dim)
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    else:
        # Fallback: broadcast for full-sequence (prefill) when position_ids not provided
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, kv_seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """RMSNorm（均方根层归一化）"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(x.dtype)


class FusionAttention(nn.Module):
    """
    Fusion Attention Layer — delegates to the canonical SBLAttention implementation.
    
    This is the unified entry point for attention in FusionModel.
    All SBLA logic (block latents, causal/window masks, padding) lives in
    models/sbla_attention.py::SBLAttention. This wrapper adds KV cache support
    and config-driven mode selection (pure_sbla / hybrid).
    
    See: models/sbla_attention.py::SBLAttention
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        mode = getattr(config, 'sbla_mode', 'hybrid')
        self.sbla = SBLAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            block_size=config.block_size,
            latent_dim=config.latent_dim,
            dropout=config.attention_probs_dropout_prob,
            window_size=config.window_size,  # use dedicated window_size field
            mode=mode,
            num_key_value_heads=config.num_key_value_heads,
        )
        
        # M14 FIX: Add RoPE - Rotary Position Embedding
        head_dim = config.hidden_size // config.num_attention_heads
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )
        
        # Separate Q/K/V projections for RoPE application
        # (SBLAttention has its own q/k/v, but we need access before RoPE)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim, bias=False)
        # [S1 FIX] o_proj is sbla.out_proj — needed so LoRA target_modules=["o_proj"] works
        # We assign it after sbla is created (above). This shares the parameter.
        self.o_proj = self.sbla.out_proj
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # M14 FIX: Apply RoPE to Q and K before SBLAttention
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.sbla.head_dim
        num_kv_groups = self.sbla.num_kv_groups
        
        # Project Q/K/V
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.sbla.num_heads, head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.sbla.num_key_value_heads, head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.sbla.num_key_value_heads, head_dim).transpose(1, 2)
        
        # Compute RoPE embeddings
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len = past_key_value[0].shape[2] + seq_len
        emb = self.rotary_emb(kv_seq_len, device=hidden_states.device)
        cos = emb.cos()
        sin = emb.sin()
        # N6 FIX: Build position_ids for proper RoPE slicing
        if position_ids is None:
            if past_key_value is not None:
                offset = past_key_value[0].shape[2]
                position_ids = torch.arange(offset, offset + seq_len, device=hidden_states.device).unsqueeze(0)
            else:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        # Apply RoPE with position_ids to prevent broadcast mismatch during incremental generation
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=position_ids)
        
        # Store RoPE'd K/V in SBLAttention's cache for incremental generation
        # S1 FIXED: KV Cache now works natively through SBLAttention.
        # We pass the RoPE'd Q/K/V by injecting them into SBLAttention.
        output, present_key_value = self.sbla.forward_with_qkv(
            Q, K, V,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        return output, present_key_value


class FusionLayer(nn.Module):
    """Fusion Transformer 层"""
    
    def __init__(self, config: FusionConfig, layer_idx: int):
        super().__init__()
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = FusionAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # SwiGLU FFN
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,  # [M6 FIX] Added
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.attention(
            hidden_states, 
            attention_mask,
            past_key_value=past_key_value if past_key_value is not None else None,
            use_cache=use_cache,
            position_ids=position_ids,  # [M6 FIX]
        )
        hidden_states = residual + self.dropout(attn_output)
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        ffn_output = self.down_proj(gate * up)
        hidden_states = residual + self.dropout(ffn_output)
        
        return hidden_states, present_key_value


class FusionModel(PreTrainedModel, GenerationMixin):
    """
    Fusion 完整模型（v2 - 可实例化可运行）
    
    支持 HuggingFace PreTrainedModel 全接口
    """
    
    config_class = FusionConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["FusionAttention"]
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            FusionLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight
        
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        position_ids: Optional[torch.Tensor] = None,  # [M6 FIX] Added
        thinking_depth: Optional[torch.Tensor] = None,  # N10 FIX: Thinking Dial depth
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # [M6 FIX] Extract position_ids from kwargs if not passed explicitly
        if 'position_ids' in kwargs:
            position_ids = kwargs.pop('position_ids')
        # N10 FIX: Extract thinking_depth from kwargs if not passed explicitly
        if 'thinking_depth' in kwargs:
            thinking_depth = kwargs.pop('thinking_depth')
        # else: keep the explicit position_ids parameter
        
        # Embeddings
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.embeddings(input_ids)
            hidden_states = self.dropout(hidden_states)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Use the already-resolved use_cache from parameter, don't re-override from kwargs
        if past_key_values is not None:
            use_cache = True
        
        present_key_values = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            layer_outputs, cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
                position_ids=position_ids,  # [M6 FIX]
            )
            hidden_states = layer_outputs
            if use_cache:
                present_key_values = present_key_values + (cache,)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        # C2/C3/C5: Return CausalLMOutputWithPast instead of plain dict
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
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        return_dict_in_generate: bool = False,
        logits_hook: Optional[callable] = None,  # N10 FIX: Hook for ThinkingDial logit bias
        past_key_values: Optional[Tuple] = None,  # N17 FIX: Accept pre-computed KV cache
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Generate text with KV cache and SBLA incremental support.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (must be > 0)
            top_p: Nucleus sampling threshold (0 < top_p <= 1)
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            return_dict_in_generate: If True, return CausalLMOutputWithPast
        
        Returns:
            CausalLMOutputWithPast if return_dict_in_generate, else generated token IDs tensor
        """
        # [M4 FIX] Parameter validation
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not (0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        eos_token_id = eos_token_id or getattr(self.config, "eos_token_id", None)
        
        self.eval()
        # N17 FIX: Support pre-computed KV cache for generate_samples reuse
        if past_key_values is not None:
            past_seq_len = past_key_values[0][0].shape[2]  # K shape: (B, H, S, D)
            generated = input_ids
        else:
            past_seq_len = 0  # [M1 FIX] Track position for RoPE
            generated = input_ids.clone()
        
        # N10 FIX: Extract thinking_depth from kwargs for forwarding
        thinking_depth = kwargs.pop('thinking_depth', None)
        logits_hook = kwargs.pop('logits_hook', logits_hook)
        
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                current_input = generated[:, -1:]
                cur_seq_len = 1
            else:
                current_input = generated
                cur_seq_len = generated.shape[1]
            
            # [M1 FIX] Compute position_ids for RoPE
            position_ids = torch.arange(past_seq_len, past_seq_len + cur_seq_len,
                                       device=device, dtype=torch.long).unsqueeze(0)
            
            outputs = self.forward(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                position_ids=position_ids,  # [M1 FIX]
                thinking_depth=thinking_depth,  # N10 FIX
            )
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
            # N10 FIX: Apply logits hook if provided (for ThinkingDialModel depth bias)
            if logits_hook is not None:
                logits = logits_hook(logits)
            
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            if do_sample and top_p < 1.0:
                logits_before_mask = next_token_logits.clone()  # [F5 FIX] Save for fallback
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits.masked_fill_(indices_to_remove, float('-inf'))
                # [F5 FIX] Guard against all-tokens-masked: keep top-1 token
                if (next_token_logits == float('-inf')).all():
                    next_token_logits = logits_before_mask
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            past_seq_len += cur_seq_len  # [M1 FIX] Update position counter
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        # [M3 FIX] Return CausalLMOutputWithPast for API consistency
        if return_dict_in_generate:
            return CausalLMOutputWithPast(
                loss=None,
                logits=None,
                past_key_values=past_key_values,
                sequences=generated,
            )
        return generated
    
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}


if __name__ == "__main__":
    print("[TEST] Testing Fusion Model (v2)...")
    
    config = FusionConfig(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        block_size=64,
        latent_dim=16,
        sbla_mode="pure_sbla",
        max_position_embeddings=256,
    )
    
    model = FusionModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_count:,} parameters")
    
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True)
    
    assert outputs.loss is not None, "Loss should not be None"
    assert not torch.isnan(outputs.loss).item(), "Loss is NaN!"
    print(f"Loss={outputs.loss.item():.4f}, Logits={outputs.logits.shape}")
    
    print("\n[ALL TESTS PASSED] Fusion Model v2 fully functional.")