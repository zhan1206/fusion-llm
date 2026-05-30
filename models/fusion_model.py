"""
Fusion 完整模型定义

集成：
1. SBLA 注意力（滑动分块潜注意力）
2. Thinking Dial（动态推理强度控制）
3. 标准 Transformer 架构

使用方法：
    from models.fusion_model import FusionModel, FusionConfig
    
    config = FusionConfig.from_pretrained("fusion-8b")
    model = FusionModel(config)
    
    # 或从头训练
    config = FusionConfig(
        vocab_size=100000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
    )
    model = FusionModel(config)

作者：朱子瞻
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from .sbla_attention import SlidingBlockLatentAttention, FusionAttentionBlock
from .thinking_dial import ThinkingDialProcessor, ThinkingConfig
import math
from typing import Optional, Tuple, List
import json
import os


class FusionConfig(PretrainedConfig):
    """
    Fusion 模型配置
    
    继承自 HuggingFace PretrainedConfig，支持 from_pretrained()
    """
    
    model_type = "fusion"
    
    def __init__(
        self,
        vocab_size: int = 100000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        intermediate_size: int = 11008,
        hidden_act: str = "silu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # SBLA 参数
        block_size: int = 512,
        latent_dim: int = 64,
        window_size: int = 2048,
        # Thinking Dial 参数
        enable_thinking_dial: bool = True,
        num_thinking_depths: int = 4,  # 0-3
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # SBLA 参数
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.window_size = window_size
        
        # Thinking Dial 参数
        self.enable_thinking_dial = enable_thinking_dial
        self.num_thinking_depths = num_thinking_depths
        
    @classmethod
    def from_pretrained(cls, config_path: str, **kwargs):
        """
        从配置文件加载
        """
        config_file = os.path.join(config_path, "config.json")
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            return cls(**config_dict)
        
        raise FileNotFoundError(f"配置文件未找到：{config_file}")


class FusionEmbeddings(nn.Module):
    """
    Fusion 词嵌入 + 位置编码
    """
    
    def __init__(self, config: FusionConfig):
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
        
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=1e-12,
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
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


class FusionLayer(nn.Module):
    """
    Fusion Transformer 层
    
    集成 SBLA 注意力 + Thinking Dial（在 embedding 层处理）
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        # SBLA 注意力块
        self.attention = FusionAttentionBlock(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.attention_probs_dropout_prob,
            block_size=config.block_size,
            latent_dim=config.latent_dim,
        )
        
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
        # SBLA 注意力 + FFN（在 FusionAttentionBlock 内实现）
        hidden_states = self.attention(hidden_states, attention_mask)
        
        return hidden_states


class FusionModel(PreTrainedModel):
    """
    Fusion 完整模型
    
    架构：
    1. Embeddings（词嵌入 + 位置编码）
    2. Transformer 层（SBLA 注意力）
    3. LM Head（语言模型头）
    
    支持 Thinking Dial（通过特殊 token 在输入中控制）
    """
    
    config_class = FusionConfig
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        
        self.config = config
        
        # 1. Embeddings
        self.embeddings = FusionEmbeddings(config)
        
        # 2. Transformer 层
        self.layers = nn.ModuleList([
            FusionLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # 3. Layer Norm（最后一层后）
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-12)
        
        # 4. LM Head
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        
        # 初始化权重
        self.init_weights()
        
        # Thinking Dial 处理器（如果有）
        if config.enable_thinking_dial:
            self.thinking_processor = ThinkingDialProcessor(self)
        
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
        elif isinstance(module, nn.LayerNorm):
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
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
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
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        """
        生成文本（简化版本，实际应使用 HuggingFace GenerationMixin）
        
        参数：
            input_ids: (batch, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: nucleus sampling
            do_sample: 是否采样
        """
        # 简化实现：实际应使用 HuggingFace 的 generate()
        # 这里只提供框架
        
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
                    torch.softmax(sorted_logits, dim=-1), dim=-1
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
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 追加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否生成 EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # 更新 input_ids（简化：实际应使用 KV 缓存）
            input_ids = generated
        
        return generated


if __name__ == "__main__":
    # 单元测试
    print("🧪 测试 Fusion 完整模型...")
    
    # 创建配置
    config = FusionConfig(
        vocab_size=100000,
        hidden_size=512,  # 小型测试
        num_hidden_layers=4,
        num_attention_heads=8,
        block_size=128,  # 小 block 用于测试
        latent_dim=32,
    )
    
    print(f"✅ 配置创建成功")
    print(f"   隐层大小：{config.hidden_size}")
    print(f"   层数：{config.num_hidden_layers}")
    print(f"   SBLA block_size：{config.block_size}")
    
    # 创建模型
    model = FusionModel(config)
    
    print(f"\n✅ 模型创建成功")
    print(f"   参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 256
    
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
    
    print("\n🎉 Fusion 模型测试完成！")
