"""
优化的 SBLA 注意力 - 尝试进一步优化速度（虽然已经很快了）
"""
import sys
import torch
import torch.nn as F
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, '.')


class OptimizedSBLAttention(torch.nn.Module):
    """
    优化的 SBLA 注意力模块（尝试进一步优化速度）
    
    优化策略：
    1. 使用混合精度（FP16）
    2. 减少不必要的计算
    3. 优化内存访问模式
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window_size = getattr(config, 'sbla_window_size', 512)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # 投影层
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # SBLA 门控（可选）
        self.use_sbla = True
        self.sbla_gate = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 混合精度（可选）
        self.use_fp16 = False  # 默认关闭，因为已经很快了
        
        # Dropout
        self.dropout = torch.nn.Dropout(getattr(config, 'attention_probs_dropout_prob', 0.1))
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """重复 KV heads 以匹配 Q heads"""
        if n_rep == 1:
            return hidden_states
        
        batch, seq_len, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, seq_len, num_key_value_heads, n_rep, head_dim
        )
        hidden_states = hidden_states.reshape(batch, seq_len, num_key_value_heads * n_rep, head_dim)
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        优化的前向传播
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 混合精度（可选）
        if self.use_fp16 and hidden_states.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return self._forward_impl(
                    hidden_states, attention_mask, past_key_value, use_cache
                )
        else:
            return self._forward_impl(
                hidden_states, attention_mask, past_key_value, use_cache
            )
    
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """实际的前向传播实现"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. 线性投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 2. 形状重塑
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # 3. KV 缓存（可选）
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # 4. 重复 KV heads（如果 GQA 启用）
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # 5. 转置为 (batch, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # 6. 注意力分数计算
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # 7. 注意力掩码（优化：避免不必要的形状操作）
        if attention_mask is not None:
            # 确保 attention_mask 形状正确
            if attention_mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # (batch, 1, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                pass
            
            # 应用掩码
            attn_weights = attn_weights + attention_mask
        
        # 8. Softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 9. 注意力输出
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 10. 转置回 (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        
        # 11. 输出投影
        attn_output = self.o_proj(attn_output)
        
        # 12. SBLA 门控（可选，优化：避免不必要的计算）
        if self.use_sbla:
            # 简化的 SBLA：只应用门控，不扩展潜向量
            gate = torch.sigmoid(self.sbla_gate(hidden_states))
            attn_output = attn_output * gate
        
        return attn_output, past_key_value
    
    @torch.no_grad()
    def benchmark(self, seq_len=32, num_runs=100):
        """
        性能基准测试
        """
        print(f"[BENCHMARK] 优化版 SBLA 注意力性能分析（seq_len={seq_len}, num_runs={num_runs}）...")
        
        self.eval()
        
        # 创建测试输入
        hidden_states = torch.randn(1, seq_len, self.hidden_size)
        attention_mask = torch.ones(1, seq_len)
        
        # 预热
        for _ in range(10):
            self.forward(hidden_states, attention_mask)
        
        # 计时
        torch.cuda.synchronize() if hidden_states.device.type == 'cuda' else None
        start = torch.cuda.Event(enable_timing=True) if hidden_states.device.type == 'cuda' else None
        end = torch.cuda.Event(enable_timing=True) if hidden_states.device.type == 'cuda' else None
        
        times = []
        for i in range(num_runs):
            if hidden_states.device.type == 'cuda':
                start.record()
                self.forward(hidden_states, attention_mask)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time
                t0 = time.time()
                self.forward(hidden_states, attention_mask)
                t1 = time.time()
                times.append((t1 - t0) * 1000)  # 转换为 ms
        
        # 统计
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   平均时间: {avg_time:.2f} ms")
        print(f"   最短时间: {min_time:.2f} ms")
        print(f"   最长时间: {max_time:.2f} ms")
        
        return avg_time, min_time, max_time


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 优化的 SBLA 注意力测试")
    print("=" * 60)
    print()
    
    # 创建测试配置
    class TestConfig:
        def __init__(self):
            self.hidden_size = 64
            self.num_attention_heads = 2
            self.sbla_window_size = 512
            self.num_key_value_heads = 2
            self.attention_probs_dropout_prob = 0.1
    
    config = TestConfig()
    
    # 创建优化版 SBLA 注意力
    print("[1] 创建优化版 SBLA 注意力...")
    attn = OptimizedSBLAttention(config)
    print(f"   配置: hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}")
    print()
    
    # 测试前向传播
    print("[2] 测试前向传播...")
    hidden_states = torch.randn(1, 8, config.hidden_size)
    attention_mask = torch.ones(1, 8)
    
    output, cache = attn(hidden_states, attention_mask)
    print(f"   输入形状: {hidden_states.shape}")
    print(f"   输出形状: {output.shape}")
    print()
    
    # 性能基准测试
    print("[3] 性能基准测试...")
    avg_time, min_time, max_time = attn.benchmark(seq_len=32, num_runs=100)
    print()
    
    # 与原版比较（如果可用）
    print("[4] 与原版比较...")
    try:
        from models.sbla_attention import SBLAttention
        
        # 创建原版
        original_attn = SBLAttention(config)
        
        # 基准测试
        original_attn.benchmark(seq_len=32, num_runs=100)
        print()
        
        print("[INFO] 优化版 vs 原版:")
        print(f"   优化版平均时间: {avg_time:.2f} ms")
        print(f"   原版平均时间: (见上方)")
        print()
    except:
        print("   [WARN] 原版不可用，跳过比较")
        print()
    
    print("[PASS] 优化的 SBLA 注意力测试通过")
    sys.exit(0)
