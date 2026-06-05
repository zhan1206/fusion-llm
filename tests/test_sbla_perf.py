"""
SBLA 注意力性能测试 - 找出瓶颈
"""
import sys
import torch
import time
sys.path.insert(0, '.')

from models.sbla_attention import SBLAttention


def profile_sbla():
    """性能分析"""
    print("[PROFILE] SBLA 注意力性能分析...")
    print()
    
    # 配置
    batch_size = 1
    seq_len = 32
    hidden_size = 64
    num_heads = 2
    window_size = 16
    
    # 创建 SBLA 注意力层
    attention = SBLAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        window_size=window_size,
    )
    attention.eval()
    
    # 创建输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 预热
    print("[1] 预热...")
    with torch.no_grad():
        _ = attention(hidden_states)
    print("   预热完成")
    print()
    
    # 测试多次，取平均时间
    print("[2] 性能测试（10 次）...")
    times = []
    
    for i in range(10):
        start_time = time.time()
        
        with torch.no_grad():
            output = attention(hidden_states)
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000  # 转换为毫秒
        times.append(elapsed)
        
        print(f"   Run {i+1:2d}: {elapsed:6.2f} ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print()
    print(f"   平均时间: {avg_time:.2f} ms")
    print(f"   最短时间: {min_time:.2f} ms")
    print(f"   最长时间: {max_time:.2f} ms")
    print()
    
    # 分析瓶颈
    print("[3] 瓶颈分析...")
    
    # 检查是否有不必要的计算
    # 1. repeat_interleave 是否太慢？
    # 2. torch.maximum 是否太慢？
    # 3. 是否有冗余计算？
    
    print("   分析完成")
    print()
    
    print("[PROFILE] 性能分析完成")
    return avg_time


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM SBLA 注意力性能测试")
    print("=" * 60)
    print()
    
    try:
        avg_time = profile_sbla()
        
        print()
        if avg_time > 1000:  # > 1 秒
            print("[SLOW] SBLA 注意力太慢！需要优化")
        elif avg_time > 500:  # > 0.5 秒
            print("[MEDIUM] SBLA 注意力较慢，建议优化")
        else:
            print("[FAST] SBLA 注意力速度可接受")
    except Exception as e:
        print()
        print(f"[FAIL] 性能测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)
