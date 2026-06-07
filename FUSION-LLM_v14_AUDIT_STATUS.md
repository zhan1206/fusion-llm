# Fusion-LLM v14 审计修复状态报告

## 已修复并推送 (commit c1899e9)

| ID | 级别 | 描述 | 文件 | 修复方式 |
|----|------|------|------|----------|
| F1 | 致命 | ThinkingDialModel thinking_embedding/gate 创建但从未使用 | thinking_dial.py | 用深度 embedding 投影到 vocab 空间注入 logits bias |
| F5 | 致命 | generate top_p 全 mask 时 NaN 风险 | fusion_model.py | fallback 到原始 logits |
| M1 | 中等 | generate() 增量推理中 RoPE position_ids 不递增 | fusion_model.py | 添加 past_seq_len 计数器 |
| M6 | 中等 | position_ids 未传递到 FusionLayer | fusion_model.py | 完整传递链: FusionModel→FusionLayer→FusionAttention |
| FWD | Bug | FusionModel.forward() 缺 position_ids 参数 | fusion_model.py | 添加参数并从 kwargs 提取 |
| M-NEW-11 | 中等 | DPOTrainer 缺 tokenizer 属性 | dpo_finetune.py | 添加 @property tokenizer 懒加载 |

## 待架构决策

### F2 (致命) - SBLA 增量推理退化
```
sbla_attention.py:547
if past_key_value is not None and seq_len <= 1:
    output = output_std  # 跳过 SBLA block latent → 退化为纯标准注意力
```
**问题**：增量生成时跳过 SBLA 块潜向量计算，等于没有 SBLA。
**可能的修复**：即使单个 token 也计算其 block latent contribution（需要 block_size 个 token 才有完整窗口，但可以计算该 token 对已有 block 的贡献）。
**状态**：需架构决策。

### F4 (致命) - 双注意力入口
**问题**：`FusionAttention` 和 `SBLAttention` 都可作为注意力入口，架构冗余。
**可能的修复**：
1. 统一为单入口（推荐）：删除 `SBLAttention.forward()` 公共接口，仅保留 `forward_with_qkv()`
2. 保留双入口但加明确区分
**状态**：需架构决策。

## 文档/告知性问题（用户需决策）

| ID | 级别 | 描述 | 建议 |
|----|------|------|------|
| README | - | num_layers vs num_hidden_layers 参数名错误 | 用户手动修正 |
| S6 | 严重 | HuggingFace Trainer 配置兼容性 | 用户决定是否适配 |
| M3 | 中等 | generate() 返回 torch.Tensor 而非 CausalLMOutputWithPast | 告知性 |
| M4 | 中等 | generate() 无 temperature/topp/bottomp 验证 | 告知性 |
| S2 | 严重 | 蒸馏依赖外部权重 | 设计决策 |

## 测试状态
- 测试套件: **12/12 通过**
- 语法检查: 全部通过
- 最新 commit: `c1899e9` (pushed to master)
