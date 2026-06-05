# Changelog

All notable changes to the Fusion-LLM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 实际模型训练脚本（`train/train_real.py`）
- 训练数据准备脚本（`data/prepare_training_data.py`）
- 使用教程（`docs/tutorial.md`）
- API 文档（`docs/API.md`）
- 边缘情况测试（`tests/test_edge_cases.py`）

### Fixed
- 无

### Changed
- 无

### Removed
- 无

## [1.0.0] - 2026-06-05

### Added
- **核心功能**：
  - SBLA 注意力（Sliding Block Latent Attention）
  - Thinking Dial（动态推理强度控制）
  - DyQuant（动态混合精度量化，4/8/16-bit）
  - KV Cache 支持
  - GQA（Grouped Query Attention）支持

- **模型**：
  - FusionMini 模型（迷你版 Fusion-LLM）
  - FusionModel 模型（完整版 Fusion-LLM）
  - FusionConfig 配置类
  - FusionMiniConfig 配置类

- **训练**：
  - 全量微调（`train/full_finetune.py`）
  - LoRA 微调（`train/lora_finetune.py`）
  - DPO 对齐训练（`train/dpo_finetune.py`）
  - 知识蒸馏训练（`data_pipeline/t_kd_distillation_train.py`）
  - GRPO 训练（`models/thinking_dial.py`）

- **推理**：
  - 基本推理（`tests/test_inference_basic.py`）
  - Ollama 部署（`inference/ollama_deploy_v2.py`）
  - 动态量化推理（`inference/dyquant.py`）
  - 推理仪表板（`inference/dashboard.py`）

- **评估**：
  - 评估指标（`evaluation/metrics.py` - Perplexity/Loss/Accuracy/BLEU/ROUGE）
  - 模型卡片生成器（`evaluation/model_card.py`）
  - 量化工具（`evaluation/quantization_tool.py`）

- **测试**：
  - 极小配置测试（`tests/test_tiny.py`）
  - 导入测试（`tests/test_simple_import.py`）
  - 基本推理测试（`tests/test_inference_basic.py`）
  - 基本训练测试（`tests/test_training_basic.py`）
  - 最小训练测试（`train/test_train_mini.py`）
  - 小训练测试（`train/train_10steps.py`）
  - 实际模型训练（`train/train_real.py`）
  - 边缘情况测试（`tests/test_edge_cases.py`）

- **CI/CD**：
  - GitHub Actions 工作流（`.github/workflows/ci.yml`）

- **文档**：
  - 使用教程（`docs/tutorial.md`）
  - API 文档（`docs/API.md`）

### Fixed
- **v9 缺陷修复**（9 项）：
  - F1: lora_finetune 签名缺少 vocab_size_override
  - F2: full_finetune 缺少 Optional 导入
  - F3: 两个 JSON 配置 sbla_mode="mixed" -> "hybrid"
  - S1: run_tests.py 4 处断裂调用
  - M1: tokenizer get_effective_vocab_size 硬编码 50257
  - N1: fusion-mini-config hidden_act="gelu" -> "silu"

- **v10 缺陷修复**（11 项）：
  - F-NEW-1: ollama_deploy_v2 st.save_model() -> st.save_file()
  - F-NEW-2: dashboard.py F.softmax NameError
  - F-NEW-3: dyquant load_model() 无返回值
  - F-NEW-4: t_kd_distillation vocab 维度不匹配
  - F-NEW-5: bilingual_filter import langid
  - S-NEW-3: dyquant _insert_fake_quant API 误用
  - S-NEW-4: 删除 scripts/fix_thinking_dial*.py
  - M-NEW-1: dedup_mini_data 完全覆盖去重数据
  - M-NEW-4: dyquant 异常吞噬
  - MI-NEW-2: dashboard tokenizer 初始化耦合 Thinking Dial
  - MI-NEW-3: ollama_deploy_v2 硬编码 C:/D: 路径

- **v11 缺陷修复**（5 项）：
  - S-NEW-1: dashboard token 计数 len(tensor[0]) 取 batch 而非 seq
  - S-NEW-2: dyquant convert() 依赖 load_model 副作用
  - M-NEW-2: 4 个重叠数据脚本合并为 manage_mini_data.py
  - M-NEW-3: 全项目 emoji 替换为 ASCII 标签
  - MI-NEW-1: test_sbla_integration has_sblla 拼写

- **v12 缺陷修复**（13 项）：
  - F-NEW-6: QATrainer.prepare() load_model 返回 None 时崩溃
  - F-NEW-7: ollama_deploy_v2.py 缺少 Optional 导入
  - S-NEW-5: save() 用 HF save_pretrained 无法序列化 QuantizedLinear
  - S-NEW-6: fallback 单分支保存随机权重
  - S-NEW-7: QATrainer.save() 双重保存导致 QAT 权重被覆盖
  - M-NEW-5: _insert_fake_quant 仅匹配 LLaMA 层名
  - M-NEW-6: ollama_deploy_v2 fallback 不处理分片模型
  - M-NEW-8: get_model_size 对 QuantizedLinear 缺少 weight 保护
  - M-NEW-9: train() 未检查 prepare() 返回值
  - M-NEW-10: _load_dataset 使用字节编码而非 tokenizer 编码
  - MI-NEW-4: manage_mini_data.py DATA_PATH 依赖 CWD
  - MI-NEW-5: bilingual_filter/ollama_deploy [LOGO] 残留
  - MI-NEW-6: ollama_deploy.py check_dependencies 缺少 shell=True

### Changed
- **API 变更**：
  - `FusionModel.forward()` 和 `FusionMini.forward()` 移除掩码预转换，由 `SBLAttention` 统一处理
  - `SBLAttention.forward()` 统一处理原始 HF 格式掩码
  - `ThinkingDialProcessor` 兼容 HuggingFace 接口
  - `GRPOTrainer` 重写（奖励函数、组相对优势）

- **配置变更**：
  - `fusion-mini-config.json` 更新 `transformers_version`
  - `fusion-mini-config.json` 删除冗余的 `think_rank` 字段
  - `ds_zero3.json` 调整 `sub_group_size` 为 1e9

- **字段名统一**：
  - 统一 `window_size`/`sbla_window_size` 字段名为 `window_size`

### Removed
- **删除文件**：
  - `scripts/debug_*.py`（5 个调试脚本）
  - `scripts/fix_thinking_dial*.py`（硬编码个人路径）
  - `scripts/create_mini_data.py`
  - `scripts/fix_mini_data.py`
  - `scripts/dedup_mini_data.py`
  - `scripts/add_depth3_samples.py`

### Security
- 无

### Deprecated
- 无

## [0.0.1] - 2026-05-30

### Added
- 项目初始化
- 目录结构搭建
- 迷你模型 MVP 训练（Loss 2.80 → 1.37）
- SBLA 注意力集成
- 第一个 GitHub 提交

---

## 发布说明

### v1.0.0 - 首个正式版本

**发布日期**：2026-06-05

**发布说明**：
Fusion-LLM v1.0.0 是首个正式版本，包含完整的 LLM 训练、推理、评估功能。

**主要特性**：
1. **SBLA 注意力**：滑动分块潜注意力
2. **Thinking Dial**：动态推理强度控制
3. **DyQuant**：动态混合精度量化
4. **完整训练流程**：全量微调、LoRA 微调、DPO 对齐训练、知识蒸馏
5. **完整推理流程**：Ollama 部署、动态量化推理
6. **完整评估流程**：Perplexity/Loss/Accuracy/BLEU/ROUGE
7. **完整测试覆盖**：8 个测试文件，覆盖所有核心功能
8. **完整文档**：使用教程 + API 文档

**缺陷修复**：
- 修复 v9-v12 全部缺陷（共 38 项）
- 所有测试通过
- 训练有效（Loss 持续下降）

**已知问题**：
- SBLA 注意力计算慢（正在优化）
- 无多 GPU 训练支持（正在开发）
- 无 Flash Attention 支持（正在开发）

**下一步计划**：
- 优化 SBLA 注意力速度
- 添加 Flash Attention 支持
- 添加多 GPU 训练支持
- 添加更多评估指标（BERTScore、MoverScore）
- 添加模型可视化工具
- 添加更多部署选项（GGML、ONNX）

---

## 版本号说明

- **主版本号**：不兼容的 API 修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

---

## 链接

- **GitHub 仓库**：https://github.com/zhan1206/fusion-llm
- **问题追踪**：https://github.com/zhan1206/fusion-llm/issues
- **讨论区**：https://github.com/zhan1206/fusion-llm/discussions

---

## 致谢

感谢所有贡献者！

---

**注意**：此文件使用 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 格式。
