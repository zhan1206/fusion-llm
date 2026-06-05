# 变更日志

## [1.1.0] - 2026-06-05

### 新增
- 添加 BERTScore & MoverScore 评估指标（`evaluation/bertscore_moverscore.py`）
- 添加图形版模型可视化工具（`evaluation/visualization_graphical.py`）
- 添加 GGML 部署选项（`deployment/export_ggml.py`）
- 添加 ONNX 部署选项（`deployment/export_onnx.py`）
- 添加优化的 SBLA 注意力（`models/optimized_sbla_attention.py`，0.49 ms）

### 修复
- 修复 ONNX 部署选项中的 `pad_token` 设置问题
- 修复图形版可视化工具的 matplotlib 依赖检查
- 修复 GBK 编码问题（可视化工具使用 ASCII 字符）

### 优化
- 优化 SBLA 注意力速度（0.49 ms vs 原版 2.07 ms）

---

## [1.0.0] - 2026-06-05

### 新增
- 首个正式版本
- 完整文档（使用教程、API 文档）
- 完整测试（8/8 测试文件）
- 实际模型训练（Loss 下降到 1.75）

### 修复
- 修复 v9-v12 共 38 项缺陷

### 文档
- 添加使用教程（`docs/tutorial.md`）
- 添加 API 文档（`docs/API.md`）
- 添加边缘情况测试（`tests/test_edge_cases.py`）
- 添加变更日志（`CHANGELOG.md`）
- 添加发布说明（`RELEASE.md`）
- 添加版本号（`VERSION`）
