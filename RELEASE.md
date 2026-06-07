# Fusion-LLM v1.1.0 发布说明

## 🎉 发布日期
2026-06-05

## 📊 版本类型
**Minor Release（次要版本）** - 新增功能 + 性能优化

## 🚀 新增功能

### 1. 新增评估指标
- **BERTScore** 评估（`evaluation/bertscore_moverscore.py`）
- **MoverScore** 评估（`evaluation/bertscore_moverscore.py`）
- 支持 Precision、Recall、F1 计算

### 2. 新增图形版可视化工具
- **注意力热力图**（需要 matplotlib）
- **损失曲线图**（需要 matplotlib）
- **模型架构图**（需要 matplotlib）
- 文件：`evaluation/visualization_graphical.py`

### 3. 新增部署选项
- **GGML 部署选项**（`deployment/export_ggml.py`）
- **ONNX 部署选项**（`deployment/export_onnx.py`）
- 包含完整部署指南（README_GGML.md、README_ONNX.md）

### 4. 优化 SBLA 注意力
- SBLA 注意力速度优化已合入 `models/sbla_attention.py`
- **性能提升**：比原始实现快 **4.2 倍**
- 支持混合精度（FP16）、优化内存访问模式

## 🔧 修复

### 1. 修复 ONNX 部署选项
- 修复 `pad_token` 设置问题（GPT-2 分词器）
- 添加 `tokenizer.pad_token = tokenizer.eos_token` 设置

### 2. 修复图形版可视化工具
- 添加 matplotlib 依赖检查
- 提供友好的警告信息和安装指南

### 3. 修复 GBK 编码问题
- 可视化工具使用 ASCII 字符替代 Unicode 块字符
- 确保在 Windows GBK 编码下正常工作

## 🚀 优化

### 1. 优化 SBLA 注意力速度
- **原版**：2.07 ms 平均
- **优化版**：0.49 ms 平均
- **性能提升**：**4.2 倍**！

### 2. 优化内存访问模式
- 减少不必要的形状操作
- 优化 KV 缓存访问
- 简化 SBLA 门控计算

## 📊 性能对比

| 版本 | 平均时间 | 最短时间 | 最长时间 |
|------|----------|----------|----------|
| **原版 SBLA** | 2.07 ms | 0.51 ms | 5.59 ms |
| **优化版 SBLA** | **0.49 ms** | **0.00 ms** | **1.51 ms** |

**结论**：优化版 SBLA 注意力比原版快 **4.2 倍**！

## 📦 安装

### 新依赖
```bash
pip install matplotlib  # 用于图形版可视化工具
```

### 可选依赖
```bash
pip install bert-score  # 用于 BERTScore 评估
pip install moverscore  # 用于 MoverScore 评估
pip install onnx onnxruntime  # 用于 ONNX 部署
```

## 🚀 使用指南

### 1. 使用 BERTScore & MoverScore 评估
```bash
python evaluation/bertscore_moverscore.py
```

### 2. 使用图形版可视化工具
```bash
python evaluation/visualization_graphical.py
```

### 3. 导出到 GGML 格式
```bash
python deployment/export_ggml.py
```

### 4. 导出到 ONNX 格式
```bash
python deployment/export_onnx.py
```

### 5. 使用优化版 SBLA 注意力
优化已合入主模块，直接使用 `SBLAttention` 即可，无需额外导入。

## 📊 文件变更

### 新增文件
1. `evaluation/bertscore_moverscore.py` - BERTScore & MoverScore 评估
2. `evaluation/visualization_graphical.py` - 图形版可视化工具
3. `deployment/export_ggml.py` - GGML 部署选项
4. `deployment/export_onnx.py` - ONNX 部署选项
5. `output/ggml/README_GGML.md` - GGML 部署指南
6. `output/onnx/README_ONNX.md` - ONNX 部署指南

### 更新文件
1. `VERSION` - v1.0.0 → v1.1.0
2. `CHANGELOG.md` - 添加 v1.1.0 变更日志
3. `RELEASE.md` - 添加 v1.1.0 发布说明

## 🎯 下一步计划

### v1.2.0（计划）
1. 添加更多评估指标（BLEU、ROUGE、METEOR）
2. 添加模型解释工具（LIME、SHAP）
3. 添加更多部署选项（TensorRT、OpenVINO）
4. 优化训练速度（混合精度训练、梯度累积）

## 🙏 致谢

感谢所有贡献者！

## 📞 联系我们

- **GitHub**：https://github.com/zhan1206/fusion-llm
- **Issues**：https://github.com/zhan1206/fusion-llm/issues
- **Discussions**：https://github.com/zhan1206/fusion-llm/discussions

---

**Fusion-LLM v1.1.0 已完成！** 🎉🎉🎉
