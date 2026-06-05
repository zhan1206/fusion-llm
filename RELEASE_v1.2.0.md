# Fusion-LLM v1.2.0 Release Notes

## New Features

### Evaluation Metrics
- **BLEU/ROUGE/METEOR** (`evaluation/bleu_rouge_meteor.py`) - Standard NLP evaluation suite with precision, recall, F1

### Deployment Options
- **TensorRT** (`deployment/export_tensorrt_openvino.py`) - GPU-optimized inference via ONNX to TensorRT engine
- **OpenVINO** (`deployment/export_tensorrt_openvino.py`) - CPU-optimized inference via ONNX to OpenVINO IR

### Training Optimization
- **AMP Trainer** (`train/training_optimizations.py`) - Automatic Mixed Precision training with gradient scaling
- **Gradient Accumulation** (`train/training_optimizations.py`) - Simulate larger batch sizes on limited VRAM

### Model Interpretability
- **LIME** (`evaluation/model_interpretability.py`) - Token-level importance via perturbation
- **SHAP** (`evaluation/model_interpretability.py`) - Shapley value-based token contribution analysis

## How to Create This Release on GitHub

1. Go to https://github.com/zhan1206/fusion-llm/releases/new
2. Tag: **v1.2.0**
3. Title: **Fusion-LLM v1.2.0**
4. Copy the content above as the description
5. Click **Publish release**
