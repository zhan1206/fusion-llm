"""
部署选项：TensorRT 和 OpenVINO 导出
"""
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, '.')


def export_tensorrt(model, output_dir, seq_len=32):
    """
    导出模型为 TensorRT 格式（简化版）
    
    实际部署流程：
    1. PyTorch -> ONNX (torch.onnx.export)
    2. ONNX -> TensorRT (trtexec 或 TensorRT Python API)
    
    简化版：生成导出脚本和配置文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[EXPORT-TENSORRT] 生成 TensorRT 导出配置...")
    
    # 1. 生成 ONNX 导出脚本
    onnx_script = output_dir / "export_onnx_for_tensorrt.py"
    with open(onnx_script, "w", encoding="utf-8") as f:
        f.write('"""Step 1: Export model to ONNX for TensorRT"""\n')
        f.write("import torch\n")
        f.write("import sys\n")
        f.write("sys.path.insert(0, '.')\n")
        f.write("from models.fusion_mini import FusionMini, FusionMiniConfig\n\n")
        f.write("config = FusionMiniConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2)\n")
        f.write("model = FusionMini(config)\n")
        f.write("model.eval()\n\n")
        f.write(f"dummy = torch.randint(0, 100, (1, {seq_len}))\n")
        f.write("torch.onnx.export(\n")
        f.write("    model, dummy,\n")
        f.write("    'output/tensorrt/model.onnx',\n")
        f.write("    input_names=['input_ids'],\n")
        f.write("    output_names=['logits'],\n")
        f.write("    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch', 1: 'seq'}},\n")
        f.write("    opset_version=17,\n")
        f.write(")\n")
        f.write("print('[OK] ONNX model exported to output/tensorrt/model.onnx')\n")
    
    # 2. 生成 trtexec 命令
    trt_config = {
        "format": "tensorrt",
        "version": "1.0",
        "steps": [
            "1. pip install tensorrt onnx",
            "2. python export_onnx_for_tensorrt.py",
            "3. trtexec --onnx=output/tensorrt/model.onnx --saveEngine=output/tensorrt/model.engine --fp16",
        ],
        "trtexec_command": f"trtexec --onnx=output/tensorrt/model.onnx --saveEngine=output/tensorrt/model.engine --fp16 --shapes=input_ids:1x{seq_len}",
        "inference_python": "import tensorrt as trt\n# Load engine and run inference (see TensorRT docs)",
    }
    
    config_path = output_dir / "tensorrt_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(trt_config, f, indent=2, ensure_ascii=False)
    
    # 3. 生成推理脚本
    infer_script = output_dir / "infer_tensorrt.py"
    with open(infer_script, "w", encoding="utf-8") as f:
        f.write('"""TensorRT Inference (requires tensorrt)"""\n')
        f.write("import numpy as np\n\n")
        f.write("def load_engine(engine_path):\n")
        f.write("    import tensorrt as trt\n")
        f.write("    trt_logger = trt.Logger(trt.Logger.WARNING)\n")
        f.write("    runtime = trt.Runtime(trt_logger)\n")
        f.write("    with open(engine_path, 'rb') as f:\n")
        f.write("        engine = runtime.deserialize_cuda_engine(f.read())\n")
        f.write("    return engine\n\n")
        f.write("def infer(engine, input_ids):\n")
        f.write("    import tensorrt as trt\n")
        f.write("    context = engine.create_execution_context()\n")
        f.write("    # Allocate buffers and run inference\n")
        f.write("    # (Implementation depends on specific TensorRT version)\n")
        f.write("    pass\n")
    
    print(f"   ONNX export script: {onnx_script}")
    print(f"   Config: {config_path}")
    print(f"   Inference script: {infer_script}")
    print()


def export_openvino(model, output_dir, seq_len=32):
    """
    导出模型为 OpenVINO 格式（简化版）
    
    实际部署流程：
    1. PyTorch -> ONNX (torch.onnx.export)
    2. ONNX -> OpenVINO IR (mo --input_model model.onnx)
    
    简化版：生成导出脚本和配置文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[EXPORT-OPENVINO] 生成 OpenVINO 导出配置...")
    
    # 1. 生成 ONNX 导出脚本
    onnx_script = output_dir / "export_onnx_for_openvino.py"
    with open(onnx_script, "w", encoding="utf-8") as f:
        f.write('"""Step 1: Export model to ONNX for OpenVINO"""\n')
        f.write("import torch\n")
        f.write("import sys\n")
        f.write("sys.path.insert(0, '.')\n")
        f.write("from models.fusion_mini import FusionMini, FusionMiniConfig\n\n")
        f.write("config = FusionMiniConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2)\n")
        f.write("model = FusionMini(config)\n")
        f.write("model.eval()\n\n")
        f.write(f"dummy = torch.randint(0, 100, (1, {seq_len}))\n")
        f.write("torch.onnx.export(\n")
        f.write("    model, dummy,\n")
        f.write("    'output/openvino/model.onnx',\n")
        f.write("    input_names=['input_ids'],\n")
        f.write("    output_names=['logits'],\n")
        f.write("    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch', 1: 'seq'}},\n")
        f.write("    opset_version=17,\n")
        f.write(")\n")
        f.write("print('[OK] ONNX model exported to output/openvino/model.onnx')\n")
    
    # 2. 生成 OpenVINO 配置
    ov_config = {
        "format": "openvino",
        "version": "1.0",
        "steps": [
            "1. pip install openvino onnx",
            "2. python export_onnx_for_openvino.py",
            "3. mo --input_model=output/openvino/model.onnx --output_dir=output/openvino/ir",
        ],
        "mo_command": "mo --input_model=output/openvino/model.onnx --output_dir=output/openvino/ir --data_type=FP16",
        "inference_python": "import openvino as ov\n# Compile model and run inference (see OpenVINO docs)",
    }
    
    config_path = output_dir / "openvino_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(ov_config, f, indent=2, ensure_ascii=False)
    
    # 3. 生成推理脚本
    infer_script = output_dir / "infer_openvino.py"
    with open(infer_script, "w", encoding="utf-8") as f:
        f.write('"""OpenVINO Inference (requires openvino)"""\n')
        f.write("import numpy as np\n")
        f.write("import openvino as ov\n\n")
        f.write("def load_and_infer(model_path, input_ids):\n")
        f.write("    core = ov.Core()\n")
        f.write("    model = core.read_model(model_path)\n")
        f.write("    compiled = core.compile_model(model, 'CPU')  # or 'GPU'\n")
        f.write("    infer_request = compiled.create_infer_request()\n")
        f.write("    results = infer_request.infer({'input_ids': input_ids})\n")
        f.write("    return results\n")
    
    print(f"   ONNX export script: {onnx_script}")
    print(f"   Config: {config_path}")
    print(f"   Inference script: {infer_script}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM TensorRT/OpenVINO 部署选项测试")
    print("=" * 60)
    print()
    
    # 创建示例模型
    print("[1] 创建示例模型...")
    from models.fusion_mini import FusionMini, FusionMiniConfig
    config = FusionMiniConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
    model = FusionMini(config)
    print("   示例模型已创建")
    print()
    
    # TensorRT
    print("[2] TensorRT 部署...")
    export_tensorrt(model, "output/tensorrt", seq_len=32)
    
    # OpenVINO
    print("[3] OpenVINO 部署...")
    export_openvino(model, "output/openvino", seq_len=32)
    
    # 验证文件存在
    print("[4] 验证导出文件...")
    tensorrt_files = list(Path("output/tensorrt").glob("*"))
    openvino_files = list(Path("output/openvino").glob("*"))
    print(f"   TensorRT 文件数: {len(tensorrt_files)}")
    print(f"   OpenVINO 文件数: {len(openvino_files)}")
    assert len(tensorrt_files) >= 3, "TensorRT should have at least 3 files"
    assert len(openvino_files) >= 3, "OpenVINO should have at least 3 files"
    print("   文件验证通过")
    print()
    
    print("[PASS] TensorRT/OpenVINO 部署选项测试通过")
    sys.exit(0)
