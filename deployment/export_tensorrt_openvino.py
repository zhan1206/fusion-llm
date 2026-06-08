"""
TensorRT and OpenVINO export for Fusion models.

Export pipeline:
  PyTorch → ONNX → TensorRT (via trtexec or TensorRT Python API)
  PyTorch → ONNX → OpenVINO (via openvino Python API)

Usage:
    python -m deployment.export_tensorrt_openvino \
        --model_path ./output/mini_model \
        --export tensorrt --output_dir ./deployment/output

Requirements (optional, only needed for the target backend):
    - TensorRT: pip install tensorrt onnx onnxruntime
    - OpenVINO: pip install openvino onnx onnxruntime

Author: zhan1206
Project: Fusion-LLM
License: Apache 2.0
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Step 1: PyTorch → ONNX (shared by both backends)
# ============================================================

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    seq_len: int = 32,
    opset_version: int = 14,
    dynamic_batch: bool = True,
) -> str:
    """Export a Fusion model to ONNX format.

    Args:
        model: The Fusion model (FusionModel or FusionMini).
        output_path: Path to write the .onnx file.
        seq_len: Sequence length for the export trace.
        opset_version: ONNX opset version.
        dynamic_batch: Whether to use dynamic batch dimension.

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    vocab_size = model.config.vocab_size
    dummy_input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    # Dynamic axes for variable-length sequences
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        }

    logger.info(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    logger.info(f"ONNX export complete: {output_path}")
    return str(output_path)


# ============================================================
# Step 2a: ONNX → TensorRT
# ============================================================

def export_to_tensorrt(
    onnx_path: str,
    output_dir: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    max_workspace_mb: int = 4096,
) -> Optional[str]:
    """Convert an ONNX model to TensorRT engine.

    Tries TensorRT Python API first, falls back to trtexec CLI.

    Args:
        onnx_path: Path to the ONNX model file.
        output_dir: Directory to write the TensorRT engine.
        fp16: Enable FP16 precision.
        int8: Enable INT8 precision (requires calibration data).
        max_batch_size: Maximum batch size for the engine.
        max_workspace_mb: Maximum workspace size in MB.

    Returns:
        Path to the TensorRT engine, or None if TensorRT is not available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = output_dir / "fusion_model.engine"

    # Try Python API first
    try:
        import tensorrt as trt

        logger.info("Using TensorRT Python API...")

        logger_obj = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_obj)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger_obj)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                return None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_mb * 1024 * 1024)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 enabled (ensure calibration data is provided)")

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input_ids",
            (1, 1),
            (max_batch_size, 32),
            (max_batch_size, 2048),
        )
        profile.set_shape(
            "attention_mask",
            (1, 1),
            (max_batch_size, 32),
            (max_batch_size, 2048),
        )
        config.add_optimization_profile(profile)

        logger.info("Building TensorRT engine (this may take a while)...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            logger.error("Failed to build TensorRT engine")
            return None

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

        logger.info(f"TensorRT engine saved: {engine_path}")
        return str(engine_path)

    except ImportError:
        logger.info("TensorRT Python API not available, trying trtexec CLI...")

    # Fallback: trtexec CLI
    import subprocess
    import shutil

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        # Try common paths
        for candidate in ["/usr/bin/trtexec", "/usr/local/bin/trtexec"]:
            if Path(candidate).exists():
                trtexec = candidate
                break

    if trtexec is None:
        logger.warning(
            "Neither TensorRT Python API nor trtexec found. "
            "Install TensorRT: https://developer.nvidia.com/tensorrt"
        )
        return None

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={max_workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")

    logger.info(f"Running trtexec: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"trtexec failed:\n{result.stderr}")
        return None

    logger.info(f"TensorRT engine saved: {engine_path}")
    return str(engine_path)


# ============================================================
# Step 2b: ONNX → OpenVINO
# ============================================================

def export_to_openvino(
    onnx_path: str,
    output_dir: str,
    fp16: bool = True,
) -> Optional[str]:
    """Convert an ONNX model to OpenVINO IR format.

    Args:
        onnx_path: Path to the ONNX model file.
        output_dir: Directory to write the OpenVINO IR files.
        fp16: Compress weights to FP16.

    Returns:
        Path to the OpenVINO XML model file, or None if OpenVINO is not available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from openvino.tools.mo import convert_model
        from openvino.runtime import serialize

        logger.info("Using OpenVINO Model Optimizer (Python API)...")

        ov_model = convert_model(onnx_path)

        xml_path = output_dir / "fusion_model.xml"
        bin_path = output_dir / "fusion_model.bin"

        # FP16 compression
        if fp16:
            from openvino.runtime import Core
            core = Core()
            # compress_weights_to_fp16 is available in OpenVINO 2023.1+
            try:
                from openvino.tools.mo import compress_weights
                ov_model = compress_weights(ov_model)
                logger.info("Weights compressed to FP16")
            except (ImportError, Exception) as e:
                logger.info(f"FP16 compression skipped: {e}")

        serialize(ov_model, str(xml_path))
        logger.info(f"OpenVINO IR saved: {xml_path} + {bin_path}")
        return str(xml_path)

    except ImportError:
        logger.info("OpenVINO Python API not available, trying mo CLI...")

    # Fallback: mo CLI
    import subprocess
    import shutil

    mo = shutil.which("mo")
    if mo is None:
        logger.warning(
            "Neither OpenVINO Python API nor mo CLI found. "
            "Install OpenVINO: pip install openvino"
        )
        return None

    cmd = [
        sys.executable, "-m", "openvino.tools.mo",
        f"--input_model={onnx_path}",
        f"--output_dir={output_dir}",
    ]
    if fp16:
        cmd.append("--compress_weights_to_fp16=True")

    logger.info(f"Running Model Optimizer: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Model Optimizer failed:\n{result.stderr}")
        return None

    xml_path = output_dir / "fusion_model.xml"
    if xml_path.exists():
        logger.info(f"OpenVINO IR saved: {xml_path}")
        return str(xml_path)

    return None


# ============================================================
# End-to-end export pipeline
# ============================================================

def export_model(
    model_path: str,
    export_format: str = "tensorrt",
    output_dir: str = "./deployment/output",
    seq_len: int = 32,
    fp16: bool = True,
) -> Optional[str]:
    """End-to-end export: load Fusion model → ONNX → target format.

    Args:
        model_path: Path to the Fusion model directory (HF format).
        export_format: "tensorrt" or "openvino".
        output_dir: Directory for exported files.
        seq_len: Sequence length for ONNX trace.
        fp16: Enable FP16 in target format.

    Returns:
        Path to the final exported model, or None on failure.
    """
    from models.fusion_model import FusionModel

    logger.info(f"Loading model from {model_path}...")
    model = FusionModel.from_pretrained(model_path)
    model.eval()

    # Step 1: Export to ONNX (intermediate)
    onnx_path = Path(output_dir) / "fusion_model.onnx"
    export_to_onnx(model, str(onnx_path), seq_len=seq_len)

    # Step 2: Convert to target format
    if export_format == "tensorrt":
        return export_to_tensorrt(str(onnx_path), output_dir, fp16=fp16)
    elif export_format == "openvino":
        return export_to_openvino(str(onnx_path), output_dir, fp16=fp16)
    else:
        raise ValueError(f"Unknown export format: {export_format}. Use 'tensorrt' or 'openvino'.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Export Fusion model to TensorRT/OpenVINO")
    parser.add_argument("--model_path", required=True, help="Path to Fusion model directory")
    parser.add_argument("--export", choices=["tensorrt", "openvino", "onnx"],
                       default="tensorrt", help="Export format")
    parser.add_argument("--output_dir", default="./deployment/output", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length for ONNX trace")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    if args.export == "onnx":
        from models.fusion_model import FusionModel
        model = FusionModel.from_pretrained(args.model_path)
        model.eval()
        onnx_path = Path(args.output_dir) / "fusion_model.onnx"
        export_to_onnx(model, str(onnx_path), seq_len=args.seq_len)
        print(f"ONNX model exported: {onnx_path}")
    else:
        result = export_model(
            args.model_path,
            export_format=args.export,
            output_dir=args.output_dir,
            seq_len=args.seq_len,
            fp16=not args.no_fp16,
        )
        if result:
            print(f"Export complete: {result}")
        else:
            print(f"Export failed. Install {args.export} and try again.")
            sys.exit(1)


if __name__ == "__main__":
    main()
