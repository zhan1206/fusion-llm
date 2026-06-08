"""
ONNX 部署 — 将 FusionModel 导出为 ONNX 格式

使用方式:
    python deployment/export_onnx.py --checkpoint output/mini_model --output output/onnx/model.onnx
    python deployment/export_onnx.py --checkpoint output/mini_model --output output/onnx/ --dynamic-batch

依赖: onnx, onnxruntime (pip install onnx onnxruntime)
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, '.')


def export_to_onnx(model, output_path, dynamic_batch=False, dynamic_seq=True, opset=14):
    """Export FusionModel to ONNX format.
    
    Args:
        model: FusionModel instance
        output_path: Output .onnx file path
        dynamic_batch: Enable dynamic batch dimension
        dynamic_seq: Enable dynamic sequence length
        opset: ONNX opset version
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnx and onnxruntime required. Install with:")
        print("  pip install onnx onnxruntime")
        return False
    
    import torch
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input
    batch_size = 1
    seq_len = 16
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Dynamic axes
    dynamic_axes = {}
    if dynamic_batch or dynamic_seq:
        input_axes = {}
        if dynamic_batch:
            input_axes[0] = "batch_size"
        if dynamic_seq:
            input_axes[1] = "seq_len"
        dynamic_axes = {
            "input_ids": input_axes,
            "attention_mask": input_axes,
            "logits": input_axes,
        }
    
    # Export
    print(f"[EXPORT] Exporting to ONNX (opset={opset})...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        opset_version=opset,
        do_constant_folding=True,
    )
    
    # Verify
    print("[VERIFY] Checking ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    # Test with ONNX Runtime
    print("[VERIFY] Running ONNX Runtime inference test...")
    session = ort.InferenceSession(str(output_path))
    ort_inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)
    
    # Compare with PyTorch output
    with torch.no_grad():
        pt_outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, return_dict=True)
    
    import numpy as np
    max_diff = np.max(np.abs(ort_outputs[0] - pt_outputs.logits.numpy()))
    print(f"[VERIFY] Max diff between PyTorch and ONNX Runtime: {max_diff:.6e}")
    
    if max_diff < 1e-4:
        print(f"[PASS] ONNX export verified: {output_path}")
    else:
        print(f"[WARN] Diff {max_diff:.6e} exceeds 1e-4, check model compatibility")
    
    # Save model metadata
    metadata = {
        "model_type": "fusion",
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "dynamic_batch": dynamic_batch,
        "dynamic_seq": dynamic_seq,
        "onnx_opset": opset,
    }
    import json
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[DONE] ONNX export complete: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FusionModel to ONNX format")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="output/onnx/model.onnx", help="Output ONNX path")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch dim")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()
    
    from models.fusion_model import FusionModel
    model = FusionModel.from_pretrained(args.checkpoint)
    export_to_onnx(model, args.output, args.dynamic_batch, opset=args.opset)
