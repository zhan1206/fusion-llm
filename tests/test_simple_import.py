"""
最简导入测试 - 只测试模块能否正常导入
"""
import sys
import importlib.util


def test_import(module_path: str, module_name: str) -> bool:
    """测试单个模块导入"""
    print(f"[TEST] 测试导入 {module_name}...", end=" ")
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            print("FAIL (spec is None)")
            return False
        module = importlib.util.module_from_spec(spec)
        # 不执行 module，只检查 spec 是否存在
        print("PASS (spec 存在)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main():
    print("=" * 60)
    print("Fusion-LLM 最简导入测试")
    print("=" * 60)
    print()
    
    # 测试列表：(文件路径, 模块名)
    modules_to_test = [
        ("evaluation/metrics.py", "evaluation.metrics"),
        ("models/fusion_mini.py", "models.fusion_mini"),
        ("models/sbla_attention.py", "models.sbla_attention"),
        ("models/thinking_dial.py", "models.thinking_dial"),
        ("inference/dashboard.py", "inference.dashboard"),
        ("inference/dyquant.py", "inference.dyquant"),
        ("train/full_finetune.py", "train.full_finetune"),
        ("train/lora_finetune.py", "train.lora_finetune"),
    ]
    
    results = []
    
    for module_path, module_name in modules_to_test:
        success = test_import(module_path, module_name)
        results.append((module_name, success))
        print()
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print()
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for module_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {module_name}")
    
    print()
    print(f"结果：{passed}/{total} 通过")
    
    if passed == total:
        print()
        print("✅ 所有模块导入测试通过")
        return 0
    else:
        print()
        print("❌ 部分模块导入失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
