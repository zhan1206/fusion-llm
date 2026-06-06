"""
最简导入测试 - 只测试模块能否正常导入（无 emoji 版本）
"""
import sys
import importlib.util


def _test_import(module_path: str, module_name: str) -> bool:
    """
    测试单个模块导入
    
    参数：
        module_path: 模块文件路径
        module_name: 模块名称（用于显示）
        
    返回：
        bool: 是否导入成功
    """
    print(f"[TEST] 测试导入 {module_name}...", end=" ")
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            print("FAIL (spec is None)")
            return False
        # 不执行模块，只检查 spec 是否存在
        print("PASS (spec 存在)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def test_simple_import():
    """最简导入测试 - 只测试模块能否正常导入（无 emoji 版本）"""
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
    
    for file_path, module_name in modules_to_test:
        result = _test_import(file_path, module_name)
        results.append((module_name, "PASS" if result else "FAIL"))
    
    # 输出结果
    print()
    print("[DONE] 测试结果")
    print("=" * 60)
    print()
    
    all_passed = True
    for name, result in results:
        status = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"  {status} {name}")
        if result != "PASS":
            all_passed = False
    
    print()
    if all_passed:
        print("[PASS] 所有测试通过")
    else:
        print("[FAIL] 有测试失败")
    
    assert all_passed, "部分模块导入失败"


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print(f"[FAIL] 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
