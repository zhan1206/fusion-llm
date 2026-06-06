"""检查所有模块导入"""
import sys
import traceback

sys.path.insert(0, '.')

print('Testing imports...')
print('=' * 60)

# Test models
tests = [
    ('models', ['FusionModel', 'FusionConfig']),
    ('models.fusion_mini', ['FusionMini', 'FusionMiniConfig']),
    ('models.sbla_attention', ['SBLAttention']),
    ('models.thinking_dial', ['ThinkingDialProcessor', 'GRPOTrainer']),
    ('models.tokenizer', ['get_tokenizer']),
]

passed = 0
failed = 0

for module_name, items in tests:
    try:
        mod = __import__(module_name, fromlist=items)
        for item in items:
            getattr(mod, item)
        print(f'[OK] {module_name}: {", ".join(items)}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] {module_name}: {e}')
        traceback.print_exc()
        failed += 1

print('=' * 60)
print(f'Passed: {passed}, Failed: {failed}')
print()

# Test model creation
print('Testing model creation...')
try:
    import torch
    from models.fusion_mini import FusionMini, FusionMiniConfig
    
    config = FusionMiniConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    model = FusionMini(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'[OK] FusionMini created, params: {param_count}')
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    outputs = model(input_ids=input_ids, labels=input_ids)
    if hasattr(outputs, 'loss'):
        print(f'[OK] Forward pass, loss: {outputs.loss.item():.4f}')
    else:
        print(f'[FAIL] Output has no loss attribute: {type(outputs)}')
        
except Exception as e:
    print(f'[FAIL] Model creation/test: {e}')
    traceback.print_exc()

print()
print('Done')
