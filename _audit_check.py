import sys; sys.path.insert(0, '.')
import inspect, torch, os, yaml

issues = []

# 1. Thinking Dial dual mechanism
from models.thinking_dial import ThinkingDialModel, ThinkingDialProcessor
from models.fusion_model import FusionModel, FusionConfig

src = inspect.getsource(ThinkingDialProcessor.process_single)
has_text_injection = 'think_token' in src
print(f'1. ThinkingDialProcessor text injection: {has_text_injection}')
has_embedding = hasattr(ThinkingDialModel, 'thinking_embedding')
has_gate = hasattr(ThinkingDialModel, 'thinking_gate')
print(f'   ThinkingDialModel embedding+gate: {has_embedding and has_gate}')

# 2. Ollama
import inference.ollama_deploy_v2 as ollama_mod
methods = [m for m in dir(ollama_mod) if not m.startswith('_') and callable(getattr(ollama_mod, m, None))]
print(f'2. ollama_deploy_v2 functions: {methods}')

# 3. Deployment scripts
for f in ['deployment/export_ggml.py', 'deployment/export_onnx.py', 'deployment/export_tensorrt_openvino.py']:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f'   {f}: exists={exists}, size={size} bytes')

# 4. Quantization - check what exists
import glob
qfiles = glob.glob('**/*quantiz*', recursive=True)
print(f'3. Quantization files: {qfiles}')
try:
    from evaluation.quantization_tool import QuantConfig
    print(f'   QuantConfig exists: True')
except ImportError as e:
    print(f'   QuantConfig import failed: {e}')

# 5. GRPOTrainer reward
from models.thinking_dial import GRPOTrainer
has_reg = hasattr(GRPOTrainer, 'REWARD_FUNCTIONS')
print(f'4. GRPOTrainer REWARD_FUNCTIONS: {has_reg}')
if has_reg:
    print(f'   Keys: {list(GRPOTrainer.REWARD_FUNCTIONS.keys())}')

# 6. model_registry URLs
with open('configs/model_registry.yaml', encoding='utf-8') as f:
    reg = yaml.safe_load(f)
for name, info in reg['models'].items():
    url = info.get('url', '')
    rel = info.get('released_at', '?')
    print(f'5. {name}: url_empty={not url}, released={rel}')

# 7. BilingualFilter
from data_pipeline.bilingual_filter import BilingualTrueFilter
methods = [m for m in dir(BilingualTrueFilter) if not m.startswith('_')]
print(f'6. BilingualTrueFilter: {methods}')

# 8. Tests
test_dir = 'tests'
test_files = sorted([f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')])
print(f'7. Tests: {len(test_files)} files -> {test_files}')

# 9. Check README claims
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()
claims = []
for keyword in ['4.2x', 'SBLA', 'Thinking Dial', 'benchmark', 'accuracy', 'F1']:
    if keyword.lower() in readme.lower():
        claims.append(keyword)
print(f'8. README performance claims: {claims}')

# 10. Check generate_with_thinking
src = inspect.getsource(FusionModel.generate)
has_thinking = 'thinking_depth' in src
print(f'9. FusionModel.generate supports thinking_depth: {has_thinking}')

# 11. KV cache in generate
has_kv = 'past_key_values' in src
print(f'10. FusionModel.generate supports KV cache: {has_kv}')

# 12. Check evaluation module
eval_files = [f for f in os.listdir('evaluation') if f.endswith('.py')]
print(f'11. Evaluation files: {eval_files}')
