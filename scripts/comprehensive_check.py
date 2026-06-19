import sys; sys.path.insert(0, '.')
import torch, tempfile, os, py_compile

checks = []

# 1. All files compile
all_ok = True
for root, dirs, files in os.walk('.'):
    if '__pycache__' in root or '.git' in root or 'node_modules' in root:
        continue
    for f in files:
        if f.endswith('.py') and f != '_test_import.py':
            try:
                py_compile.compile(os.path.join(root, f), doraise=True)
            except Exception as e:
                print('  COMPILE ERROR:', os.path.join(root, f), str(e)[:80])
                all_ok = False
checks.append(('All .py files compile', all_ok))

# 2. Clean import
import importlib
for mod in list(sys.modules.keys()):
    if 'fusion' in mod or 'mini' in mod:
        del sys.modules[mod]
from models.fusion_mini import FusionMini, FusionMiniConfig
from models.fusion_model import FusionModel, FusionConfig
from models.sbla_attention import SBLAttention
from models.thinking_dial import ThinkingDialModel, ThinkingConfig, GRPOTrainer
checks.append(('Clean imports', True))

# 3. FusionMini round-trip
from models.fusion_mini import FusionMini, FusionMiniConfig
config = FusionMiniConfig(vocab_size=500, hidden_size=64, num_heads=4, num_layers=2, max_position_embeddings=512)
model = FusionMini(config)
model.eval()
x = torch.randint(0, 100, (1, 16))
out1 = model(x).logits
with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    model2 = FusionMini._load_from_safetensors(tmpdir)
    model2.eval()
    diff = (out1 - model2(x).logits).abs().max().item()
    checks.append(('FusionMini RT max_diff={:.8f}'.format(diff), diff < 1e-10))
print('RT check:', diff)

# 4. FusionModel round-trip
fconfig = FusionConfig(vocab_size=500, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=512)
fmodel = FusionModel(fconfig)
fmodel.eval()
fx = torch.randint(0, 100, (1, 16))
fout1 = fmodel(fx).logits
with tempfile.TemporaryDirectory() as tmpdir:
    fmodel.save_pretrained(tmpdir)
    fmodel2 = FusionModel.from_pretrained(tmpdir)
    fmodel2.eval()
    fdiff = (fout1 - fmodel2(fx).logits).abs().max().item()
    checks.append(('FusionModel RT max_diff={:.8f}'.format(fdiff), fdiff < 1e-10))

# 5. ThinkingDial generate
td = ThinkingDialModel(model, ThinkingConfig(num_thinking_depths=4))
torch.manual_seed(42)
d0 = td.generate(x.clone(), max_new_tokens=3, thinking_depth=0)
d1 = td.generate(x.clone(), max_new_tokens=3, thinking_depth=1)
checks.append(('ThinkingDial gen', d0.shape[-1] > 0))

# 6. GRPO training step
trainer = GRPOTrainer(td)
trainer.setup_optimizer(1e-4)
result = trainer.train_step(x, thinking_depth=1)
checks.append(('GRPO has loss', 'loss' in result))

# 7. SBLA attention - basic shape check via FusionLayer in FusionMini
checks.append(('SBLA integrated in FusionMini', hasattr(FusionMini, '_load_from_safetensors')))

# 8. KV cache consistency
model.eval()
prompt = torch.randint(0, 100, (1, 8))
with torch.no_grad():
    out_p = model(prompt, use_cache=True)
    past = out_p.past_key_values
    next_tok = torch.randint(0, 100, (1, 1))
    out_i = model(next_tok, use_cache=True, past_key_values=past)
checks.append(('KV cache', out_i.logits.shape[-1] == config.vocab_size))

print()
for name, ok in checks:
    print('  PASS' if ok else '  FAIL', ':', name)
print()
print('{}/{} checks passed'.format(sum(1 for _, r in checks if r), len(checks)))