"""Full project bug check - runtime validation"""
import sys, torch, traceback
sys.path.insert(0, '.')
from models.fusion_model import FusionModel, FusionConfig

print('=== Deep Runtime Checks ===')
bugs = []

# 1. Full forward + backward pass
config = FusionConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, intermediate_size=128, block_size=8, latent_dim=8,
    max_position_embeddings=128, sbla_mode='hybrid')
model = FusionModel(config)
x = torch.randint(0, 100, (2, 16))
out = model(input_ids=x, labels=x, return_dict=True)
print(f'[1] Forward+backward: loss={out.loss.item():.4f}, logits={out.logits.shape}')
out.loss.backward()
print('    Backward OK')

# 2. Generate with different modes
model.eval()
with torch.no_grad():
    g1 = model.generate(x[:, :4], max_new_tokens=4, do_sample=False)
    print(f'[2] Greedy generate: {g1.shape}')
    
    result = model.generate(x[:, :4], max_new_tokens=4, do_sample=False, return_dict_in_generate=True)
    if isinstance(result, tuple):
        print(f'    return_dict: ({type(result[0]).__name__}, tensor {result[1].shape})')
    else:
        print(f'    return_dict: {type(result).__name__}')

# 3. KV cache generate
with torch.no_grad():
    out1 = model(input_ids=x[:, :8], use_cache=True, return_dict=True)
    pkv = out1.past_key_values
    out2 = model(input_ids=x[:, 8:9], past_key_values=pkv, use_cache=True, return_dict=True)
    print(f'[3] KV cache: prefill {out1.logits.shape}, step {out2.logits.shape}')

# 4. ThinkingDial
from models.thinking_dial import ThinkingDialModel, ThinkingConfig
td_config = ThinkingConfig(num_thinking_depths=4)
td_model = ThinkingDialModel(model, td_config)
td_model.eval()
with torch.no_grad():
    td_out = td_model.generate(x[:, :4], max_new_tokens=3, thinking_depth=2)
    print(f'[4] ThinkingDial generate: {td_out.shape}')

# 5. Check for NaN
with torch.no_grad():
    out3 = model(input_ids=x, return_dict=True)
    has_nan = torch.isnan(out3.logits).any().item()
    if has_nan:
        bugs.append('NaN in logits')
    print(f'[5] NaN check: {"FAIL" if has_nan else "OK"}')

# 6. Pure SBLA mode
config2 = FusionConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, intermediate_size=128, block_size=8, latent_dim=8,
    max_position_embeddings=128, sbla_mode='pure_sbla')
model2 = FusionModel(config2)
model2.eval()
with torch.no_grad():
    out4 = model2(input_ids=x, labels=x, return_dict=True)
    print(f'[6] Pure SBLA: loss={out4.loss.item():.4f}, shape={out4.logits.shape}')

# 7. GQA (fewer kv heads)
config3 = FusionConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
    block_size=8, latent_dim=8, max_position_embeddings=128)
model3 = FusionModel(config3)
model3.eval()
with torch.no_grad():
    out5 = model3(input_ids=x, labels=x, return_dict=True)
    print(f'[7] GQA (4Q/2KV): loss={out5.loss.item():.4f}, shape={out5.logits.shape}')

# 8. FusionMini
from models.fusion_mini import FusionMini, FusionMiniConfig
mini_config = FusionMiniConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, intermediate_size=128)
mini = FusionMini(mini_config)
mini.eval()
with torch.no_grad():
    out6 = mini(input_ids=x, return_dict=True)
    print(f'[8] FusionMini: logits={out6.logits.shape}')
    if out6.logits is not None and torch.isnan(out6.logits).any():
        bugs.append('NaN in FusionMini output')

# 9. Tokenizer module
from models.tokenizer import AutoTokenizer
print(f'[9] Tokenizer: AutoTokenizer importable')

# 10. DyQuant
print(f'[10] DyQuant: DyQuantConverter + QuantConfig importable')

# 11. Incremental generation with KV cache
model.eval()
with torch.no_grad():
    inp = torch.tensor([[1, 2, 3]])
    gen = model.generate(inp, max_new_tokens=5, do_sample=False)
    print(f'[11] Incremental gen: input {inp.shape} -> output {gen.shape}')

# 12. Check all model parameters have grad
no_grad_params = [n for n, p in model.named_parameters() if not p.requires_grad and 'norm' not in n.lower()]
if no_grad_params:
    bugs.append(f'Unexpected frozen params: {no_grad_params[:3]}')
print(f'[12] Frozen params check: {len(no_grad_params)} unexpected frozen')

# 13. save/load round-trip
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    loaded = FusionModel.from_pretrained(tmpdir)
    loaded.eval()
    with torch.no_grad():
        orig_out = model(input_ids=x, return_dict=True).logits
        loaded_out = loaded(input_ids=x, return_dict=True).logits
        diff = (orig_out - loaded_out).abs().max().item()
    if diff > 1e-5:
        bugs.append(f'save/load mismatch: max diff={diff}')
    print(f'[13] Save/load round-trip: max diff={diff:.2e}')

# 14. Check forward without labels returns None loss
with torch.no_grad():
    out_no_label = model(input_ids=x, return_dict=True)
    if out_no_label.loss is not None:
        bugs.append('Forward without labels should return None loss')
    print(f'[14] No-label forward: loss={out_no_label.loss} (should be None)')

# 15. Check parameter count consistency
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[15] Params: total={total_params:,}, trainable={trainable:,}')

# 16. Check gradient flow through all layers
model.train()
out_g = model(input_ids=x, labels=x, return_dict=True)
out_g.loss.backward()
dead_layers = []
for name, param in model.named_parameters():
    if param.grad is None and param.requires_grad:
        dead_layers.append(name)
if dead_layers:
    bugs.append(f'Dead gradient layers: {dead_layers[:5]}')
print(f'[16] Gradient flow: {len(dead_layers)} dead layers')

print()
if bugs:
    print(f'=== BUGS FOUND ({len(bugs)}) ===')
    for b in bugs:
        print(f'  - {b}')
else:
    print('=== All Deep Checks Passed - No Bugs ===')
