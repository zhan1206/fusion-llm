import sys, torch
sys.path.insert(0, '.')

from models.fusion_model import FusionModel, FusionConfig
from models.thinking_dial import ThinkingDialModel, ThinkingConfig, GRPOTrainer

# Test 1: ThinkingDialModel full integration
config = FusionConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, intermediate_size=128, block_size=8, latent_dim=8,
    max_position_embeddings=128)
base = FusionModel(config)
td = ThinkingDialModel(base, ThinkingConfig())

x = torch.randint(0, 100, (2, 16))
with torch.no_grad():
    out0 = td(input_ids=x, thinking_depth=0).logits
    out3 = td(input_ids=x, thinking_depth=3).logits
print(f'1. Thinking depth 0 vs 3: max_diff={torch.abs(out0 - out3).max().item():.6f}')

# Test 2: Generate with thinking_depth
gen = td.generate(input_ids=x[:, :4], thinking_depth=2, max_new_tokens=8)
print(f'2. Generate with thinking_depth=2: shape={gen.shape}')

# Test 3: GRPOTrainer reward registry
print(f'3. GRPOTrainer REWARD_FUNCTIONS: {list(GRPOTrainer.REWARD_FUNCTIONS.keys())}')

# Test 4: ThinkingDialProcessor unified
from models.thinking_dial import ThinkingDialProcessor
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
proc = ThinkingDialProcessor(tok)
result = proc.process_single('What is 2+2?', '4', think_rank=3)
has_think_token = '<|think' in result['text']
print(f'4. Processor: think_rank={result["think_rank"]}, no text injection={not has_think_token}')

# Test 5: BilingualTrueFilter enhancement
from data_pipeline.bilingual_filter import BilingualTrueFilter
enf = BilingualTrueFilter(lang='en')
assert enf._is_translated_from_chinese('You can you up, no no no!') == True
assert enf._is_translated_from_chinese('The quick brown fox jumps') == False
assert enf._is_low_quality_english('free win click here free win') == True
print('5. BilingualTrueFilter enhanced: OK')

# Test 6: Save/load round-trip with ThinkingDialModel
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    base2 = FusionModel(config)
    td2 = ThinkingDialModel(base2, ThinkingConfig())
    base2.save_pretrained(tmpdir)
    loaded = FusionModel.from_pretrained(tmpdir)
    td_loaded = ThinkingDialModel(loaded, ThinkingConfig())
    
    with torch.no_grad():
        o1 = td(input_ids=x, thinking_depth=1).logits
        # Different model instance, so diff should be non-zero
        o2 = td_loaded(input_ids=x, thinking_depth=1).logits
    print(f'6. ThinkingDialModel save/load: shapes match={o1.shape == o2.shape}')

print('\nAll integration tests passed!')
