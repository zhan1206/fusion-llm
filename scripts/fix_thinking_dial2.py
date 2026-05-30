import codecs

filepath = 'C:/Users/朱子瞻/.qclaw/workspace/fusion-llm/models/thinking_dial.py'
with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find the forward signature with **kwargs and remove it
# The pattern: thinking_depth line, **kwargs line, ) -> Dict line
old_sig = '''    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        thinking_depth: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:'''

new_sig = '''    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        thinking_depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:'''

if old_sig in content:
    content = content.replace(old_sig, new_sig)
    print('Replaced forward signature')
else:
    print('Pattern not found in content')
    # Show what we have around line 538
    lines = content.split('\n')
    for j in range(530, 545):
        print(f'{j+1}: {repr(lines[j])}')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')