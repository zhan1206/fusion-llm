import codecs

filepath = 'C:/Users/朱子瞻/.qclaw/workspace/fusion-llm/models/thinking_dial.py'
with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

lines = content.split('\n')
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'thinking_depth: Optional[torch.Tensor] = None,' in line and i+1 < len(lines) and '**kwargs' in lines[i+1]:
        # Keep the first two lines of the signature
        new_lines.append(line)  # thinking_depth line
        i += 1
        new_lines.append(lines[i])  # **kwargs line
        i += 1
        new_lines.append(lines[i])  # ) -> Dict line
        i += 1
        # Skip docstring
        while i < len(lines) and '"""' not in lines[i]:
            i += 1
        if i < len(lines):
            i += 1  # skip opening """
        while i < len(lines) and '"""' not in lines[i]:
            i += 1
        if i < len(lines):
            i += 1  # skip closing """
        # Skip old body until return
        while i < len(lines) and 'return base_outputs' not in lines[i]:
            i += 1
        if i < len(lines):
            i += 1  # skip return base_outputs
        # Skip pass and remaining code
        while i < len(lines) and (lines[i].strip().startswith('pass') or lines[i].strip().startswith('#') or lines[i].strip() == ''):
            i += 1
        # Add new docstring
        new_lines.append('        """')
        new_lines.append('        前向传播')
        new_lines.append('')
        new_lines.append('        参数：')
        new_lines.append('            input_ids: (batch, seq_len)')
        new_lines.append('            attention_mask: (batch, seq_len)')
        new_lines.append('            labels: (batch, seq_len)')
        new_lines.append('            thinking_depth: (batch,) 推理深度（0-3）')
        new_lines.append('')
        new_lines.append('        返回：')
        new_lines.append('            包含 loss, logits 的字典')
        new_lines.append('        """')
        new_lines.append('        # 基础模型前向传播（移除 **kwargs 透传，避免 HF 不兼容）')
        new_lines.append('        base_outputs = self.base_model(')
        new_lines.append('            input_ids=input_ids,')
        new_lines.append('            attention_mask=attention_mask,')
        new_lines.append('            labels=labels,')
        new_lines.append('        )')
        new_lines.append('        return base_outputs')
    else:
        new_lines.append(line)
        i += 1

result = '\n'.join(new_lines)
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(result)
print('Fixed thinking_dial.py')