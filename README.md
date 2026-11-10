```markdown
# Mini-GPT: Character-Level Language Model on Tiny Shakespeare

This repository contains a custom-built, PyTorch-native Generative Pre-trained Transformer (GPT) trained on the **Tiny Shakespeare** dataset. Built entirely from scratch, it features multi-head causal self-attention, Layer Normalization, residual skip connections, and a customizable token generation head.

## Model Architecture Specifications

- **Parameters**: 19,014,144 (19M)
- **Layers (Blocks)**: 6
- **Embedding Dimension**: 512
- **Attention Heads**: 8 (64-dimensional head size)
- **Context Window (Block Size)**: 128 characters
- **Vocabulary Size**: 65 unique characters
- **Dropout Rate**: 0.2

## Training Metrics & Performance

- **Optimizer**: AdamW (`lr=3e-4`, `weight_decay=0.1`, cosine learning rate decay with linear warmup)
- **Batch Size**: 64
- **Best Validation Loss**: `1.482` achieved at training step `3000`

## How to Load and Generate Text in PyTorch

Ensure you have PyTorch installed, and the model weight checkpoint `mini_gpt.pt` present in your directory.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

# 1. Define Model Config
@dataclass
class Config:
    block_size: int = 128
    vocab_size: int = 65
    n_layers: int = 6
    n_heads: int = 8
    n_embd: int = 512
    dropout: float = 0.2

# 2. Load Checkpoint
checkpoint = torch.load("mini_gpt.pt", map_location="cpu")
chars = checkpoint["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

# 3. Instantiate and Load Weights
# (Make sure the GPT, Block, MLP, and CausalSelfAttention classes are defined in your script)
model = GPT(Config(**checkpoint["cfg"]))
model.load_state_dict(checkpoint["model"])
model.eval()

# 4. Generate Text
def write(prompt="ROMEO:", n_tokens=500, temp=0.8, top_k=40):
    clean_prompt = "".join(c for c in prompt if c in stoi) or chars[0]
    idx = torch.tensor([[stoi[c] for c in clean_prompt]], dtype=torch.long)
    out = model.generate(idx, n_tokens, temp, top_k)[0].tolist()
    return "".join(itos[i] for i in out)

print(write("ROMEO:"))
```

## Generation Examples (Varying Temperatures)

### Low Temperature (0.3) — Safe and Repetitive
```
ROMEO:
What is the matter?

MERCUTIO:
No, I have said thee before the heavens of the heaven
That thou hast said 'twas to be the good time.
```

### Medium Temperature (0.8) — Well-Balanced
```
ROMEO:
I will I speak to my captain with support,
And tell me not, for me revenge men, I give my life.
```

### High Temperature (1.4) — Creative / Experimental
```
ROMEO:
We beSicide the wise wherein Henry's right,
Let us have had taken the prick wrongs;
```

