# Mamba-3 MIMO in PyTorch

An unofficial, pure-PyTorch implementation of the MIMO (multi-input, multi-output) block from **Mamba-3: Improved Sequence Modeling using State Space Principles** (Lahoti et al., [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)).

It is written to be readable rather than fast: there are no custom CUDA kernels, so it is suited to studying the architecture and running small experiments.

## What is implemented

- Exponential-trapezoidal discretisation of the state space dynamics
- MIMO state updates
- Complex-valued state tracking, simulated with a pair of real values
- Depthwise causal convolution and gated input projection, as in earlier Mamba blocks

## Usage

```bash
pip install -r requirements.txt
python example_usage.py
```

```python
import torch
from mamba3_mimo import Mamba3MIMO

model = Mamba3MIMO(d_model=128, d_state=16)
x = torch.randn(1, 1024, 128)   # (batch, length, d_model)
y = model(x)                    # same shape as x
```

## Notes

This is a community implementation and has not been validated against the authors' reported results. Corrections and comparisons with the official code are welcome.
