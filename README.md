# Mamba-3 MIMO (PyTorch Implementation)

A community-led reproduction of the **Mamba-3** architecture as described in *Lahoti et al. (arXiv:2603.15569)*.

### Key Features
*   **Exponential-Trapezoidal Discretization:** Improved numerical stability and expressive dynamics compared to standard Euler discretization.
*   **MIMO (Multi-Input, Multi-Output) Formulation:** Hardware-aware state updates designed for maximum GPU throughput.
*   **Complex-Valued State Tracking:** Enhanced retrieval capabilities for long-context sequences.

### Usage
```python
from mamba3_mimo import Mamba3MIMO
import torch

model = Mamba3MIMO(d_model=128)
x = torch.randn(2, 64, 128)
output = model(x) # (2, 64, 128)
```

*Note: This is a community implementation. Official CUDA kernels are still pending from the authors.*
