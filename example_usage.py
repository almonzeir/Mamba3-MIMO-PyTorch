import torch
from mamba3_mimo import Mamba3MIMO

def run_example():
    print("🚀 Initializing Mamba-3 MIMO Block...")
    
    # Model configuration
    d_model = 128
    d_state = 16
    
    model = Mamba3MIMO(d_model=d_model, d_state=d_state)
    
    # Create dummy long-context input (Batch=1, Length=1024, Dim=128)
    x = torch.randn(1, 1024, d_model)
    
    print(f"📥 Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        
    print(f"📤 Output shape: {output.shape}")
    print("✅ Mamba-3 MIMO Forward Pass Successful!")

if __name__ == "__main__":
    run_example()
