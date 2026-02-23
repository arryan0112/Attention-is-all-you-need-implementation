import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
    
    # 1️⃣ Compute mean across last dimension (features)
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # 2️⃣ Compute variance across last dimension
    var = np.var(x, axis=-1, keepdims=True)
    
    # 3️⃣ Normalize
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    # 4️⃣ Apply learnable scale and shift
    out = gamma * x_hat + beta
    
    return out