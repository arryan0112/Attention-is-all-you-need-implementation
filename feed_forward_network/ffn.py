import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.

    x  : (batch_size, seq_len, d_model)
    W1 : (d_model, d_ff)
    b1 : (d_ff,)
    W2 : (d_ff, d_model)
    b2 : (d_model,)

    Returns:
        (batch_size, seq_len, d_model)
    """

    # 1️⃣ First linear layer (expand)
    hidden = x @ W1 + b1
    # Shape: (batch_size, seq_len, d_ff)

    # 2️⃣ ReLU activation
    hidden = np.maximum(0, hidden)
    # Shape: (batch_size, seq_len, d_ff)

    # 3️⃣ Second linear layer (compress back)
    output = hidden @ W2 + b2
    # Shape: (batch_size, seq_len, d_model)

    return output