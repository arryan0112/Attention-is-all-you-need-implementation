import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Q, K, V: (batch, seq_len, d_model)
    W_q, W_k, W_v: (d_model, d_model)
    W_o: (d_model, d_model)
    
    Returns:
        output: (batch, seq_len, d_model)
    """

    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # 1️⃣ Linear projections
    Q_proj = Q @ W_q   # (batch, seq_len, d_model)
    K_proj = K @ W_k
    V_proj = V @ W_v

    # 2️⃣ Split into heads
    Q_proj = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_proj = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_proj = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    # 3️⃣ Transpose to (batch, num_heads, seq_len, d_k)
    Q_proj = Q_proj.transpose(0, 2, 1, 3)
    K_proj = K_proj.transpose(0, 2, 1, 3)
    V_proj = V_proj.transpose(0, 2, 1, 3)

    # 4️⃣ Scaled dot-product attention per head
    scores = Q_proj @ K_proj.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scores, axis=-1)

    head_output = attention_weights @ V_proj
    # (batch, num_heads, seq_len, d_k)

    # 5️⃣ Concatenate heads
    head_output = head_output.transpose(0, 2, 1, 3)
    # (batch, seq_len, num_heads, d_k)

    concat_output = head_output.reshape(batch_size, seq_len, d_model)
    # (batch, seq_len, d_model)

    # 6️⃣ Final output projection
    output = concat_output @ W_o
    # (batch, seq_len, d_model)

    return output
