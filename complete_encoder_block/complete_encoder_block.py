import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# -------------------------------------------------
# 1️⃣ Layer Normalization
# -------------------------------------------------
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    return gamma * x_hat + beta


# -------------------------------------------------
# 2️⃣ Multi-Head Attention
# -------------------------------------------------
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    
    batch_size, seq_len, d_model = Q.shape
    head_dim = d_model // num_heads
    
    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    
    # Split into heads
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, head_dim)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, head_dim)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, head_dim)
    
    # Transpose to shape: (batch, heads, seq, head_dim)
    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)
    
    # Scaled dot-product attention
    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(head_dim)
    
    attn_weights = softmax(scores, axis=-1)
    
    attn_output = attn_weights @ V_heads
    
    # Concatenate heads
    attn_output = attn_output.transpose(0, 2, 1, 3)
    attn_output = attn_output.reshape(batch_size, seq_len, d_model)
    
    # Final linear projection
    output = attn_output @ W_o
    
    return output


# -------------------------------------------------
# 3️⃣ Feed Forward Network
# -------------------------------------------------
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    
    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)  # ReLU
    output = hidden @ W2 + b2
    
    return output


# -------------------------------------------------
# 4️⃣ Encoder Block
# -------------------------------------------------
def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    
    # --- Multi-Head Attention ---
    attn_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    # Residual + LayerNorm
    x = layer_norm(x + attn_output, gamma1, beta1)
    
    # --- Feed Forward ---
    ff_output = feed_forward(x, W1, b1, W2, b2)
    
    # Residual + LayerNorm
    output = layer_norm(x + ff_output, gamma2, beta2)
    
    return output
