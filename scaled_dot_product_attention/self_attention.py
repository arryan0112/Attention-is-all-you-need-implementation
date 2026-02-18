import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor, 
                                 V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Q: [batch, seq_len_q, d_k]
    K: [batch, seq_len_k, d_k]
    V: [batch, seq_len_k, d_v]

    Returns:
        output: [batch, seq_len_q, d_v]
    """

    # 1. Compute raw attention scores
    # (batch, seq_len_q, d_k) @ (batch, d_k, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # Result shape: (batch, seq_len_q, seq_len_k)

    # 2. Scale scores by sqrt(d_k)
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)

    # 3. Apply softmax over key dimension
    attention_weights = F.softmax(scores, dim=-1)
    # Shape: (batch, seq_len_q, seq_len_k)

    # 4. Multiply attention weights with values
    # (batch, seq_len_q, seq_len_k) @ (batch, seq_len_k, d_v)
    output = torch.matmul(attention_weights, V)
    # Shape: (batch, seq_len_q, d_v)

    return output
