import torch
import torch.nn as nn

class AddNormPostLN(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, sublayer):
        """
        x: (batch, seq_len, d_model)
        sublayer: function that takes x and returns same shape

        Returns:
            (batch, seq_len, d_model)
        """

        # 1️⃣ Apply sublayer directly (no normalization before)
        sublayer_output = sublayer(x)

        # 2️⃣ Add residual
        residual_added = x + sublayer_output

        # 3️⃣ Apply LayerNorm after addition
        output = self.norm(residual_added)

        return output
