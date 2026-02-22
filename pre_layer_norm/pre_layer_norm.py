import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, sublayer):
        """
        x: (batch, seq_len, d_model)
        sublayer: function that takes normalized x and returns output of same shape
        
        Returns:
            (batch, seq_len, d_model)
        """

        # 1️⃣ Normalize input
        normalized_x = self.norm(x)

        # 2️⃣ Pass through sublayer (e.g., attention or FFN)
        sublayer_output = sublayer(normalized_x)

        # 3️⃣ Add residual connection
        output = x + sublayer_output

        return output
