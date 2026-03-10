class SelfAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = Q @ K.transpose(-2,-1)
        scores = scores / math.sqrt(x.size(-1))

        seq_len = x.size(1)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)

        output = weights @ V

        return output