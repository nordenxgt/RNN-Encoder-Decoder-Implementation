import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int, hidden_dim: int, dropout: int = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim+hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(embedding_dim+hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc(output)
        return prediction, hidden