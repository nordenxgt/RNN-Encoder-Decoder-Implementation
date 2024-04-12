import random

import torch
from torch import nn

class RNNEncoderDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.cuda.device) -> None:
        super().__init__()
        assert encoder.hidden_dim == decoder.hidden_dim
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        self._init_weights()

    def _init_weights(self):
        for _, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.01)

    def forward(self, src: torch.Tensor, target: torch.Tensor, tf_ratio: float) -> torch.Tensor:
        target_length = target.shape[0]

        outputs = torch.zeros(target_length, target.shape[1], self.decoder.output_dim).to(self.device)
        
        hidden = self.encoder(src)
        context = hidden
        input = target[0, :]

        for t in range(1, target_length):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            input = target[t] if random.random() < tf_ratio else output.argmax(1)

        return outputs