import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, temperature=1.0):
        """
        :param temperature: Factor by which the logits are divided.
                    Small numbers make the model more confident on each position, but also more conservative.
                    Large values result in more random predictions at each step.
        """
        logits = self.proj(x) / temperature
        return F.log_softmax(logits, dim=-1)
