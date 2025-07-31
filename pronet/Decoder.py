import torch.nn as nn
import torch.nn.functional as F
#3
class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=20):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id=None):
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs