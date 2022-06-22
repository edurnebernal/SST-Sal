import Modules
import torch.nn as nn
import torch

class SST_Sal(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=36, output_dim=1):
        super(SST_Sal, self).__init__()

        self.encoder = Modules.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder = Modules.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)


    def forward(self, x):

        b, _, _, h, w = x.size()
        state_e = self.encoder.init_hidden(b, (h, w))
        state_d = self.decoder.init_hidden(b, (h//2, w//2))

        outputs = []

        for t in range(x.shape[1]):
            out, state_e = self.encoder(x[:, t, :, :, :], state_e)
            out, state_d = self.decoder(out, state_d)
            outputs.append(out)
        return torch.stack(outputs, dim=1)
