import torch.nn as nn
import torch
from spherenet import SphereConv2D, SphereMaxPool2D

class SpherConvLSTM_EncoderCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, stride=2):
        super(SpherConvLSTM_EncoderCell, self).__init__()

        # Encoder
        self.lstm = SpherConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, bias=bias)
        self.pool = SphereMaxPool2D(stride=stride)

    def forward(self, x, state):
        h, c = self.lstm(x, state)
        out = self.pool(h)
        return out, [h, c]

    def init_hidden(self, b, shape):
        h, c = self.lstm.init_hidden(b, shape)
        return [h, c]

class SpherConvLSTM_DecoderCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, scale_factor=2):
        super(SpherConvLSTM_DecoderCell, self).__init__()

        # Decoder
        self.lstm = SpherConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, bias=bias)
        self.up_sampling = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x, state):
        h, c = self.lstm(x, state)
        out = self.up_sampling(h)
        return out, [h, c]

    def init_hidden(self, b, shape):
        h, c = self.lstm.init_hidden(b, shape)
        return [h, c]  

class SpherConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, bias):
        """
        Initialize Spherical ConvLSTM cell.
        ----------
        input_dim: Number of channels of input tensor.
        hidden_dim: Dimension of the hidden states.  
        bias: Whether or not to add the bias.
        """

        super(SpherConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = (3,3) # Spherical convolutions only compatible with 3x3 kernels
        self.bias = bias

        self.conv = SphereConv2D(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
