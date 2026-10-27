import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GDN(nn.Module):
    def __init__(self, in_channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.in_channels = in_channels

        self.beta_bound = np.sqrt(beta_min)
        self.gamma_bound = 0.0

        self.beta = nn.Parameter(torch.sqrt(torch.ones(in_channels) + self.beta_bound ** 2))
        self.gamma = nn.Parameter(torch.sqrt(torch.eye(in_channels) * gamma_init))

    def forward(self, x):
        beta = torch.clamp(self.beta, min=self.beta_bound) ** 2
        gamma = torch.clamp(self.gamma, min=self.gamma_bound) ** 2
        gamma = gamma.view(self.in_channels, self.in_channels, 1, 1)

        norm = F.conv2d(x ** 2, gamma, bias=beta)
        norm = torch.clamp(norm, min=1e-6)

        if self.inverse:
            out = x * torch.sqrt(norm)
        else:
            out = x / torch.sqrt(norm)
        return out


class GFR_Encoder_Module(nn.Module):
    def __init__(self, in_channels, num_filter, kernel_size, stride, activation=None):
        super(GFR_Encoder_Module, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, num_filter, kernel_size, stride=stride, padding=padding)
        self.gdn = GDN(num_filter, inverse=False)
        self.activation_type = activation
        if activation == 'prelu':
            self.prelu = nn.PReLU(num_parameters=num_filter)

    def forward(self, x):
        x = self.conv(x)
        x = self.gdn(x)
        if self.activation_type == 'prelu':
            x = self.prelu(x)
        return x


class GFR_Decoder_Module(nn.Module):
    def __init__(self, in_channels, num_filter, kernel_size, stride, activation=None):
        super(GFR_Decoder_Module, self).__init__()
        self.activation_type = activation
        padding = kernel_size // 2
        if stride > 1:
            out_padding = stride - 1
            self.conv = nn.ConvTranspose2d(in_channels, num_filter, kernel_size, stride=stride,
                                           padding=padding, output_padding=out_padding)
        else:
            self.conv = nn.Conv2d(in_channels, num_filter, kernel_size, stride=1, padding=padding)
        self.gdn = GDN(num_filter, inverse=True)
        if activation == 'prelu':
            self.prelu = nn.PReLU(num_parameters=num_filter)
        elif activation == 'sigmoid':
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.gdn(x)
        if self.activation_type == 'prelu':
            x = self.prelu(x)
        elif self.activation_type == 'sigmoid':
            x = self.sigmoid(x)
        return x


class AF_Module(nn.Module):
    def __init__(self, ch_num):
        super(AF_Module, self).__init__()
        self.fc1 = nn.Linear(ch_num + 1, ch_num // 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(ch_num // 16, ch_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, snr):
        if snr.dtype != inputs.dtype:
            snr = snr.to(inputs.dtype)
        gap = inputs.mean(dim=(2, 3))
        m = torch.cat([gap, snr], dim=1)
        m = self.relu(self.fc1(m))
        m = self.sigmoid(self.fc2(m))
        m = m.unsqueeze(2).unsqueeze(3)
        return inputs * m


class AF_Module_H(nn.Module):
    def __init__(self, ch_num):
        super(AF_Module_H, self).__init__()
        self.fc1 = nn.Linear(ch_num + 3, ch_num // 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(ch_num // 16, ch_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, snr, h_real, h_imag):
        if snr.dtype != inputs.dtype:
            snr = snr.to(inputs.dtype)
        if h_real.dtype != inputs.dtype:
            h_real = h_real.to(inputs.dtype)
        if h_imag.dtype != inputs.dtype:
            h_imag = h_imag.to(inputs.dtype)
        gap = inputs.mean(dim=(2, 3))
        m = torch.cat([gap, snr, h_real, h_imag], dim=1)
        m = self.relu(self.fc1(m))
        m = self.sigmoid(self.fc2(m))
        m = m.unsqueeze(2).unsqueeze(3)
        return inputs * m


class Basic_Encoder(nn.Module):
    def __init__(self, tcn, in_channels=3):
        super(Basic_Encoder, self).__init__()
        self.en1 = GFR_Encoder_Module(in_channels, 256, 9, 2, 'prelu')
        self.en2 = GFR_Encoder_Module(256, 256, 5, 2, 'prelu')
        self.en3 = GFR_Encoder_Module(256, 256, 5, 1, 'prelu')
        self.en4 = GFR_Encoder_Module(256, 256, 5, 1, 'prelu')
        self.en5 = GFR_Encoder_Module(256, tcn, 5, 1)

    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        x = self.en4(x)
        x = self.en5(x)
        return x


class Basic_Decoder(nn.Module):
    def __init__(self, tcn, out_channels=3):
        super(Basic_Decoder, self).__init__()
        self.de1 = GFR_Decoder_Module(tcn, 256, 5, 1, 'prelu')
        self.de2 = GFR_Decoder_Module(256, 256, 5, 1, 'prelu')
        self.de3 = GFR_Decoder_Module(256, 256, 5, 1, 'prelu')
        self.de4 = GFR_Decoder_Module(256, 256, 5, 2, 'prelu')
        self.de5 = GFR_Decoder_Module(256, out_channels, 9, 2, 'sigmoid')

    def forward(self, x):
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = self.de4(x)
        x = self.de5(x)
        return x


class Attention_Encoder(nn.Module):
    def __init__(self, tcn, in_channels=3):
        super(Attention_Encoder, self).__init__()
        self.en1 = GFR_Encoder_Module(in_channels, 256, 9, 2, 'prelu')
        self.af1 = AF_Module(256)
        self.en2 = GFR_Encoder_Module(256, 256, 5, 2, 'prelu')
        self.af2 = AF_Module(256)
        self.en3 = GFR_Encoder_Module(256, 256, 5, 1, 'prelu')
        self.af3 = AF_Module(256)
        self.en4 = GFR_Encoder_Module(256, 256, 5, 1, 'prelu')
        self.af4 = AF_Module(256)
        self.en5 = GFR_Encoder_Module(256, tcn, 5, 1)

    def forward(self, x, snr):
        x = self.af1(self.en1(x), snr)
        x = self.af2(self.en2(x), snr)
        x = self.af3(self.en3(x), snr)
        x = self.af4(self.en4(x), snr)
        x = self.en5(x)
        return x


class Attention_Decoder(nn.Module):
    def __init__(self, tcn, out_channels=3):
        super(Attention_Decoder, self).__init__()
        self.de1 = GFR_Decoder_Module(tcn, 256, 5, 1, 'prelu')
        self.af1 = AF_Module(256)
        self.de2 = GFR_Decoder_Module(256, 256, 5, 1, 'prelu')
        self.af2 = AF_Module(256)
        self.de3 = GFR_Decoder_Module(256, 256, 5, 1, 'prelu')
        self.af3 = AF_Module(256)
        self.de4 = GFR_Decoder_Module(256, 256, 5, 2, 'prelu')
        self.af4 = AF_Module(256)
        self.de5 = GFR_Decoder_Module(256, out_channels, 9, 2, 'sigmoid')

    def forward(self, x, snr):
        x = self.af1(self.de1(x), snr)
        x = self.af2(self.de2(x), snr)
        x = self.af3(self.de3(x), snr)
        x = self.af4(self.de4(x), snr)
        x = self.de5(x)
        return x


class Attention_Encoder_H(nn.Module):
    def __init__(self, tcn, in_channels=3):
        super(Attention_Encoder_H, self).__init__()
        self.en1 = GFR_Encoder_Module(in_channels, 256, 9, 2, 'prelu')
        self.af1 = AF_Module_H(256)
        self.en2 = GFR_Encoder_Module(256, 256, 5, 2, 'prelu')
        self.af2 = AF_Module_H(256)
        self.en3 = GFR_Encoder_Module(256, 256, 5, 1, 'prelu')
        self.af3 = AF_Module_H(256)
        self.en4 = GFR_Encoder_Module(256, 256, 5, 1, 'prelu')
        self.af4 = AF_Module_H(256)
        self.en5 = GFR_Encoder_Module(256, tcn, 5, 1)

    def forward(self, x, snr, h_real, h_imag):
        x = self.af1(self.en1(x), snr, h_real, h_imag)
        x = self.af2(self.en2(x), snr, h_real, h_imag)
        x = self.af3(self.en3(x), snr, h_real, h_imag)
        x = self.af4(self.en4(x), snr, h_real, h_imag)
        x = self.en5(x)
        return x


class Attention_Decoder_H(nn.Module):
    def __init__(self, tcn, out_channels=3):
        super(Attention_Decoder_H, self).__init__()
        self.de1 = GFR_Decoder_Module(tcn, 256, 5, 1, 'prelu')
        self.af1 = AF_Module_H(256)
        self.de2 = GFR_Decoder_Module(256, 256, 5, 1, 'prelu')
        self.af2 = AF_Module_H(256)
        self.de3 = GFR_Decoder_Module(256, 256, 5, 1, 'prelu')
        self.af3 = AF_Module_H(256)
        self.de4 = GFR_Decoder_Module(256, 256, 5, 2, 'prelu')
        self.af4 = AF_Module_H(256)
        self.de5 = GFR_Decoder_Module(256, out_channels, 9, 2, 'sigmoid')

    def forward(self, x, snr, h_real, h_imag):
        x = self.af1(self.de1(x), snr, h_real, h_imag)
        x = self.af2(self.de2(x), snr, h_real, h_imag)
        x = self.af3(self.de3(x), snr, h_real, h_imag)
        x = self.af4(self.de4(x), snr, h_real, h_imag)
        x = self.de5(x)
        return x