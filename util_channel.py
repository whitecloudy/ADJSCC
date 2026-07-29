import torch
import torch.nn as nn
import numpy as np


class Channel(nn.Module):
    def __init__(self, channel_type):
        super(Channel, self).__init__()
        self.channel_type = channel_type

    def forward(self, features, snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None):
        shape = features.shape
        f = features.reshape(shape[0], -1)
        dim_z = f.shape[1] // 2

        # Convert to complex channel signal
        z_in = torch.complex(f[:, :dim_z], f[:, dim_z:])

        # Power constraint: average complex symbol power = 1
        norm_factor = torch.sum(z_in.real ** 2 + z_in.imag ** 2, dim=1, keepdim=True)
        z_in_norm = z_in * torch.complex(torch.sqrt(dim_z / norm_factor), torch.zeros_like(norm_factor))

        if self.channel_type == 'awgn':
            if snr_db is None:
                raise ValueError("This input snr should exist!")
            z_out = self.awgn(z_in_norm, snr_db)
        elif self.channel_type == 'slow_fading':
            if snr_db is None or h_real is None or h_imag is None:
                raise ValueError("This input snr, h_real, h_imag should exist!")
            z_out = self.slow_fading(z_in_norm, snr_db, h_real, h_imag)
        elif self.channel_type == 'slow_fading_eq':
            if snr_db is None or h_real is None or h_imag is None:
                raise ValueError("This input snr, h_real, h_imag should exist!")
            z_out = self.slow_fading_eq(z_in_norm, snr_db, h_real, h_imag)
        elif self.channel_type == 'burst':
            if snr_db is None or b_prob is None or b_stddev is None:
                raise ValueError("This input snr, b_prob, b_stddev should exist!")
            z_out = self.burst(z_in_norm, snr_db, b_prob, b_stddev)
        else:
            raise ValueError(f"Unknown channel_type: {self.channel_type}")

        # Convert signal back to intermediate shape
        z_out = torch.cat([z_out.real, z_out.imag], dim=1)
        z_out = z_out.reshape(shape)
        return z_out

    def awgn(self, x, snr_db):
        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)
        return x + noise_stddev * awgn_noise

    def slow_fading(self, x, snr_db, h_real, h_imag):
        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        h = torch.complex(h_real, h_imag).reshape(-1, 1)
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)
        return h * x + noise_stddev * awgn_noise

    def slow_fading_eq(self, x, snr_db, h_real, h_imag):
        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        h = torch.complex(h_real, h_imag).reshape(-1, 1)
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)
        return x + noise_stddev * awgn_noise / h

    def burst(self, x, snr_db, b_prob, b_stddev):
        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)

        b_sample = torch.bernoulli(b_prob.expand_as(x.real)).to(x.device)
        b_noise_real = torch.randn_like(x.real) / np.sqrt(2)
        b_noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        burst_noise = b_sample * b_stddev * torch.complex(b_noise_real, b_noise_imag)
        return x + noise_stddev * awgn_noise + burst_noise
