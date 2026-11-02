import torch
import torch.nn as nn
import numpy as np


class Channel(nn.Module):
    def __init__(self, channel_type):
        super(Channel, self).__init__()
        self.channel_type = str(channel_type).strip().lower() if channel_type is not None else 'awgn'

    def forward(self, features, snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None, equalize=True):
        shape = features.shape
        f = features.reshape(shape[0], -1)
        dim_z = f.shape[1] // 2

        # Convert to complex channel signal
        z_in = torch.complex(f[:, :dim_z], f[:, dim_z:])

        # Power constraint: average complex symbol power = 1
        norm_factor = torch.sum(z_in.real ** 2 + z_in.imag ** 2, dim=1, keepdim=True)
        z_in_norm = z_in * torch.complex(torch.sqrt(dim_z / torch.clamp(norm_factor, min=1e-8)), torch.zeros_like(norm_factor))

        ch_type = self.channel_type
        if ch_type == 'awgn':
            if snr_db is None:
                raise ValueError("This input snr should exist!")
            z_out = self.awgn(z_in_norm, snr_db)
        elif ch_type == 'slow_fading':
            if snr_db is None or h_real is None or h_imag is None:
                raise ValueError("This input snr, h_real, h_imag should exist!")
            z_out = self.slow_fading(z_in_norm, snr_db, h_real, h_imag)
        elif ch_type == 'slow_fading_eq':
            if snr_db is None or h_real is None or h_imag is None:
                raise ValueError("This input snr, h_real, h_imag should exist!")
            z_out = self.slow_fading_eq(z_in_norm, snr_db, h_real, h_imag)
        elif ch_type in ['rayleigh_fading', 'rayleigh']:
            if snr_db is None:
                raise ValueError("This input snr should exist!")
            z_out = self.rayleigh_fading(z_in_norm, snr_db, h_real=h_real, h_imag=h_imag, equalize=equalize)
        elif ch_type in ['rayleigh_fading_eq', 'rayleigh_eq']:
            if snr_db is None:
                raise ValueError("This input snr should exist!")
            z_out = self.rayleigh_fading_eq(z_in_norm, snr_db, h_real=h_real, h_imag=h_imag)
        elif ch_type == 'burst':
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
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(snr_db, device=x.device, dtype=x.real.dtype)
        if snr_db.ndim == 1 and snr_db.shape[0] == x.shape[0]:
            snr_db = snr_db.reshape(-1, 1)

        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)
        return x + noise_stddev * awgn_noise

    def slow_fading(self, x, snr_db, h_real, h_imag):
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(snr_db, device=x.device, dtype=x.real.dtype)
        if snr_db.ndim == 1 and snr_db.shape[0] == x.shape[0]:
            snr_db = snr_db.reshape(-1, 1)

        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        h = torch.complex(h_real, h_imag).reshape(-1, 1)
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)
        return h * x + noise_stddev * awgn_noise

    def slow_fading_eq(self, x, snr_db, h_real, h_imag):
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(snr_db, device=x.device, dtype=x.real.dtype)
        if snr_db.ndim == 1 and snr_db.shape[0] == x.shape[0]:
            snr_db = snr_db.reshape(-1, 1)

        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        h = torch.complex(h_real, h_imag).reshape(-1, 1)
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)
        eps_eq = 1e-4
        h_mag = torch.abs(h)
        h_safe = torch.where(h_mag < eps_eq, eps_eq * torch.exp(1j * torch.angle(h)), h)
        return x + noise_stddev * awgn_noise / h_safe

    def rayleigh_fading(self, x, snr_db, h_real=None, h_imag=None, equalize=True):
        """
        Rayleigh Fading Channel (Y = H * Z + N, with H ~ CN(0, 1) i.i.d.)
        Matches definition in train_SC.py / test_SC.py.
        If equalize=True (default), applies MMSE equalization as in test_SC.py (lines 746-753):
            Z_hat = (Y * conj(H)) / (|H|^2 + noise_var)
        If equalize=False, returns raw received signal Y.
        """
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(snr_db, device=x.device, dtype=x.real.dtype)
        if snr_db.ndim == 1 and snr_db.shape[0] == x.shape[0]:
            snr_db = snr_db.reshape(-1, 1)

        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))

        if h_real is not None and h_imag is not None:
            if not isinstance(h_real, torch.Tensor):
                h_real = torch.tensor(h_real, device=x.device, dtype=x.real.dtype)
            if not isinstance(h_imag, torch.Tensor):
                h_imag = torch.tensor(h_imag, device=x.device, dtype=x.imag.dtype)
            h = torch.complex(h_real, h_imag)
            if h.ndim == 1:
                h = h.reshape(-1, 1)
            elif h.ndim > 2:
                h = h.reshape(x.shape[0], -1)
        else:
            # H ~ CN(0, 1) i.i.d. per complex symbol: E[|H|^2] = 1
            hr = torch.randn_like(x.real) / np.sqrt(2.0)
            hi = torch.randn_like(x.imag) / np.sqrt(2.0)
            h = torch.complex(hr, hi)

        # Complex Gaussian noise per dimension sigma / sqrt(2)
        noise_real = torch.randn_like(x.real) / np.sqrt(2.0)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2.0)
        awgn_noise = torch.complex(noise_real, noise_imag)
        noise = noise_stddev * awgn_noise
        rx_signal = h * x + noise

        if equalize:
            # MMSE equalization matching test_SC.py:
            h_mag2 = torch.abs(h) ** 2
            noise_var = 10.0 ** (-snr_db / 10.0)
            return (rx_signal * torch.conj(h)) / torch.clamp(h_mag2 + noise_var, min=1e-8)
        return rx_signal

    def rayleigh_fading_eq(self, x, snr_db, h_real=None, h_imag=None, eps_eq=1e-4):
        """
        Equalized Rayleigh Fading Channel (Y/H = Z + N/H, with H ~ CN(0, 1) i.i.d.)
        Matches definition in train_SC.py / test_SC.py (Zero-Forcing equalization with eps_eq safety).
            Y_eq = Z + N / H_safe
        """
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(snr_db, device=x.device, dtype=x.real.dtype)
        if snr_db.ndim == 1 and snr_db.shape[0] == x.shape[0]:
            snr_db = snr_db.reshape(-1, 1)

        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))

        if h_real is not None and h_imag is not None:
            if not isinstance(h_real, torch.Tensor):
                h_real = torch.tensor(h_real, device=x.device, dtype=x.real.dtype)
            if not isinstance(h_imag, torch.Tensor):
                h_imag = torch.tensor(h_imag, device=x.device, dtype=x.imag.dtype)
            h = torch.complex(h_real, h_imag)
            if h.ndim == 1:
                h = h.reshape(-1, 1)
            elif h.ndim > 2:
                h = h.reshape(x.shape[0], -1)
        else:
            # H ~ CN(0, 1) i.i.d. per complex symbol: E[|H|^2] = 1
            hr = torch.randn_like(x.real) / np.sqrt(2.0)
            hi = torch.randn_like(x.imag) / np.sqrt(2.0)
            h = torch.complex(hr, hi)

        noise_real = torch.randn_like(x.real) / np.sqrt(2.0)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2.0)
        awgn_noise = torch.complex(noise_real, noise_imag)
        noise = noise_stddev * awgn_noise

        # ZF equalization with eps_eq protection as in test_SC.py / semantic_comm_dataset.py
        h_mag = torch.abs(h)
        h_safe = torch.where(h_mag < eps_eq, eps_eq * torch.exp(1j * torch.angle(h)), h)
        return x + noise / h_safe

    def burst(self, x, snr_db, b_prob, b_stddev):
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(snr_db, device=x.device, dtype=x.real.dtype)
        if snr_db.ndim == 1 and snr_db.shape[0] == x.shape[0]:
            snr_db = snr_db.reshape(-1, 1)

        noise_stddev = torch.sqrt(10.0 ** (-snr_db / 10.0))
        noise_real = torch.randn_like(x.real) / np.sqrt(2)
        noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        awgn_noise = torch.complex(noise_real, noise_imag)

        b_sample = torch.bernoulli(b_prob.expand_as(x.real)).to(x.device)
        b_noise_real = torch.randn_like(x.real) / np.sqrt(2)
        b_noise_imag = torch.randn_like(x.imag) / np.sqrt(2)
        burst_noise = b_sample * b_stddev * torch.complex(b_noise_real, b_noise_imag)
        return x + noise_stddev * awgn_noise + burst_noise
