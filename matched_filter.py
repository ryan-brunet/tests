"""
matched_filter.py
-----------------
Injects a known template signal into white and red noise, then recovers
it with a matched filter applied in the frequency domain.

Theory
------
For a signal s(t) buried in additive noise with one-sided PSD Sn(f),
the optimal (matched) filter is:

    H(f) = S*(f) / Sn(f)

where S(f) = FFT(s).  The filter output

    y(t) = IFFT( X(f) · H(f) )

peaks at the arrival time of the signal.  Dividing by the expected
noiseless peak normalises y to units of SNR (sigma).

For white noise Sn(f) = const, so H(f) ∝ S*(f) — equivalent to a
plain cross-correlation with the template.

For red noise Sn(f) ∝ 1/f^alpha the filter up-weights high-frequency
content, effectively "whitening" the data before correlation.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
N          = 4096     # number of samples
dt         = 1.0      # sampling interval (arbitrary units)
ALPHA      = 2.0      # red-noise spectral exponent  (P ∝ 1/f^alpha)
SIGMA      = 20.0     # Gaussian envelope half-width of template (samples)
F0         = 0.05     # carrier frequency of template (cycles / sample)
INJECT_IDX = N // 3   # sample index of injected signal (the "unknown")
AMPLITUDE  = 5.0      # signal amplitude in units of noise std
SEED       = 7

rng = np.random.default_rng(SEED)
t   = np.arange(N) * dt

# ── 1. Template signal ────────────────────────────────────────────────────────
# Gaussian-windowed sinusoid centred at index 0 (will be shifted to INJECT_IDX)
t_tmpl   = np.arange(N) - N // 2    # centred index array
template = (np.exp(-0.5 * (t_tmpl / SIGMA) ** 2)
            * np.cos(2 * np.pi * F0 * t_tmpl))
template /= np.linalg.norm(template)  # unit-norm so SNR has clean σ units

# Shift to the injection location for adding to data
signal_at_injection = np.roll(template, INJECT_IDX)

# ── 2. Generate noise ─────────────────────────────────────────────────────────

def make_white(n, rng):
    x = rng.standard_normal(n)
    return (x - x.mean()) / x.std()

def make_red(n, alpha, rng):
    """Spectrally shaped Gaussian noise with P(f) ∝ 1/f^alpha."""
    wf      = np.fft.rfft(rng.standard_normal(n))
    freqs   = np.fft.rfftfreq(n)
    shaping = np.ones_like(freqs)
    shaping[1:] = freqs[1:] ** (-alpha / 2.0)   # amplitude filter → power ∝ 1/f^α
    x = np.fft.irfft(wf * shaping, n=n)
    return (x - x.mean()) / x.std()

white = make_white(N, rng)
red   = make_red(N, ALPHA, rng)

# ── 3. Inject signal ──────────────────────────────────────────────────────────
white_data = white + AMPLITUDE * signal_at_injection
red_data   = red   + AMPLITUDE * signal_at_injection

# ── 4. Noise PSD models ───────────────────────────────────────────────────────
freqs = np.fft.rfftfreq(N, d=dt)

# White: flat
psd_white = np.ones(len(freqs))

# Red: 1/f^alpha (DC bin set to neighbouring value to avoid singularity)
psd_red       = np.ones(len(freqs))
psd_red[1:]   = freqs[1:] ** (-ALPHA)
psd_red[0]    = psd_red[1]

# Normalise so the mean PSD = 1 (keeps SNR numbers interpretable)
psd_white /= psd_white.mean()
psd_red   /= psd_red.mean()

# ── 5. Matched filter ─────────────────────────────────────────────────────────

def matched_filter(data, template, psd):
    """
    Apply frequency-domain matched filter and return SNR time series.

    Parameters
    ----------
    data     : observed (noisy) time series
    template : signal template (same length, unit norm)
    psd      : noise PSD at each rfft frequency bin

    Returns
    -------
    snr : array of shape (N,), in units of sigma

    Normalisation
    -------------
    Let  rho = irfft(|S|^2 / Sn)[0]  (the "inner product" of the template
    with itself through the whitened filter, equal to ||s||^2 = 1 for white
    noise with unit-norm template).

    For a signal injected with amplitude A the noiseless peak is A * rho,
    and the noise std at the filter output is sqrt(rho), giving:

        SNR_peak = A * rho / sqrt(rho) = A * sqrt(rho)

    Dividing y by sqrt(rho) therefore puts the output in sigma units.
    """
    N  = len(data)
    X  = np.fft.rfft(data)
    S  = np.fft.rfft(template)
    H  = np.conj(S) / psd                               # matched filter kernel
    y  = np.fft.irfft(X * H, n=N)
    # rho: noiseless peak value when amplitude = 1; also = noise variance at output
    rho  = np.real(np.fft.irfft(np.abs(S) ** 2 / psd, n=N)[0])
    norm = np.sqrt(rho)                                  # noise std at output
    return y / norm                                      # SNR in σ units

mf_white = matched_filter(white_data, template, psd_white)
mf_red   = matched_filter(red_data,   template, psd_red)

# ── 6. Detection peaks ────────────────────────────────────────────────────────
peak_white = int(np.argmax(np.abs(mf_white)))
peak_red   = int(np.argmax(np.abs(mf_red)))

print(f"Signal injected at sample : {INJECT_IDX}")
print(f"White-noise MF peak       : sample {peak_white}  "
      f"(SNR = {mf_white[peak_white]:.1f} σ)")
print(f"Red-noise   MF peak       : sample {peak_red}  "
      f"(SNR = {mf_red[peak_red]:.1f} σ)")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
fig.suptitle(
    f"Matched filter demo  —  signal injected at sample {INJECT_IDX}, "
    f"amplitude = {AMPLITUDE} σ",
    fontsize=13,
)

true_kw = dict(color="limegreen", lw=1.6, ls="--",
               label=f"True injection (sample {INJECT_IDX})")
peak_kw = dict(color="orange", lw=1.6, ls=":",
               label="MF peak (detected)")

# Row 0: noisy data
axes[0, 0].plot(t, white_data, lw=0.5, color="steelblue")
axes[0, 0].axvline(INJECT_IDX, **true_kw)
axes[0, 0].set_title("White noise + injected signal")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(t, red_data, lw=0.5, color="firebrick")
axes[0, 1].axvline(INJECT_IDX, **true_kw)
axes[0, 1].set_title(f"Red noise (α = {ALPHA}) + injected signal")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].legend(fontsize=8)

# Row 1: template shown for reference (same in both columns)
for ax in axes[1]:
    ax.plot(t, AMPLITUDE * signal_at_injection, lw=1.2, color="purple")
    ax.axvline(INJECT_IDX, **true_kw)
    ax.set_title("Injected template (reference)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)

# Row 2: matched filter SNR output
axes[2, 0].plot(t, mf_white, lw=0.7, color="steelblue")
axes[2, 0].axvline(INJECT_IDX, **true_kw)
axes[2, 0].axvline(peak_white, color="orange", lw=1.6, ls=":",
                   label=f"MF peak (sample {peak_white})")
axes[2, 0].set_title("Matched filter output — white noise")
axes[2, 0].set_ylabel("SNR (σ)")
axes[2, 0].set_xlabel("Sample")
axes[2, 0].legend(fontsize=8)

axes[2, 1].plot(t, mf_red, lw=0.7, color="firebrick")
axes[2, 1].axvline(INJECT_IDX, **true_kw)
axes[2, 1].axvline(peak_red, color="orange", lw=1.6, ls=":",
                   label=f"MF peak (sample {peak_red})")
axes[2, 1].set_title("Matched filter output — red noise")
axes[2, 1].set_ylabel("SNR (σ)")
axes[2, 1].set_xlabel("Sample")
axes[2, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("matched_filter.png", dpi=150, bbox_inches="tight")
print("Plot saved to matched_filter.png")
plt.show()
