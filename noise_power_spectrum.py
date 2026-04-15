"""
noise_power_spectrum.py
-----------------------
Generates white and red noise time series, computes their power spectra
(via FFT), and plots both the time series and spectra side by side.

Red noise is produced by shaping white noise in the frequency domain:
    P(f) ∝ 1 / f^alpha
where alpha = 0 gives white noise and alpha > 0 gives progressively
"redder" (steeper) spectra.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Simulation parameters ────────────────────────────────────────────────────
N = 4096          # number of time samples (power of 2 for FFT efficiency)
dt = 1.0          # sampling interval (arbitrary units → frequency in cycles/unit)
ALPHA = 2.0       # red-noise spectral exponent  (P ∝ 1/f^alpha)
SEED = 42         # random seed for reproducibility

rng = np.random.default_rng(SEED)

# ── Time axis ────────────────────────────────────────────────────────────────
t = np.arange(N) * dt                       # time vector

# ── 1. White noise ───────────────────────────────────────────────────────────
white = rng.standard_normal(N)              # i.i.d. Gaussian samples

# ── 2. Red noise via spectral shaping ────────────────────────────────────────
# Step 1 – transform white noise to the frequency domain
white_fft = np.fft.rfft(rng.standard_normal(N)) # Compute the one-dimensional discrete Fourier Transform for real input.

# Step 2 – build the 1/f^alpha shaping filter
#   rfft returns N//2+1 non-negative frequencies; skip f=0 (DC) to avoid /0
freqs = np.fft.rfftfreq(N, d=dt)           # frequencies in cycles per unit time
shaping = np.ones_like(freqs)
shaping[1:] = freqs[1:] ** (-ALPHA / 2.0) # amplitude ∝ 1/f^(alpha/2)
                                            # → power ∝ 1/f^alpha

# Step 3 – colour the spectrum and transform back to the time domain
red_fft = white_fft * shaping
red = np.fft.irfft(red_fft, n=N)

# Normalise to zero-mean, unit-variance for fair visual comparison
red = (red - red.mean()) / red.std()
white = (white - white.mean()) / white.std()

# ── 3. Power spectral density (one-sided, via FFT) ───────────────────────────
def power_spectrum(x, dt=1.0):
    """Return (frequencies, PSD) for a real-valued series x.

    Uses a rectangular window (no tapering) and normalises so that the
    integral of the PSD over positive frequencies equals the signal variance.
    """
    n = len(x)
    fft_vals = np.fft.rfft(x)
    psd = (np.abs(fft_vals) ** 2) / n      # raw periodogram
    psd[1:-1] *= 2                          # fold negative freqs onto positive
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, psd

freqs_w, psd_w = power_spectrum(white, dt)
freqs_r, psd_r = power_spectrum(red, dt)

# Reference slope: P ∝ 1/f^alpha (skip DC for log plot)
ref_freqs = freqs_r[1:]
ref_slope = ref_freqs[0] ** ALPHA * ref_freqs ** (-ALPHA)   # normalised to first point

# ── 4. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("White noise vs Red noise  (α = {:.1f})".format(ALPHA), fontsize=14)

# -- Time series (top row) ---------------------------------------------------
axes[0, 0].plot(t, white, lw=0.6, color="steelblue")
axes[0, 0].set_title("White noise time series")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")

axes[0, 1].plot(t, red, lw=0.6, color="firebrick")
axes[0, 1].set_title(f"Red noise time series  (α = {ALPHA})")
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")

# -- Power spectra (bottom row) ----------------------------------------------
# White noise: flat spectrum
axes[1, 0].loglog(freqs_w[1:], psd_w[1:], lw=0.8, color="steelblue", alpha=0.7,
                  label="PSD")
axes[1, 0].axhline(psd_w[1:].mean(), color="k", ls="--", lw=1.2,
                   label="Mean power")
axes[1, 0].set_title("White noise power spectrum")
axes[1, 0].set_xlabel("Frequency")
axes[1, 0].set_ylabel("Power")
axes[1, 0].legend()

# Red noise: sloped spectrum with reference line
axes[1, 1].loglog(freqs_r[1:], psd_r[1:], lw=0.8, color="firebrick", alpha=0.7,
                  label="PSD")
axes[1, 1].loglog(ref_freqs, ref_slope * psd_r[1], color="k", ls="--", lw=1.5,
                  label=f"1/f^{ALPHA:.1f} reference")
axes[1, 1].set_title(f"Red noise power spectrum  (α = {ALPHA})")
axes[1, 1].set_xlabel("Frequency")
axes[1, 1].set_ylabel("Power")
axes[1, 1].legend()

plt.tight_layout()
# plt.savefig("noise_power_spectrum.png", dpi=150, bbox_inches="tight")
# print("Plot saved to noise_power_spectrum.png")
plt.show()
