"""
Connectivity Module
===================

Jitter-corrected cross-correlogram (CCG) functional connectivity,
faithfully ported from:

    Tang, Disheng et al. (2024).
    "Stimulus type shapes the topology of cellular functional networks
    in mouse visual cortex." Nature Communications 15, 5753.
    https://github.com/HChoiLab/functional-network  (ccg_library.py)
    Author of ccg_library.py: Disheng Tang

KEY DESIGN DECISIONS (from Disheng's implementation):
  - Input: binned spike trains (1ms bins), trial-structured (N x T per trial)
  - CCG computed within each trial window; averaged across trials
  - Jitter: Pattern jitter (Harrison & Geman 2009), L=25 bins
  - Significance: z-score threshold (mean ± n*sigma of surrogate distribution)
  - CCG is one-sided (lags 0 → +window): directed A→B connectivity
  - Parallelism: joblib across trials

Adaptation for Allen Visual Behavior sessions:
  - "Trials" = change-detection trials (active behavior window per trial)
  - Spike trains extracted per unit per trial window
  - Only engaged trials used (Bennett criterion)
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal


# =============================================================================
# PATTERN JITTER  (Harrison & Geman 2009, via Disheng Tang ccg_library.py)
# =============================================================================

class PatternJitter:
    """
    Pattern jitter algorithm for generating synthetic spike trains that
    preserve the recent spiking history of all spikes.

    Harrison, M. T., & Geman, S. (2009). A rate and history-preserving
    resampling algorithm for neural spike trains. Neural Computation, 21(5).

    Port of ccg_library.pattern_jitter by Disheng Tang.
    """

    def __init__(self, num_sample: int, spike_train: np.ndarray,
                 L: int, R: Optional[int] = None,
                 memory: bool = False, seed: Optional[int] = None):
        """
        Parameters
        ----------
        num_sample : int
            Number of surrogate samples to generate.
        spike_train : np.ndarray
            Binary spike train, shape (T,) or (N, T).
        L : int
            Jitter window length in bins.
        R : int, optional
            Memory parameter (max interval to consider recent history).
            Required if memory=True.
        memory : bool
            If True, uses history-preserving (pattern) jitter.
            If False, uses simple spike jitter.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.num_sample = num_sample
        self.spike_train = np.array(spike_train)
        if self.spike_train.ndim > 1:
            self.N, self.T = self.spike_train.shape
        else:
            self.T = len(self.spike_train)
            self.N = None
        self.L = L
        self.memory = memory
        if memory:
            assert R is not None, 'R must be given when memory=True'
        self.R = R
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def _spike_time2train(self, spike_time: np.ndarray) -> np.ndarray:
        if spike_time.ndim == 1:
            train = np.zeros(self.T)
            train[spike_time.astype(int)] = 1
        else:
            train = np.zeros((spike_time.shape[0], self.T))
            train[np.repeat(np.arange(spike_time.shape[0]), spike_time.shape[1]),
                  spike_time.ravel().astype(int)] = 1
        return train

    def _spike_train2time(self, spike_train: np.ndarray) -> np.ndarray:
        if spike_train.ndim == 1:
            return np.squeeze(np.where(spike_train > 0)).ravel()
        times = np.zeros((spike_train.shape[0],
                          len(np.where(spike_train[0, :] > 0)[0])))
        for i in range(spike_train.shape[0]):
            times[i, :] = np.squeeze(np.where(spike_train[i, :] > 0)).ravel()
        return times

    def _get_init_dist(self):
        vals = (self.rng.random(self.L) if self.rng is not None
                else np.random.rand(self.L))
        return vals / vals.sum()

    def _get_transition_matrices(self, num_spike: int) -> np.ndarray:
        mats = np.zeros((num_spike - 1, self.L, self.L))
        for i in range(len(mats)):
            m = (self.rng.random((self.L, self.L)) if self.rng is not None
                 else np.random.rand(self.L, self.L))
            mats[i] = (m / m.sum(axis=1)[:, None]).astype('f')
        return mats

    def _get_omega(self, spike_time: np.ndarray) -> list:
        omega = []
        for s in spike_time:
            lo = max(0, s - int(np.ceil(self.L / 2)) + 1)
            lo = min(lo, self.T - self.L)
            omega.append(np.arange(lo, lo + self.L, 1))
        return omega

    def _get_gamma(self, spike_time: np.ndarray) -> list:
        gamma, ks = [], [0]
        lo = max(0, min(int(spike_time[0] / self.L) * self.L, self.T - self.L))
        gamma.append(np.arange(lo, lo + self.L, 1))
        for i in range(1, len(spike_time)):
            if spike_time[i] - spike_time[i - 1] > self.R:
                ks.append(i)
                lo = max(0, min(
                    int(spike_time[ks[-1]] / self.L) * self.L
                    + spike_time[i] - spike_time[ks[-1]],
                    self.T - self.L))
                gamma.append(np.arange(lo, lo + self.L, 1))
        return gamma

    def _get_surrogate(self, spike_time, init_dist, t_mats) -> list:
        surrogate = []
        jw = self._get_gamma(spike_time) if self.memory else self._get_omega(spike_time)
        # first spike
        rand_x = (self.rng.random() if self.rng is not None else np.random.random())
        idx = np.where(rand_x <= np.cumsum(init_dist))[0][0]
        given_x = jw[0][idx]
        surrogate.append(given_x)
        for i, row in enumerate(t_mats):
            if self.memory and spike_time[i + 1] - spike_time[i] <= self.R:
                given_x = surrogate[-1] + spike_time[i + 1] - spike_time[i]
            else:
                idx2 = np.where(np.array(jw[i]) == given_x)[0]
                p_i = np.squeeze(np.array(row[idx2]))
                init_x = jw[i + 1][0] + np.sum(p_i == 0)
                rx = (self.rng.random() if self.rng is not None else np.random.random())
                larger = np.where(rx <= np.cumsum(p_i))[0]
                ind = larger[0] if len(larger) else len(p_i) - 1
                given_x = init_x + np.sum(p_i[:ind] != 0)
                given_x = min(self.T - 1, given_x)
                if given_x in surrogate:
                    locs = jw[i + 1]
                    available = [l for l in locs if l not in surrogate]
                    if available:
                        given_x = (self.rng.choice(available) if self.rng is not None
                                   else np.random.choice(available))
            surrogate.append(given_x)
        return surrogate

    def jitter(self) -> np.ndarray:
        """Generate num_sample surrogate spike trains."""
        if self.N is not None:
            out = np.zeros((self.num_sample, self.N, self.T))
            for n in range(self.N):
                st = self._spike_train2time(self.spike_train[n, :])
                if st.size:
                    init = self._get_init_dist()
                    mats = self._get_transition_matrices(st.size)
                    for s in range(self.num_sample):
                        surr = self._get_surrogate(st, init, mats)
                        out[s, n, self._spike_time2train(np.array(surr)).astype(int)] = 1
        else:
            out = np.zeros((self.num_sample, self.T))
            st = self._spike_train2time(self.spike_train)
            init = self._get_init_dist()
            mats = self._get_transition_matrices(st.size)
            for s in range(self.num_sample):
                surr = self._get_surrogate(st, init, mats)
                out[s, self._spike_time2train(np.array(surr)).astype(int)] = 1
        return out.squeeze()


# =============================================================================
# CCG CORE  (port of CCG.calculate_ccg_pair_single_trial, Disheng Tang)
# =============================================================================

def compute_ccg_pair(padded_st1: np.ndarray, padded_st2: np.ndarray,
                     firing_rates: np.ndarray,
                     ind_a: int, ind_b: int,
                     T: int, window: int) -> np.ndarray:
    """
    CCG between neuron ind_a and ind_b for a single trial.

    Uses the as_strided sliding-window matrix multiply from ccg_library.py.
    Returns the CCG at lags 0 → +window (one-sided, directed A→B).

    Normalization: divide by sqrt(fr_A * fr_B) * effective_time_s
    (coincidences per sqrt(Hz) per second, matching Disheng's normalization).

    Parameters
    ----------
    padded_st1 : np.ndarray, shape (N, T + 2*window)
        Spike train matrix padded with zeros on both sides.
    padded_st2 : np.ndarray, shape (N, T + window)
        Spike train matrix padded with zeros on the right.
    firing_rates : np.ndarray, shape (N,)
        Firing rates in Hz.
    ind_a, ind_b : int
        Unit indices.
    T : int
        Number of time bins in the unpadded trial.
    window : int
        CCG window in bins.

    Returns
    -------
    np.ndarray, shape (window+1,)
        CCG values at lags 0 … window.
    """
    px = padded_st1[ind_a, :]
    py = padded_st2[ind_b, :]
    shifted = as_strided(
        px[window:],
        shape=(window + 1, T + window),
        strides=(-px.strides[0], px.strides[0]),
    )
    denom = ((T - np.arange(window + 1)) / 1000.0
             * np.sqrt(firing_rates[ind_a] * firing_rates[ind_b]))
    return (shifted @ py) / denom


def compute_all_ccgs_single_trial(spike_train: np.ndarray,
                                  firing_rates: np.ndarray,
                                  window: int) -> np.ndarray:
    """
    All pairwise CCGs for one trial.

    Parameters
    ----------
    spike_train : np.ndarray, shape (N, T)
        Binary spike train for one trial.
    firing_rates : np.ndarray, shape (N,)
        Firing rates in Hz.
    window : int
        CCG window in bins.

    Returns
    -------
    np.ndarray, shape (N, N, window+1)
        CCG matrix. Diagonal is NaN (auto-correlogram excluded).
    """
    N, T = spike_train.shape
    ccgs = np.zeros((N, N, window + 1))
    mask = np.eye(N, dtype=bool)[:, :, None]
    ccgs[np.broadcast_to(mask, ccgs.shape)] = np.nan

    padded_st1 = np.concatenate(
        [np.zeros((N, window)), spike_train, np.zeros((N, window))], axis=1)
    padded_st2 = np.concatenate(
        [spike_train, np.zeros((N, window))], axis=1)

    valid = np.where(firing_rates > 0)[0]
    for ind_a, ind_b in itertools.permutations(valid, 2):
        ccgs[ind_a, ind_b, :] = compute_ccg_pair(
            padded_st1, padded_st2, firing_rates, ind_a, ind_b, T, window)
    return ccgs


# =============================================================================
# TRIAL-AVERAGED CCG WITH JITTER CORRECTION
# =============================================================================

def compute_ccg_corrected(
    spike_tensor: np.ndarray,
    window: int = 100,
    num_jitter: int = 100,
    L: int = 25,
    memory: bool = False,
    seed: Optional[int] = None,
    use_parallel: bool = True,
    num_cores: int = -1,
) -> np.ndarray:
    """
    Compute trial-averaged jitter-corrected CCG matrix.

    Port of CCG.calculate_mean_ccg_corrected from ccg_library.py.

    Parameters
    ----------
    spike_tensor : np.ndarray, shape (N, n_trials, T)
        Binary spike trains per neuron per trial.
    window : int
        CCG lag window in bins (default 100 bins = 100ms at 1ms bins).
    num_jitter : int
        Number of pattern-jitter surrogates per trial.
    L : int
        Jitter window length in bins.
    memory : bool
        Pattern jitter memory parameter.
    seed : int, optional
        Random seed.
    use_parallel : bool
        Use joblib parallel across trials.
    num_cores : int
        Number of CPU cores (-1 = all available).

    Returns
    -------
    np.ndarray, shape (N, N, window+1)
        Jitter-corrected CCG (raw CCG minus mean surrogate CCG).
    """
    N, n_trials, T = spike_tensor.shape
    assert T > window, (
        f"Trial length ({T} bins) must exceed window ({window} bins). "
        "Reduce window or use longer trial epochs.")

    firing_rates = (np.count_nonzero(spike_tensor, axis=(1, 2))
                    / (n_trials * T / 1000.0))

    def _process_trial(trial_idx):
        st = spike_tensor[:, trial_idx, :]
        ccg_trial = compute_all_ccgs_single_trial(st, firing_rates, window)

        pj = PatternJitter(
            num_sample=num_jitter, spike_train=st,
            L=L, memory=memory,
            seed=(None if seed is None
                  else int((seed + 1000003 * trial_idx) % (2**32 - 1)))
        )
        surrogates = pj.jitter()  # shape: (num_jitter, N, T)

        ccg_jitter = np.zeros_like(ccg_trial)
        for j in range(num_jitter):
            ccg_jitter += compute_all_ccgs_single_trial(
                surrogates[j], firing_rates, window)
        return ccg_trial, ccg_jitter / num_jitter

    if use_parallel:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=num_cores)(
                delayed(_process_trial)(i) for i in range(n_trials))
        except ImportError:
            results = [_process_trial(i) for i in range(n_trials)]
    else:
        results = [_process_trial(i) for i in range(n_trials)]

    ccgs = np.zeros((N, N, window + 1))
    ccg_jitter = np.zeros((N, N, window + 1))
    for raw, jit in results:
        ccgs += raw
        ccg_jitter += jit

    return (ccgs - ccg_jitter) / n_trials


# =============================================================================
# SIGNIFICANCE DETECTION  (z-score, port of CCG.get_significant_ccg)
# =============================================================================

def get_significant_connections(
    ccg_corrected: np.ndarray,
    n_sigma: float = 5.0,
    max_duration: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify significant connections from jitter-corrected CCG matrix.

    Port of CCG.get_significant_ccg from ccg_library.py.
    Significance: deviation from mean surrogate > n_sigma standard deviations,
    assessed on integrated CCG over windows of duration 0…max_duration bins.

    Parameters
    ----------
    ccg_corrected : np.ndarray, shape (N, N, window+1)
        Jitter-corrected CCG from compute_ccg_corrected().
    n_sigma : float
        Z-score threshold (default 5.0 from Disheng's lab).
    max_duration : int
        Maximum integration window to test for sharp peaks vs. intervals.

    Returns
    -------
    significant_ccg : np.ndarray (N, N) -- peak CCG value (NaN if not sig.)
    confidence : np.ndarray (N, N) -- z-score of peak
    offset : np.ndarray (N, N) -- lag of peak in bins
    duration : np.ndarray (N, N) -- integration window at peak
    """
    N = ccg_corrected.shape[0]
    sig_ccg = np.full((N, N), np.nan)
    sig_conf = np.full((N, N), np.nan)
    sig_off = np.full((N, N), np.nan)
    sig_dur = np.full((N, N), np.nan)
    maxlag = ccg_corrected.shape[2]

    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    for duration in np.arange(max_duration, -1, -1):
        filt = np.array([[[1]]]).repeat(duration + 1, axis=2)
        ccg_int = signal.convolve(ccg_corrected, filt, mode='valid', method='fft')
        mu = np.nanmean(ccg_int, axis=-1, keepdims=True)
        sigma = np.nanstd(ccg_int, axis=-1, keepdims=True)
        clipped = ccg_int[:, :, :maxlag - duration]
        abs_dev = np.abs(clipped - mu)
        ext_off = np.argmax(abs_dev, axis=-1)
        ext_val = np.take_along_axis(clipped, ext_off[..., None], axis=-1).squeeze(-1)
        mu2 = mu.squeeze(-1)
        sig2 = sigma.squeeze(-1)
        clevel = np.where(sig2 > 0, (ext_val - mu2) / sig2, 0.0)
        sig_mask = np.abs(clevel) > n_sigma
        update = sig_mask & (np.abs(np.nan_to_num(clevel))
                             > np.abs(np.nan_to_num(sig_conf)))
        sig_ccg[update] = ext_val[update]
        sig_conf[update] = clevel[update]
        sig_off[update] = ext_off[update]
        sig_dur[update] = duration

    warnings.resetwarnings()
    return sig_ccg, sig_conf, sig_off, sig_dur


# =============================================================================
# SPIKE TIME → TRIAL SPIKE TRAIN CONVERSION (Allen VB adapter)
# =============================================================================

def spike_times_to_trial_tensor(
    spike_times_dict: Dict[int, np.ndarray],
    unit_ids: List[int],
    trial_start_times: np.ndarray,
    trial_duration_ms: float,
    bin_size_ms: float = 1.0,
    pre_pad_ms: float = 0.0,
) -> np.ndarray:
    """
    Convert spike times to a binary trial spike tensor.

    Parameters
    ----------
    spike_times_dict : dict
        {unit_id: spike_times_array_in_seconds}
    unit_ids : list
        Ordered list of unit IDs (determines row order in tensor).
    trial_start_times : np.ndarray
        Start time of each trial in seconds.
    trial_duration_ms : float
        Duration of each trial window in milliseconds.
    bin_size_ms : float
        Bin size in milliseconds (default 1ms).
    pre_pad_ms : float
        Optional pre-trial padding in milliseconds.

    Returns
    -------
    np.ndarray, shape (N_units, N_trials, T_bins)
        Binary spike tensor.
    """
    n_units = len(unit_ids)
    n_trials = len(trial_start_times)
    T = int(round((trial_duration_ms + pre_pad_ms) / bin_size_ms))
    bin_s = bin_size_ms / 1000.0
    pre_s = pre_pad_ms / 1000.0

    tensor = np.zeros((n_units, n_trials, T), dtype=np.int8)
    for u_idx, uid in enumerate(unit_ids):
        if uid not in spike_times_dict:
            continue
        st = np.asarray(spike_times_dict[uid])
        for t_idx, t_start in enumerate(trial_start_times):
            lo = t_start - pre_s
            hi = lo + (trial_duration_ms + pre_pad_ms) / 1000.0
            mask = (st >= lo) & (st < hi)
            spikes_in_trial = st[mask]
            bins = ((spikes_in_trial - lo) / bin_s).astype(int)
            bins = bins[(bins >= 0) & (bins < T)]
            tensor[u_idx, t_idx, bins] = 1

    return tensor


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    from .constants import (CCG_WINDOW_BINS, CCG_N_SURROGATES,
                             CCG_JITTER_WINDOW_BINS, CCG_N_SIGMA, RANDOM_SEED)
    import time

    rng = np.random.default_rng(RANDOM_SEED)
    N, n_trials, T = 5, 40, 250  # 5 neurons, 40 trials, 250ms each

    # Synthetic: neuron 0 drives neuron 1 with 5ms delay
    spike_tensor = (rng.random((N, n_trials, T)) < 0.02).astype(np.int8)
    delay = 5
    for tr in range(n_trials):
        for t in range(T - delay):
            if spike_tensor[0, tr, t]:
                if t + delay < T:
                    spike_tensor[1, tr, t + delay] = 1

    print("=" * 60)
    print("CONNECTIVITY MODULE TEST (Disheng Tang ccg_library port)")
    print("=" * 60)
    print(f"Tensor shape: {spike_tensor.shape}")

    t0 = time.perf_counter()
    ccg = compute_ccg_corrected(
        spike_tensor, window=CCG_WINDOW_BINS,
        num_jitter=20, L=CCG_JITTER_WINDOW_BINS,
        seed=RANDOM_SEED, use_parallel=False)
    elapsed = time.perf_counter() - t0

    print(f"CCG shape: {ccg.shape}")
    print(f"Time: {elapsed:.2f}s for {N}x{N} pairs, 20 surrogates, {n_trials} trials")

    sig_ccg, sig_conf, sig_off, sig_dur = get_significant_connections(
        ccg, n_sigma=CCG_N_SIGMA)

    sig_pairs = np.argwhere(~np.isnan(sig_ccg))
    print(f"Significant connections: {len(sig_pairs)}")
    for i, j in sig_pairs:
        print(f"  {i} → {j}: lag={sig_off[i,j]:.0f}ms, z={sig_conf[i,j]:.1f}")

    if any(i == 0 and j == 1 for i, j in sig_pairs):
        off_0_1 = sig_off[0, 1]
        if abs(off_0_1 - delay) <= 2:
            print(f"\n[OK] Detected 0→1 monosynaptic connection at lag {off_0_1}ms "
                  f"(injected delay: {delay}ms)")
        else:
            print(f"\n[WARN] 0→1 detected but peak at {off_0_1}ms, expected {delay}ms")
    else:
        print(f"\n[NOTE] 0→1 not significant at n_sigma={CCG_N_SIGMA} "
              f"(may need more trials)")
