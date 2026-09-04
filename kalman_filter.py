"""
Kalman filtering for OHLCV noise reduction (Section III-E / IV-C / IV-D of
the paper).

State-space model (applied independently to each of open/high/low/close/
tick_volume, i.e. diagonal Q, R and F_t = H_t = identity):

    x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)      (random walk with drift = 0)
    y_t = x_t + v_t,       v_t ~ N(0, R)

Q and R are estimated by maximum likelihood via EM on the innovation
sequence (Eq. 21 in the paper). The EM forward/backward passes are plain
Python loops, so for speed they only run on a bounded subsample of the
series. The fitted (Q, R) imply a closed-form **steady-state** Kalman gain;
applying that fixed gain reduces filtering to a first-order IIR filter,
which we run on the *full* series with `scipy.signal.lfilter` -- vectorised
and O(n), which is what makes this fast enough for multi-million-row M1
files.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfiltic


def _forward_pass(y, Q, R):
    n = len(y)
    x_filt = np.empty(n)
    P_filt = np.empty(n)
    x_pred = np.empty(n)
    P_pred = np.empty(n)
    x_filt[0] = y[0]
    P_filt[0] = R
    for t in range(1, n):
        x_pred[t] = x_filt[t - 1]
        P_pred[t] = P_filt[t - 1] + Q
        K = P_pred[t] / (P_pred[t] + R)
        x_filt[t] = x_pred[t] + K * (y[t] - x_pred[t])
        P_filt[t] = (1 - K) * P_pred[t]
    return x_filt, P_filt, x_pred, P_pred


def _backward_pass(x_filt, P_filt, x_pred, P_pred):
    n = len(x_filt)
    x_smooth = np.empty(n)
    P_smooth = np.empty(n)
    P_lag = np.zeros(n)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for t in range(n - 2, -1, -1):
        J = P_filt[t] / P_pred[t + 1] if P_pred[t + 1] > 0 else 0.0
        x_smooth[t] = x_filt[t] + J * (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + J ** 2 * (P_smooth[t + 1] - P_pred[t + 1])
        P_lag[t + 1] = J * P_smooth[t + 1]
    return x_smooth, P_smooth, P_lag


def _fit_em(y, n_iter=15, Q0=1e-5, R0=1e-2, tol=1e-8):
    """EM estimation of (Q, R) on a 1-D series (Eq. 21 of the paper)."""
    y = np.asarray(y, dtype=np.float64)
    Q, R = Q0, R0
    for _ in range(n_iter):
        x_filt, P_filt, x_pred, P_pred = _forward_pass(y, Q, R)
        x_smooth, P_smooth, P_lag = _backward_pass(x_filt, P_filt, x_pred, P_pred)
        Q_new = float(np.mean((x_smooth[1:] - x_smooth[:-1]) ** 2
                               + P_smooth[1:] + P_smooth[:-1] - 2 * P_lag[1:]))
        R_new = float(np.mean((y - x_smooth) ** 2 + P_smooth))
        Q_new = max(Q_new, 1e-10)
        R_new = max(R_new, 1e-10)
        if abs(Q_new - Q) < tol and abs(R_new - R) < tol:
            Q, R = Q_new, R_new
            break
        Q, R = Q_new, R_new
    return Q, R


def _steady_state_gain(Q, R):
    """Closed-form steady-state Kalman gain for the scalar local-level model."""
    ratio = Q / R
    K = (-ratio + np.sqrt(ratio ** 2 + 4 * ratio)) / 2.0
    return float(np.clip(K, 1e-6, 1.0))


class KalmanFilter1D:
    def __init__(self):
        self.Q = None
        self.R = None
        self.K = None  # steady-state gain

    def fit(self, y, n_iter=15, subsample=50_000):
        y = np.asarray(y, dtype=np.float64)
        y = y[np.isfinite(y)]
        if len(y) > subsample:
            idx = np.linspace(0, len(y) - 1, subsample).astype(int)
            y_fit = y[idx]
        else:
            y_fit = y
        self.Q, self.R = _fit_em(y_fit, n_iter=n_iter)
        self.K = _steady_state_gain(self.Q, self.R)
        return self

    def filter(self, y):
        """Vectorised causal (no look-ahead) filtering with the fitted
        steady-state gain: x[t] = (1-K) x[t-1] + K y[t]."""
        y = np.asarray(y, dtype=np.float64)
        b = [self.K]
        a = [1.0, -(1.0 - self.K)]
        zi = lfiltic(b, a, [y[0]])
        x, _ = lfilter(b, a, y, zi=zi)
        return x


class MultivariateKalmanOHLCV:
    """Independent 1D Kalman filter per OHLCV column -- matches the
    diagonal-noise state-space specification used in the paper."""

    COLUMNS = ["open", "high", "low", "close", "tick_volume"]

    def __init__(self):
        self.filters = {c: KalmanFilter1D() for c in self.COLUMNS}

    def fit(self, df: pd.DataFrame, n_iter=15, subsample=50_000):
        for c in self.COLUMNS:
            self.filters[c].fit(df[c].values, n_iter=n_iter, subsample=subsample)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {f"kf_{c}": self.filters[c].filter(df[c].values) for c in self.COLUMNS}
        return pd.DataFrame(out, index=df.index)

    def params(self):
        return {c: {"Q": f.Q, "R": f.R, "K": f.K} for c, f in self.filters.items()}
