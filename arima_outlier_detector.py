# arima_outlier_detector.py
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import find_peaks

class ARIMAOutlierDetector:
    """
    ARIMA-based outlier detector implementing Chang et al. (1988) for AO/IO.
    
    Fits an ARIMA(p,d,q), computes residuals, and flags points where the 
    test statistics (scaled residual or its ARMA-filtered sum) exceed a threshold.
    
    Parameters
    ----------
    p, d, q : int
        ARIMA order. If None, will use ARIMA(1,0,1) by default or could use AIC selection.
    threshold : float
        Critical value C for the test statistic (default 3.0 for high sensitivity:contentReference[oaicite:5]{index=5}).
    robust_sigma : bool
        If True, estimate residual sigma by median(|resid|)*sqrt(pi/2) for robustness.
    """
    def __init__(self, p=1, d=0, q=1, threshold=3.0, robust_sigma=True):
        self.p = p
        self.d = d
        self.q = q
        self.threshold = threshold
        self.robust_sigma = robust_sigma
        # Will be filled after fit:
        self.resid_ = None
        self.sigma_ = None
        self.lambda1_ = None
        self.lambda2_ = None
        self.anomaly_scores_ = None
        self.is_outlier_ = None
        self.model_ = None
        self.fitted_ = None

    def fit(self, series):
        """
        Fit ARIMA to the series and detect outliers.
        
        Parameters
        ----------
        series : array-like or pandas.Series
            1D time series of observations.
            
        Sets
        ----
        anomaly_scores_ : numpy array of shape (n,)
            The outlier score at each point (max of IO/AO stats).
        is_outlier_ : bool array of shape (n,)
            True where the score exceeds the threshold.
        """
        # Ensure pandas Series for index alignment
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        series = series.astype(float)
        
        # Fit ARIMA model (no constant by default)
        self.model_ = ARIMA(series, order=(self.p, self.d, self.q))
        self.fitted_ = self.model_.fit()
        
        # Get residuals
        resid = self.fitted_.resid
        n = len(resid)
        self.resid_ = np.asarray(resid)
        
        # Estimate sigma (noise std); use robust estimate if requested:contentReference[oaicite:6]{index=6}.
        if self.robust_sigma:
            med_abs = np.median(np.abs(self.resid_ - np.median(self.resid_)))
            self.sigma_ = med_abs * np.sqrt(np.pi/2)
        else:
            self.sigma_ = np.std(self.resid_, ddof=1)
        if self.sigma_ <= 0:
            self.sigma_ = 1.0
        
        # Compute test statistics
        # λ1_t = residual / sigma  (IO test statistic)
        self.lambda1_ = self.resid_ / self.sigma_
        
        # λ2_t = AO statistic: we approximate by filtering future residuals.
        # If ARMA(φ,θ) known, one would compute λ2 exactly; here we use a simple approach:
        arparams = np.r_[1, -self.fitted_.arparams] if hasattr(self.fitted_, 'arparams') else [1.0]
        # Compute truncated ARMA impulse response (up to L future lags)
        L = 5  # small horizon for approximate effect of AO
        lambda2 = np.zeros(n)
        for t in range(n):
            # sum future residuals with AR weights as a simple proxy
            weights = [arparams[0]]
            for k in range(1, L+1):
                if k < len(arparams):
                    weights.append(arparams[k])
                else:
                    weights.append(0.0)
            # Weighted sum of residuals at t..t+L (if available)
            segment = self.resid_[t : min(n, t+L+1)]
            w = np.array(weights[:len(segment)])
            # compute a test statistic c_hat for an AO at t (scaled sum)
            c_hat = np.dot(w, segment)
            lambda2[t] = c_hat / self.sigma_
        self.lambda2_ = lambda2
        
        # Anomaly score = max(|λ1|, |λ2|) at each t
        self.anomaly_scores_ = np.maximum(np.abs(self.lambda1_), np.abs(self.lambda2_))
        # Flag as outlier if score > threshold
        self.is_outlier_ = self.anomaly_scores_ > self.threshold
        
        return self

    def plot_series_with_outliers(self, series):
        """
        (Optional) Plot the series and mark detected outliers for visualization.
        """
        import matplotlib.pyplot as plt
        if not hasattr(self, 'anomaly_scores_'):
            raise RuntimeError("Fit the detector before plotting.")
        x = np.arange(len(series))
        plt.figure(figsize=(10,4))
        plt.plot(x, series, label='Series')
        # Mark outliers
        out_idx = np.where(self.is_outlier_)[0]
        # red dots smaller size
        plt.plot(out_idx, series.iloc[out_idx], 'ro', markersize=1, label='Detected Outlier')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('ARIMA Outlier Detection (p={},d={},q={})'.format(self.p, self.d, self.q))
        plt.show()



class SlidingWindowARIMAOutlierDetector:
    """
    ARIMA-based outlier detector using sliding windows.

    Args:
        window_size (int): Number of time steps in each ARIMA window.
        step_size (int): How far to slide the window each step.
        p, d, q: ARIMA order.
        threshold: Outlier detection threshold on λ score.
        robust_sigma: Whether to estimate residual sigma robustly.
    """
    def __init__(self, window_size=180, step_size=30, p=1, d=0, q=1, threshold=3.0, robust_sigma=True):
        self.window_size = window_size
        self.step_size = step_size
        self.p = p
        self.d = d
        self.q = q
        self.threshold = threshold
        self.robust_sigma = robust_sigma

        # Outputs
        self.anomaly_scores_ = None
        self.is_outlier_ = None

    def fit(self, series):
        """
        Apply ARIMA-based outlier detection in sliding windows.
        
        Args:
            series (pd.Series): Full time series to process.
        """
        series = series.astype(float)
        n = len(series)
        scores = np.zeros(n)
        counts = np.zeros(n)

        for start in range(0, n - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_series = series.iloc[start:end]
            try:
                detector = ARIMAOutlierDetector(
                    p=self.p, d=self.d, q=self.q,
                    threshold=self.threshold,
                    robust_sigma=self.robust_sigma
                )
                detector.fit(window_series)

                # Assign scores back into global array
                scores[start:end] += detector.anomaly_scores_
                counts[start:end] += 1
            except Exception as e:
                print(f"Warning: ARIMA failed in window {start}:{end} - {e}")

        # Average overlapping windows
        counts[counts == 0] = 1
        self.anomaly_scores_ = scores / counts
        self.is_outlier_ = self.anomaly_scores_ > self.threshold
        return self

class ARIMAOutlierDetectorEnhanced:
    def __init__(self, p=1, d=0, q=1, base_threshold=3.0, robust_sigma=True):
        self.p = p
        self.d = d
        self.q = q
        self.base_threshold = base_threshold
        self.robust_sigma = robust_sigma


    def fit(self, series):
        # --- 1) Fit ARIMA and get residuals as before ---
        series = pd.Series(series).astype(float)
        model = ARIMA(series, order=(self.p, self.d, self.q))
        self.fitted_ = model.fit()
        resid = np.asarray(self.fitted_.resid)
        # robust sigma estimate
        med_abs = np.median(np.abs(resid - np.median(resid)))
        self.sigma_ = med_abs * np.sqrt(np.pi/2) if self.robust_sigma else np.std(resid, ddof=1)
        if self.sigma_ <= 0: self.sigma_ = 1.0

        # Compute λ1 and λ2 as before
        lambda1 = resid / self.sigma_
        arparams = np.r_[1, -self.fitted_.arparams] if hasattr(self.fitted_, 'arparams') else np.array([1.0])
        L = 5
        lambda2 = np.zeros_like(lambda1)
        for t in range(len(resid)):
            # compute additive-outlier statistic (approx)
            weights = np.zeros(L+1)
            weights[0] = arparams[0]
            for k in range(1, L+1):
                weights[k] = arparams[k] if k < len(arparams) else 0.0
            segment = resid[t: t+L+1]
            lambda2[t] = np.dot(weights[:len(segment)], segment) / self.sigma_

        # --- 2) Combine λ1 and λ2 into composite score ---
        # Option A: Euclidean norm
        combined_score = np.sqrt(lambda1**2 + lambda2**2)
        # Option B: weighted max (uncomment if preferred)
        # combined_score = np.maximum(np.abs(lambda1), np.abs(lambda2))
        self.anomaly_scores_ = combined_score

        # --- 3) Adaptive thresholding (robust) ---
        med_score = np.median(combined_score)
        mad_score = np.median(np.abs(combined_score - med_score))
        # e.g., 3*MAD above median as threshold
        adaptive_thresh = med_score + 3.0 * mad_score
        self.is_outlier_ = (combined_score > adaptive_thresh)

        return self

class SlidingWindowARIMAOutlierDetectorEnhanced:
    def __init__(self, window_size=180, step_size=30, **kwargs):
        # pass p,d,q,threshold to ARIMAOutlierDetectorEnhanced
        self.window_size = window_size
        self.step_size = step_size
        self.detector_kwargs = kwargs


    def fit(self, series):
        n = len(series)
        scores_accum = np.zeros(n)
        counts = np.zeros(n)
        for start in range(0, n - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window = series.iloc[start:end]
            try:
                det = ARIMAOutlierDetectorEnhanced(**self.detector_kwargs)
                det.fit(window)
                scores_accum[start:end] += det.anomaly_scores_
                counts[start:end] += 1
            except Exception as e:
                print(f"ARIMA failed on window {start}:{end} - {e}")
        # Average overlapping scores
        counts[counts == 0] = 1
        avg_scores = scores_accum / counts
        # Optional: smooth the averaged scores
        window_len = 5
        kernel = np.ones(window_len) / window_len
        smoothed = np.convolve(avg_scores, kernel, mode='same')
        self.anomaly_scores_ = smoothed
        # Determine global threshold or re-use average threshold
        med = np.median(smoothed)
        mad = np.median(np.abs(smoothed - med))
        self.is_outlier_ = (smoothed > med + 3.0*mad)
        return self


def main():
    # Simulated example data with outliers
    # ARMA(1,0) process with injected outliers
    np.random.seed(42)
    n = 200
    data = np.cumsum(np.random.randn(n))  # random walk
    # Inject some outliers
    outlier_indices = [50, 120, 180]
    data[outlier_indices] += np.array([15, -20, 25])
    series = pd.Series(data)
    
    detector = ARIMAOutlierDetector(p=1, d=0, q=0, threshold=12.0, robust_sigma=True)
    detector.fit(series)
    
    print("Detected outliers at indices:", np.where(detector.is_outlier_)[0])
    peaks, _ = find_peaks(detector.anomaly_scores_, height=detector.threshold)
    print("Final outlier peaks:", peaks)
    # Optional: plot results
    detector.plot_series_with_outliers(series)
  
if __name__ == "__main__":
    main()