"""
Utilities Module
================

Helper functions and utilities for Ca²⁺ signature analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Parameters
    ----------
    path : str
        Directory path
        
    Returns
    -------
    path : Path
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def estimate_bandwidth(t: np.ndarray, signal: np.ndarray) -> float:
    """
    Estimate signal bandwidth using FFT.
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    signal : np.ndarray
        Signal trace
        
    Returns
    -------
    bandwidth : float
        Estimated bandwidth (Hz)
    """
    # Sampling rate
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    
    # FFT
    fft = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(len(signal), dt)
    power = np.abs(fft) ** 2
    
    # Find bandwidth (frequency containing 95% of power)
    cumsum = np.cumsum(power)
    cumsum /= cumsum[-1]
    
    idx_95 = np.where(cumsum >= 0.95)[0]
    if len(idx_95) > 0:
        bandwidth = freqs[idx_95[0]]
    else:
        bandwidth = freqs[-1]
    
    return bandwidth


def estimate_snr(signal: np.ndarray, baseline_fraction: float = 0.1) -> float:
    """
    Estimate signal-to-noise ratio.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal trace
    baseline_fraction : float
        Fraction of trace to use for baseline estimation
        
    Returns
    -------
    snr : float
        Signal-to-noise ratio (linear, not dB)
    """
    # Baseline from first portion
    n_baseline = int(len(signal) * baseline_fraction)
    baseline = signal[:n_baseline]
    
    # Noise = std of baseline
    noise = np.std(baseline)
    
    # Signal = max - baseline mean
    signal_amplitude = np.max(signal) - np.mean(baseline)
    
    # SNR
    if noise > 0:
        snr = signal_amplitude / noise
    else:
        snr = np.inf
    
    return snr


def detect_oscillations(t: np.ndarray, signal: np.ndarray,
                       threshold: float = 0.5) -> Tuple[bool, Optional[float]]:
    """
    Detect if signal is oscillatory and estimate frequency.
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    signal : np.ndarray
        Signal trace
    threshold : float
        Relative threshold for peak detection
        
    Returns
    -------
    is_oscillatory : bool
        Whether signal is oscillatory
    frequency : Optional[float]
        Dominant frequency if oscillatory (Hz)
    """
    from scipy.signal import find_peaks
    
    # Find peaks
    baseline = np.median(signal[:int(len(signal)*0.1)])
    peak_height = baseline + threshold * (np.max(signal) - baseline)
    
    min_distance = max(1, int(round(5.0 / (t[1] - t[0]))))  # At least 5s between peaks (>= 1 sample)
    peaks, _ = find_peaks(signal, height=peak_height, distance=min_distance)
    
    # Classify as oscillatory if >2 peaks
    is_oscillatory = len(peaks) > 2
    
    if is_oscillatory and len(peaks) > 2:
        # Estimate frequency from inter-peak intervals
        peak_times = t[peaks]
        intervals = np.diff(peak_times)
        frequency = 1.0 / np.mean(intervals)
    else:
        frequency = None
    
    return is_oscillatory, frequency


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix [n_samples, n_features]
        
    Returns
    -------
    normalized : np.ndarray
        Normalized features
    means : np.ndarray
        Feature means
    stds : np.ndarray
        Feature standard deviations
    """
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0) + 1e-10  # Avoid division by zero
    
    normalized = (features - means) / stds
    
    return normalized, means, stds


def calculate_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Data matrix [n_samples, n_features]
        
    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix [n_features, n_features]
    """
    return np.corrcoef(matrix.T)


def export_to_csv(data: Dict[str, np.ndarray], filepath: str):
    """
    Export dictionary of arrays to CSV.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary with column_name: data pairs
    filepath : str
        Output file path
    """
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Exported to {filepath}")


def create_summary_table(results: Dict) -> pd.DataFrame:
    """
    Create summary table from results dictionary.
    
    Parameters
    ----------
    results : Dict
        Dictionary of results
        
    Returns
    -------
    df : pd.DataFrame
        Summary table
    """
    rows = []
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            rows.append({'Metric': key, 'Value': value})
        elif isinstance(value, (list, np.ndarray)):
            rows.append({'Metric': key, 'Value': f'Array ({len(value)} elements)'})
    
    return pd.DataFrame(rows)


class ProgressTracker:
    """Simple progress tracker for long-running analyses."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
    def update(self, step: int = 1, message: str = ""):
        """Update progress."""
        self.current_step += step
        percent = 100 * self.current_step / self.total_steps
        
        bar_length = 40
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = '█' * filled + '-' * (bar_length - filled)
        
        print(f'\r{self.description}: |{bar}| {percent:.1f}% {message}', end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line at completion
    
    def finish(self, message: str = "Complete"):
        """Mark as finished."""
        self.current_step = self.total_steps
        self.update(0, message)


if __name__ == "__main__":
    # Test utilities
    print("Testing Utilities Module\n")
    
    # Test 1: Directory creation
    print("Test 1: Directory creation")
    test_dir = ensure_directory("/home/claude/ca_signature_analyzer/test_output")
    print(f"  Created: {test_dir}\n")
    
    # Test 2: Signal analysis
    print("Test 2: Signal analysis")
    t = np.linspace(0, 100, 1000)
    signal = 100 + 500 * (np.sin(2*np.pi*0.5*t) + 1) / 2
    
    bandwidth = estimate_bandwidth(t, signal)
    snr = estimate_snr(signal)
    is_osc, freq = detect_oscillations(t, signal)
    
    print(f"  Bandwidth: {bandwidth:.3f} Hz")
    print(f"  SNR: {snr:.2f}")
    print(f"  Oscillatory: {is_osc}, Frequency: {freq:.3f} Hz\n")
    
    # Test 3: Progress tracker
    print("Test 3: Progress tracker")
    tracker = ProgressTracker(10, "Test progress")
    for i in range(10):
        import time
        time.sleep(0.1)
        tracker.update(1, f"Step {i+1}")
    
    print("\nUtilities tests complete!")
