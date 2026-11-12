"""
Biophysical Model Module
=========================

ODE-based simulation of plant Ca²⁺ dynamics including:
- Plasma membrane channels and pumps
- ER/vacuolar Ca²⁺ stores
- InsP3 signaling

Buffering is not modelled: ModelParameters.B_total and K_buffer (and
Ca_vac_max) are defined but not used by the equations.
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class ModelParameters:
    """Parameters for the Ca²⁺ dynamics model."""
    
    # Channel conductances (nM/s)
    V_PM_GLR: float = 50.0      # PM GLR channel activity
    V_PM_CNGC: float = 30.0     # PM CNGC channel activity
    V_ER_release: float = 100.0  # ER InsP3-induced release
    V_vac_release: float = 80.0  # Vacuolar release
    
    # Pump activities (nM/s)
    V_PM_pump: float = 40.0     # PM Ca-ATPase
    V_ER_pump: float = 60.0     # ER Ca-ATPase (SERCA-like)
    V_vac_pump: float = 50.0    # Vacuolar Ca-ATPase
    
    # Half-maximal concentrations (nM)
    K_pump: float = 200.0       # Pump affinity for Ca
    K_ER: float = 300.0         # ER release sensitivity
    K_vac: float = 500.0        # Vacuolar release sensitivity
    
    # InsP3 parameters
    K_IP3: float = 100.0        # InsP3 binding affinity (nM)
    n_IP3: float = 2.0          # InsP3 cooperativity
    V_PLC: float = 50.0         # PLC activity (nM/s)
    k_IP3_deg: float = 0.5      # InsP3 degradation rate (1/s)
    
    # Buffering
    B_total: float = 100000.0   # Total buffer concentration (nM)
    K_buffer: float = 500.0     # Buffer affinity for Ca (nM)
    
    # Store capacities (nM)
    Ca_ER_max: float = 100000.0
    Ca_vac_max: float = 500000.0
    
    # Volumes (relative)
    vol_ratio_ER: float = 0.1   # ER volume / cytosol volume
    vol_ratio_vac: float = 10.0  # Vacuole volume / cytosol volume


class CalciumDynamicsModel:
    """Plant Ca²⁺ dynamics ODE model."""
    
    def __init__(self, params: Optional[ModelParameters] = None):
        self.params = params if params is not None else ModelParameters()
        
        # State variables: [Ca_cyt, Ca_ER, Ca_vac, IP3]
        self.state_names = ['Ca_cyt', 'Ca_ER', 'Ca_vac', 'IP3']
        
    def stimulus_function(self, t: float, stimulus_type: str, 
                          intensity: float = 1.0) -> Dict[str, float]:
        """
        Define stimulus-specific activation patterns.
        
        Returns dict with modulation factors for channels/enzymes.
        """
        # Default: no stimulus
        modulation = {
            'GLR_activity': 0.0,
            'CNGC_activity': 0.0,
            'PLC_activity': 0.0,
        }
        
        if stimulus_type == "ABA":
            # ABA increases PM Ca2+ influx and PLC activity
            if t > 60:  # 60s latency
                modulation['GLR_activity'] = intensity * 0.5
                modulation['PLC_activity'] = intensity * 1.0
        
        elif stimulus_type == "NaCl":
            # NaCl causes rapid PM channel opening
            if t > 10:
                modulation['GLR_activity'] = intensity * 0.8
                modulation['CNGC_activity'] = intensity * 0.6
        
        elif stimulus_type == "mechanical_touch":
            # Touch causes very rapid, transient activation
            if 0.5 < t < 5:
                modulation['GLR_activity'] = intensity * 2.0
        
        elif stimulus_type == "flg22":
            # PAMP causes biphasic response
            if t > 5:
                # First phase: PM channels
                if t < 60:
                    modulation['GLR_activity'] = intensity * 1.0
                # Second phase: PLC activation
                if t > 30:
                    modulation['PLC_activity'] = intensity * 0.8
        
        elif stimulus_type == "cold_shock":
            # Cold shock rapid transient
            if 2 < t < 60:
                modulation['GLR_activity'] = intensity * 1.2
        
        elif stimulus_type == "H2O2":
            # H2O2 causes oscillatory PLC activation
            if t > 30:
                # Oscillatory component
                modulation['PLC_activity'] = intensity * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * (t - 30) / 60))
                modulation['GLR_activity'] = intensity * 0.3
        
        elif stimulus_type == "glutamate":
            # Glutamate directly activates GLRs
            if t > 1:
                # Rapid, saturating activation
                tau = 5.0
                modulation['GLR_activity'] = intensity * (1.0 - np.exp(-(t-1)/tau))
        
        return modulation
    
    def derivatives(self, y: np.ndarray, t: float, 
                   stimulus_type: str, intensity: float) -> np.ndarray:
        """
        Calculate derivatives for ODE system.
        
        State vector: [Ca_cyt, Ca_ER, Ca_vac, IP3]
        """
        Ca_cyt, Ca_ER, Ca_vac, IP3 = y
        p = self.params
        
        # Get stimulus modulation
        stim = self.stimulus_function(t, stimulus_type, intensity)
        
        # --- Fluxes ---
        
        # PM Ca2+ influx
        J_PM_in = (p.V_PM_GLR * stim['GLR_activity'] + 
                   p.V_PM_CNGC * stim['CNGC_activity'])
        
        # ER Ca2+ release (InsP3-induced)
        J_ER_release = (p.V_ER_release * 
                       (IP3**p.n_IP3 / (p.K_IP3**p.n_IP3 + IP3**p.n_IP3)) *
                       (Ca_ER / p.Ca_ER_max))
        
        # Vacuolar Ca2+ release (voltage/ligand-gated)
        J_vac_release = (p.V_vac_release * 
                        (Ca_vac / (p.K_vac + Ca_vac)) *
                        stim['GLR_activity'])  # Linked to depolarization
        
        # PM Ca2+ pump (efflux)
        J_PM_pump = p.V_PM_pump * (Ca_cyt**2 / (p.K_pump**2 + Ca_cyt**2))
        
        # ER Ca2+ pump (uptake)
        J_ER_pump = p.V_ER_pump * (Ca_cyt**2 / (p.K_pump**2 + Ca_cyt**2))
        
        # Vacuolar Ca2+ pump (uptake)
        J_vac_pump = p.V_vac_pump * (Ca_cyt**2 / (p.K_pump**2 + Ca_cyt**2))
        
        # InsP3 synthesis (PLC activity)
        J_IP3_synth = p.V_PLC * stim['PLC_activity']
        
        # InsP3 degradation
        J_IP3_deg = p.k_IP3_deg * IP3
        
        # --- ODEs ---
        
        # Cytosolic Ca2+
        dCa_cyt = (J_PM_in + 
                   J_ER_release / p.vol_ratio_ER + 
                   J_vac_release / p.vol_ratio_vac - 
                   J_PM_pump - 
                   J_ER_pump - 
                   J_vac_pump)
        
        # ER Ca2+
        dCa_ER = p.vol_ratio_ER * (J_ER_pump - J_ER_release)
        
        # Vacuolar Ca2+
        dCa_vac = p.vol_ratio_vac * (J_vac_pump - J_vac_release)
        
        # InsP3
        dIP3 = J_IP3_synth - J_IP3_deg
        
        return np.array([dCa_cyt, dCa_ER, dCa_vac, dIP3])
    
    VALID_STIMULI = ("ABA", "NaCl", "mechanical_touch", "flg22",
                     "cold_shock", "H2O2", "glutamate")

    def simulate(self, stimulus_type: str, duration: float = 600.0,
                dt: float = 0.1, intensity: float = 1.0,
                initial_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Ca²⁺ dynamics.

        Parameters
        ----------
        stimulus_type : str
            Type of stimulus ('ABA', 'NaCl', 'mechanical_touch', etc.)
        duration : float
            Simulation duration (seconds)
        dt : float
            Time step (seconds)
        intensity : float
            Stimulus intensity (0-1 scale, or higher)
        initial_state : Optional[np.ndarray]
            Initial state [Ca_cyt, Ca_ER, Ca_vac, IP3]

        Returns
        -------
        t : np.ndarray
            Time points
        y : np.ndarray
            State trajectories [n_timepoints, 4]

        Raises
        ------
        ValueError
            If stimulus_type is invalid, duration is negative, or dt is non-positive.
        """
        if stimulus_type not in self.VALID_STIMULI:
            raise ValueError(
                f"Invalid stimulus_type '{stimulus_type}'. "
                f"Must be one of: {self.VALID_STIMULI}"
            )
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if intensity < 0:
            raise ValueError(f"intensity must be >= 0, got {intensity}")

        # Time points
        t = np.arange(0, duration, dt)
        
        # Initial conditions
        if initial_state is None:
            y0 = np.array([
                100.0,    # Ca_cyt (nM) - resting level
                50000.0,  # Ca_ER (nM) - ~500 μM
                300000.0, # Ca_vac (nM) - ~300 μM  
                10.0      # IP3 (nM) - basal
            ])
        else:
            y0 = initial_state
        
        # Solve ODE
        solution = odeint(self.derivatives, y0, t, args=(stimulus_type, intensity))
        
        return t, solution
    
    def extract_signature_features(self, t: np.ndarray, ca_trace: np.ndarray) -> Dict:
        """
        Extract signature features from simulated Ca²⁺ trace.
        
        Parameters
        ----------
        t : np.ndarray
            Time points
        ca_trace : np.ndarray
            Cytosolic Ca²⁺ concentration
            
        Returns
        -------
        features : Dict
            Dictionary of signature features
        """
        baseline = np.median(ca_trace[:int(len(ca_trace)*0.1)])  # First 10%
        peak = np.max(ca_trace)
        
        # Find first crossing of half-max
        threshold = baseline + 0.5 * (peak - baseline)
        above_threshold = ca_trace > threshold
        
        if not np.any(above_threshold):
            # No significant response
            return {
                'baseline_ca': baseline,
                'peak_amplitude': peak,
                'rise_time': np.nan,
                'decay_tau': np.nan,
                'duration': 0.0,
                'latency': np.nan,
                'oscillatory': False,
                'frequency': None,
                'n_peaks': 0
            }
        
        # Latency: time to first crossing
        first_cross_idx = np.where(above_threshold)[0][0]
        latency = t[first_cross_idx]
        
        # Rise time: 10-90%
        rise_10 = baseline + 0.1 * (peak - baseline)
        rise_90 = baseline + 0.9 * (peak - baseline)
        
        idx_10 = np.where(ca_trace > rise_10)[0]
        idx_90 = np.where(ca_trace > rise_90)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = t[idx_90[0]] - t[idx_10[0]]
        else:
            rise_time = np.nan
        
        # Duration: time above half-max
        duration = np.sum(above_threshold) * (t[1] - t[0])
        
        # Decay tau: fit exponential to decay phase
        peak_idx = np.argmax(ca_trace)
        if peak_idx < len(ca_trace) - 10:
            decay_trace = ca_trace[peak_idx:] - baseline
            decay_time = t[peak_idx:] - t[peak_idx]
            
            # Fit exponential: y = A * exp(-t/tau)
            if len(decay_trace) > 5 and np.max(decay_trace) > 0:
                log_decay = np.log(decay_trace + 1e-10)
                valid = np.isfinite(log_decay)
                if np.sum(valid) > 5:
                    coeffs = np.polyfit(decay_time[valid], log_decay[valid], 1)
                    decay_tau = -1.0 / coeffs[0] if coeffs[0] != 0 else np.nan
                else:
                    decay_tau = np.nan
            else:
                decay_tau = np.nan
        else:
            decay_tau = np.nan
        
        # Detect oscillations
        from scipy.signal import find_peaks
        # At least 5 s between peaks (in samples; round, since 5 // 0.1 == 49.0 in floating point)
        min_distance = max(1, int(round(5.0 / (t[1] - t[0]))))
        peaks_idx, _ = find_peaks(ca_trace, height=threshold, distance=min_distance)
        
        oscillatory = len(peaks_idx) > 1
        frequency = None
        if oscillatory and len(peaks_idx) > 2:
            # Calculate frequency from inter-peak intervals
            peak_times = t[peaks_idx]
            intervals = np.diff(peak_times)
            frequency = 1.0 / np.mean(intervals)  # Hz
        
        return {
            'baseline_ca': float(baseline),
            'peak_amplitude': float(peak),
            'rise_time': float(rise_time),
            'decay_tau': float(decay_tau),
            'duration': float(duration),
            'latency': float(latency),
            'oscillatory': oscillatory,
            'frequency': float(frequency) if frequency is not None else None,
            'n_peaks': len(peaks_idx)
        }


def run_example_simulations():
    """Run example simulations for different stimuli."""
    model = CalciumDynamicsModel()
    
    stimuli = ['ABA', 'NaCl', 'mechanical_touch', 'flg22', 'cold_shock', 'H2O2', 'glutamate']
    
    fig, axes = plt.subplots(len(stimuli), 1, figsize=(10, 12))
    
    for i, stimulus in enumerate(stimuli):
        t, sol = model.simulate(stimulus, duration=600, intensity=1.0)
        ca_cyt = sol[:, 0]
        
        axes[i].plot(t, ca_cyt / 1000, 'b-', linewidth=1.5)
        axes[i].set_ylabel('[Ca²⁺] (μM)')
        axes[i].set_title(f'{stimulus}', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Extract features
        features = model.extract_signature_features(t, ca_cyt)
        print(f"\n{stimulus}:")
        for key, value in features.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                print(f"  {key}: {value}")
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(str(output_dir / "example_simulations.png"), dpi=150)
    print(f"\nSaved plot to {output_dir / 'example_simulations.png'}")


if __name__ == "__main__":
    run_example_simulations()
