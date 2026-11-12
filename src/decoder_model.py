"""
Decoder Model Module
====================

Models for Ca²⁺ decoders including:
- CDPK (Calcium-Dependent Protein Kinases)
- CBL-CIPK networks
- Calmodulin
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.integrate import odeint

# np.trapz was removed in NumPy 2.0 (renamed np.trapezoid); support both.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


@dataclass
class CDPKParameters:
    """Parameters for a CDPK decoder."""
    
    name: str
    K_d: float           # Ca2+ dissociation constant (nM)
    n_Hill: float        # Cooperativity (Hill coefficient)
    k_on: float          # Activation rate (1/nM/s)
    k_off: float         # Deactivation rate (1/s)
    k_cat: float         # Catalytic rate for phosphorylation (1/s)
    
    def __post_init__(self):
        """Validate parameters."""
        assert self.K_d > 0, "K_d must be positive"
        assert self.n_Hill > 0, "Hill coefficient must be positive"
        assert self.k_on > 0, "k_on must be positive"
        assert self.k_off > 0, "k_off must be positive"


class CDPKLibrary:
    """Library of Arabidopsis CDPK parameters."""
    
    @staticmethod
    def get_default_cdpks() -> List[CDPKParameters]:
        """
        Get default parameters for major Arabidopsis CDPKs.
        
        Parameters are estimated from literature or predicted from
        EF-hand sequences.
        """
        cdpks = [
            # Low-affinity CDPKs (respond to high-amplitude signals)
            CDPKParameters(
                name="CPK1",
                K_d=2000.0,  # 2 μM
                n_Hill=3.0,
                k_on=1e8,
                k_off=100.0,
                k_cat=10.0
            ),
            CDPKParameters(
                name="CPK2",
                K_d=1500.0,
                n_Hill=2.8,
                k_on=5e7,
                k_off=80.0,
                k_cat=8.0
            ),
            
            # Medium-affinity CDPKs (general sensors)
            CDPKParameters(
                name="CPK3",
                K_d=500.0,  # 500 nM
                n_Hill=3.2,
                k_on=1e8,
                k_off=50.0,
                k_cat=12.0
            ),
            CDPKParameters(
                name="CPK4",
                K_d=400.0,
                n_Hill=3.0,
                k_on=8e7,
                k_off=45.0,
                k_cat=11.0
            ),
            CDPKParameters(
                name="CPK5",  # Important for immunity
                K_d=300.0,
                n_Hill=3.5,
                k_on=1.2e8,
                k_off=40.0,
                k_cat=15.0
            ),
            CDPKParameters(
                name="CPK6",  # Important for immunity
                K_d=350.0,
                n_Hill=3.4,
                k_on=1.1e8,
                k_off=42.0,
                k_cat=14.0
            ),
            
            # High-affinity CDPKs (respond to low-amplitude signals)
            CDPKParameters(
                name="CPK21",  # Salt stress
                K_d=150.0,
                n_Hill=2.5,
                k_on=5e7,
                k_off=25.0,
                k_cat=8.0
            ),
            CDPKParameters(
                name="CPK23",  # Salt stress
                K_d=180.0,
                n_Hill=2.6,
                k_on=6e7,
                k_off=28.0,
                k_cat=9.0
            ),
            
            # Very high-affinity CDPKs
            CDPKParameters(
                name="CPK28",
                K_d=100.0,
                n_Hill=2.0,
                k_on=3e7,
                k_off=20.0,
                k_cat=6.0
            ),
            CDPKParameters(
                name="CPK32",
                K_d=120.0,
                n_Hill=2.2,
                k_on=4e7,
                k_off=22.0,
                k_cat=7.0
            ),
        ]
        
        return cdpks


class CDPKDecoder:
    """Model CDPK activation in response to Ca²⁺ signals."""
    
    def __init__(self, cdpk_params: CDPKParameters):
        self.params = cdpk_params
        
    def steady_state_activation(self, ca_concentration: float) -> float:
        """
        Calculate steady-state CDPK activation (Hill equation).
        
        Parameters
        ----------
        ca_concentration : float
            Ca²⁺ concentration (nM)
            
        Returns
        -------
        f_active : float
            Fraction of activated CDPK (0-1)
        """
        p = self.params
        ca_n = ca_concentration ** p.n_Hill
        kd_n = p.K_d ** p.n_Hill
        return ca_n / (kd_n + ca_n)
    
    def dynamic_activation(self, t: np.ndarray, ca_trace: np.ndarray) -> np.ndarray:
        """
        Calculate dynamic CDPK activation with kinetics.
        
        Includes activation/deactivation kinetics, allowing frequency
        decoding through temporal filtering.
        
        Parameters
        ----------
        t : np.ndarray
            Time points (s)
        ca_trace : np.ndarray
            Ca²⁺ concentration over time (nM)
            
        Returns
        -------
        activation : np.ndarray
            CDPK activation over time (0-1)
        """
        p = self.params
        activation = np.zeros_like(ca_trace)
        
        # Initial condition
        activation[0] = self.steady_state_activation(ca_trace[0])
        
        # Integrate activation kinetics
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            ca = ca_trace[i]
            
            # Activation rate
            d_activation = (p.k_on * ca * (1 - activation[i-1]) - 
                          p.k_off * activation[i-1])
            
            activation[i] = activation[i-1] + d_activation * dt
            activation[i] = np.clip(activation[i], 0, 1)
        
        return activation
    
    def calculate_integrated_activation(self, t: np.ndarray, ca_trace: np.ndarray) -> float:
        """
        Calculate time-integrated CDPK activation.
        
        This represents the cumulative activity of the decoder,
        which relates to target phosphorylation.
        """
        activation = self.dynamic_activation(t, ca_trace)
        return _trapezoid(activation, t)


class DecoderPanel:
    """Panel of multiple CDPK decoders for signature analysis."""
    
    def __init__(self, cdpks: Optional[List[CDPKParameters]] = None):
        if cdpks is None:
            cdpks = CDPKLibrary.get_default_cdpks()
        
        self.decoders = [CDPKDecoder(params) for params in cdpks]
        self.names = [d.params.name for d in self.decoders]
        
    def analyze_signature(self, t: np.ndarray, ca_trace: np.ndarray) -> Dict:
        """
        Analyze Ca²⁺ signature with full decoder panel.
        
        Returns
        -------
        response_dict : Dict
            Dictionary with decoder name -> activation metrics
        """
        responses = {}
        
        for decoder in self.decoders:
            activation = decoder.dynamic_activation(t, ca_trace)
            integrated = decoder.calculate_integrated_activation(t, ca_trace)
            peak = np.max(activation)
            mean = np.mean(activation)
            
            responses[decoder.params.name] = {
                'peak_activation': peak,
                'mean_activation': mean,
                'integrated_activation': integrated,
                'activation_trace': activation
            }
        
        return responses
    
    def get_response_matrix(self, t_list: List[np.ndarray], 
                           ca_traces: List[np.ndarray]) -> np.ndarray:
        """
        Get response matrix for multiple signatures.
        
        Parameters
        ----------
        t_list : List[np.ndarray]
            List of time arrays
        ca_traces : List[np.ndarray]
            List of Ca²⁺ traces
            
        Returns
        -------
        matrix : np.ndarray
            Matrix of shape [n_decoders, n_signatures]
            Each entry is the integrated activation
        """
        n_decoders = len(self.decoders)
        n_signatures = len(ca_traces)
        
        matrix = np.zeros((n_decoders, n_signatures))
        
        for i, decoder in enumerate(self.decoders):
            for j, (t, ca_trace) in enumerate(zip(t_list, ca_traces)):
                matrix[i, j] = decoder.calculate_integrated_activation(t, ca_trace)
        
        return matrix
    
    def plot_dose_response_curves(self, output_path: Optional[str] = None):
        """Plot Ca²⁺ dose-response curves for all decoders."""
        import matplotlib.pyplot as plt
        
        ca_range = np.logspace(1, 4, 100)  # 10 nM to 10 μM
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for decoder in self.decoders:
            activation = [decoder.steady_state_activation(ca) for ca in ca_range]
            label = f"{decoder.params.name} (Kd={decoder.params.K_d:.0f} nM)"
            ax.semilogx(ca_range, activation, label=label, linewidth=2)
        
        ax.set_xlabel('[Ca²⁺] (nM)', fontsize=12)
        ax.set_ylabel('Fractional Activation', fontsize=12)
        ax.set_title('CDPK Ca²⁺ Dose-Response Curves', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved dose-response curves to {output_path}")
        else:
            plt.show()


class PhosphorylationTarget:
    """Model of target protein phosphorylation by CDPKs."""
    
    def __init__(self, n_sites: int = 1, k_dephos: float = 0.1):
        """
        Parameters
        ----------
        n_sites : int
            Number of phosphorylation sites
        k_dephos : float
            Dephosphorylation rate by phosphatases (1/s)
        """
        self.n_sites = n_sites
        self.k_dephos = k_dephos
    
    def simulate_phosphorylation(self, t: np.ndarray, cdpk_activation: np.ndarray,
                                k_cat: float = 10.0) -> np.ndarray:
        """
        Simulate target phosphorylation dynamics.
        
        Parameters
        ----------
        t : np.ndarray
            Time points
        cdpk_activation : np.ndarray
            CDPK activation over time
        k_cat : float
            Catalytic rate constant (1/s)
            
        Returns
        -------
        phosphorylation : np.ndarray
            Fraction of phosphorylated target (0-1)
        """
        if self.n_sites == 1:
            # Single-site phosphorylation
            phos = np.zeros_like(t)
            phos[0] = 0.0
            
            for i in range(1, len(t)):
                dt = t[i] - t[i-1]
                k_phos = k_cat * cdpk_activation[i]
                
                dphos = k_phos * (1 - phos[i-1]) - self.k_dephos * phos[i-1]
                phos[i] = phos[i-1] + dphos * dt
                phos[i] = np.clip(phos[i], 0, 1)
            
            return phos
        
        else:
            # Multi-site phosphorylation (simplified)
            # This creates ultrasensitivity (Goldbeter-Koshland)
            phos = np.zeros_like(t)
            
            for i in range(len(t)):
                k_phos = k_cat * cdpk_activation[i]
                
                # Goldbeter-Koshland equation (approximate)
                if k_phos + self.k_dephos > 0:
                    phos[i] = k_phos / (k_phos + self.k_dephos)
                else:
                    phos[i] = 0
            
            # Apply Hill-like transformation for multisite
            phos = phos ** self.n_sites
            
            return phos


if __name__ == "__main__":
    # Test code
    import matplotlib.pyplot as plt
    
    # Create decoder panel
    panel = DecoderPanel()
    
    # Plot dose-response curves
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    panel.plot_dose_response_curves(str(output_dir / "cdpk_dose_response.png"))
    
    # Test with example Ca2+ trace
    t = np.linspace(0, 300, 3000)
    # Simulate oscillatory Ca2+ signal
    ca_baseline = 100
    ca_amplitude = 600
    frequency = 0.5  # Hz
    ca_trace = ca_baseline + ca_amplitude * (np.sin(2*np.pi*frequency*t) + 1) / 2
    
    # Analyze with panel
    responses = panel.analyze_signature(t, ca_trace)
    
    # Plot example responses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Ca2+ trace
    ax1.plot(t, ca_trace, 'b-', linewidth=2)
    ax1.set_ylabel('[Ca²⁺] (nM)', fontsize=12)
    ax1.set_title('Example Ca²⁺ Oscillations', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # CDPK activations
    for name in ['CPK28', 'CPK5', 'CPK1']:  # Low, medium, high Kd
        activation = responses[name]['activation_trace']
        ax2.plot(t, activation, linewidth=2, label=name)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('CDPK Activation', fontsize=12)
    ax2.set_title('Differential CDPK Responses', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "cdpk_example_response.png"), dpi=150)
    print(f"Saved example response to {output_dir / 'cdpk_example_response.png'}")
