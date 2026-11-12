"""
Signature Database Module
==========================

Manages Ca²⁺ signature data including loading, storage, analysis,
and feature extraction.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class CalciumSignature:
    """Data class representing a Ca²⁺ signature."""
    
    signature_id: str
    species: str
    cell_type: str
    stimulus: str
    stimulus_concentration: Optional[float] = None
    
    # Amplitude features (nM)
    baseline_ca: float = 100.0
    peak_amplitude: float = 800.0
    
    # Temporal features (seconds)
    rise_time: float = 15.0
    decay_tau: float = 120.0
    duration: float = 600.0
    latency: float = 5.0
    
    # Oscillation features
    oscillatory: bool = False
    frequency: Optional[float] = None  # Hz (events per second; 1 per min = 1/60 Hz)
    duty_cycle: Optional[float] = None
    spike_amplitude_cv: Optional[float] = None
    
    # Spatial features (if available)
    wave_velocity: Optional[float] = None  # μm/s
    spatial_range: Optional[float] = None  # μm
    
    # Time series data (if available)
    time_points: Optional[np.ndarray] = None
    ca_concentration: Optional[np.ndarray] = None
    
    # Metadata
    reference: Optional[str] = None
    notes: Optional[str] = None
    
    @property
    def fold_change(self) -> float:
        """Calculate fold change from baseline."""
        return self.peak_amplitude / self.baseline_ca
    
    @property
    def snr(self) -> float:
        """Estimate signal-to-noise ratio."""
        # Assume 10% baseline noise
        noise = self.baseline_ca * 0.1
        signal = self.peak_amplitude - self.baseline_ca
        return signal / noise if noise > 0 else np.inf
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = {
            'signature_id': self.signature_id,
            'species': self.species,
            'cell_type': self.cell_type,
            'stimulus': self.stimulus,
            'stimulus_concentration': self.stimulus_concentration,
            'baseline_ca': self.baseline_ca,
            'peak_amplitude': self.peak_amplitude,
            'rise_time': self.rise_time,
            'decay_tau': self.decay_tau,
            'duration': self.duration,
            'latency': self.latency,
            'oscillatory': self.oscillatory,
            'frequency': self.frequency,
            'duty_cycle': self.duty_cycle,
            'spike_amplitude_cv': self.spike_amplitude_cv,
            'wave_velocity': self.wave_velocity,
            'spatial_range': self.spatial_range,
            'reference': self.reference,
            'notes': self.notes
        }
        return d
    
    def get_feature_vector(self) -> np.ndarray:
        """Extract feature vector for machine learning."""
        features = [
            self.baseline_ca,
            self.peak_amplitude,
            self.fold_change,
            self.rise_time,
            self.decay_tau,
            self.duration,
            self.latency,
            self.frequency if self.frequency is not None else 0,
            self.duty_cycle if self.duty_cycle is not None else 0,
        ]
        return np.array(features)


class SignatureDatabase:
    """Manager for Ca²⁺ signature database."""
    
    def __init__(self):
        self.signatures: List[CalciumSignature] = []
        
    def add_signature(self, signature: CalciumSignature):
        """Add a signature to the database."""
        self.signatures.append(signature)
    
    def load_from_csv(self, filepath: str):
        """Load signatures from CSV file.

        Empty cells (and missing optional columns) become None, or the
        dataclass default for latency (5 s) and oscillatory (False).
        """
        df = pd.read_csv(filepath)

        def cell(row, key, default=None):
            value = row.get(key, default)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return default
            return value

        def number(row, key, default=None):
            value = cell(row, key, default)
            return float(value) if value is not None else None

        def flag(value):
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes")
            return bool(value)

        for _, row in df.iterrows():
            sig = CalciumSignature(
                signature_id=str(row['signature_id']),
                species=row['species'],
                cell_type=row['cell_type'],
                stimulus=row['stimulus'],
                stimulus_concentration=number(row, 'stimulus_concentration'),
                baseline_ca=float(row['baseline_ca']),
                peak_amplitude=float(row['peak_amplitude']),
                rise_time=float(row['rise_time']),
                decay_tau=float(row['decay_tau']),
                duration=float(row['duration']),
                latency=number(row, 'latency', 5.0),
                oscillatory=flag(cell(row, 'oscillatory', False)),
                frequency=number(row, 'frequency'),
                duty_cycle=number(row, 'duty_cycle'),
                spike_amplitude_cv=number(row, 'spike_amplitude_cv'),
                wave_velocity=number(row, 'wave_velocity'),
                spatial_range=number(row, 'spatial_range'),
                reference=cell(row, 'reference'),
                notes=cell(row, 'notes')
            )
            self.add_signature(sig)

    def save_to_csv(self, filepath: str):
        """Save signatures to CSV file."""
        data = [sig.to_dict() for sig in self.signatures]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert database to pandas DataFrame."""
        data = [sig.to_dict() for sig in self.signatures]
        return pd.DataFrame(data)
    
    def get_feature_matrix(self) -> np.ndarray:
        """Get feature matrix for all signatures."""
        return np.array([sig.get_feature_vector() for sig in self.signatures])
    
    def get_stimuli_list(self) -> List[str]:
        """Get unique list of stimuli."""
        return list(set(sig.stimulus for sig in self.signatures))
    
    def get_cell_types(self) -> List[str]:
        """Get unique list of cell types."""
        return list(set(sig.cell_type for sig in self.signatures))
    
    def filter_by_stimulus(self, stimulus: str) -> List[CalciumSignature]:
        """Filter signatures by stimulus type."""
        return [sig for sig in self.signatures if sig.stimulus == stimulus]
    
    def filter_by_cell_type(self, cell_type: str) -> List[CalciumSignature]:
        """Filter signatures by cell type."""
        return [sig for sig in self.signatures if sig.cell_type == cell_type]
    
    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics for the database."""
        df = self.to_dataframe()
        
        stats = {
            'total_signatures': len(self.signatures),
            'unique_stimuli': len(self.get_stimuli_list()),
            'unique_cell_types': len(self.get_cell_types()),
            'oscillatory_fraction': df['oscillatory'].mean(),
            'mean_peak_amplitude': df['peak_amplitude'].mean(),
            'mean_duration': df['duration'].mean(),
            'mean_frequency': df[df['frequency'].notna()]['frequency'].mean() if any(df['frequency'].notna()) else None,
        }
        
        return stats
    
    def __len__(self):
        return len(self.signatures)
    
    def __getitem__(self, idx):
        return self.signatures[idx]


def create_example_database() -> SignatureDatabase:
    """Create example database with literature-derived signatures."""
    db = SignatureDatabase()
    
    # Example signatures from literature
    examples = [
        # ABA-induced oscillations in guard cells
        CalciumSignature(
            signature_id="ABA_GC_001",
            species="Arabidopsis thaliana",
            cell_type="guard_cell",
            stimulus="ABA",
            stimulus_concentration=10e-6,  # 10 μM
            baseline_ca=100,
            peak_amplitude=800,
            rise_time=20,
            decay_tau=40,
            duration=300,
            latency=60,
            oscillatory=True,
            frequency=0.5 / 60,  # 0.5 per min (2 min period), in Hz
            duty_cycle=0.3,
            reference="Allen et al. 2001, Plant Cell"
        ),
        
        # NaCl-induced sustained plateau
        CalciumSignature(
            signature_id="NaCl_ROOT_001",
            species="Arabidopsis thaliana",
            cell_type="root_epidermal",
            stimulus="NaCl",
            stimulus_concentration=150e-3,  # 150 mM
            baseline_ca=100,
            peak_amplitude=600,
            rise_time=30,
            decay_tau=300,
            duration=600,
            latency=10,
            oscillatory=False,
            reference="Knight et al. 1997, Plant J"
        ),
        
        # Touch-induced rapid spike
        CalciumSignature(
            signature_id="TOUCH_LEAF_001",
            species="Arabidopsis thaliana",
            cell_type="leaf_mesophyll",
            stimulus="mechanical_touch",
            baseline_ca=100,
            peak_amplitude=1200,
            rise_time=2,
            decay_tau=10,
            duration=30,
            latency=0.5,
            oscillatory=False,
            reference="Legue et al. 1997, FEBS Lett"
        ),
        
        # Pathogen-induced biphasic
        CalciumSignature(
            signature_id="PAMP_MESO_001",
            species="Arabidopsis thaliana",
            cell_type="leaf_mesophyll",
            stimulus="flg22",
            stimulus_concentration=100e-9,  # 100 nM
            baseline_ca=100,
            peak_amplitude=900,
            rise_time=15,
            decay_tau=60,
            duration=400,
            latency=5,
            oscillatory=True,
            frequency=0.2 / 60,  # 0.2 per min (5 min period), in Hz
            reference="Ranf et al. 2011, Plant J"
        ),
        
        # Cold shock transient
        CalciumSignature(
            signature_id="COLD_ROOT_001",
            species="Arabidopsis thaliana",
            cell_type="root_hair",
            stimulus="cold_shock",
            baseline_ca=100,
            peak_amplitude=700,
            rise_time=5,
            decay_tau=120,
            duration=180,
            latency=2,
            oscillatory=False,
            reference="Knight et al. 1996, Plant Cell"
        ),
        
        # H2O2-induced oscillations
        CalciumSignature(
            signature_id="H2O2_GC_001",
            species="Arabidopsis thaliana",
            cell_type="guard_cell",
            stimulus="H2O2",
            stimulus_concentration=10e-6,  # 10 μM
            baseline_ca=100,
            peak_amplitude=650,
            rise_time=25,
            decay_tau=50,
            duration=400,
            latency=30,
            oscillatory=True,
            frequency=0.4 / 60,  # 0.4 per min (2.5 min period), in Hz
            duty_cycle=0.25,
            reference="Pei et al. 2000, Nature"
        ),
        
        # Glutamate-induced wave
        CalciumSignature(
            signature_id="GLU_ROOT_001",
            species="Arabidopsis thaliana",
            cell_type="root_epidermal",
            stimulus="glutamate",
            stimulus_concentration=1e-3,  # 1 mM
            baseline_ca=100,
            peak_amplitude=1000,
            rise_time=3,
            decay_tau=20,
            duration=60,
            latency=1,
            oscillatory=False,
            wave_velocity=400,  # μm/s
            spatial_range=1000,
            reference="Toyota et al. 2018, Science"
        ),
    ]
    
    for sig in examples:
        db.add_signature(sig)
    
    return db


if __name__ == "__main__":
    # Test code
    db = create_example_database()
    print(f"Created database with {len(db)} signatures")
    print("\nSummary statistics:")
    stats = db.get_summary_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save example database
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    db.save_to_csv(output_dir / "example_signatures.csv")
    print(f"\nSaved to {output_dir / 'example_signatures.csv'}")
