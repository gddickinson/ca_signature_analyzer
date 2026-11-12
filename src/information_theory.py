"""
Information Theory Module
==========================

Calculate information-theoretic metrics for Ca²⁺ signaling:
- Mutual information
- Channel capacity  
- Signature discriminability
- Information flow through pathway
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
import warnings


class InformationAnalyzer:
    """Calculate information-theoretic metrics for Ca²⁺ signaling."""
    
    @staticmethod
    def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate mutual information between discrete variables.
        
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Parameters
        ----------
        x, y : np.ndarray
            Discrete random variables (integer labels)
            
        Returns
        -------
        mi : float
            Mutual information in bits
        """
        # Ensure integer labels
        x = np.asarray(x, dtype=int)
        y = np.asarray(y, dtype=int)
        
        # Calculate using sklearn
        mi_nats = mutual_info_score(x, y)
        
        # Convert nats to bits
        mi_bits = mi_nats / np.log(2)
        
        return mi_bits
    
    @staticmethod
    def mutual_information_continuous(x: np.ndarray, y: np.ndarray, 
                                    n_bins: int = 10) -> float:
        """
        Calculate mutual information between continuous variables.
        
        Uses adaptive binning to discretize continuous variables.
        
        Parameters
        ----------
        x, y : np.ndarray
            Continuous random variables
        n_bins : int
            Number of bins for discretization
            
        Returns
        -------
        mi : float
            Mutual information in bits
        """
        # Discretize using uniform binning
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        
        x_binned = discretizer.fit_transform(x.reshape(-1, 1)).astype(int).flatten()
        y_binned = discretizer.fit_transform(y.reshape(-1, 1)).astype(int).flatten()
        
        # Calculate MI
        return InformationAnalyzer.mutual_information_discrete(x_binned, y_binned)
    
    @staticmethod
    def conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
        """
        Calculate conditional entropy H(Y|X).
        
        H(Y|X) = H(Y,X) - H(X)
        
        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        x : np.ndarray
            Conditioning variable
            
        Returns
        -------
        h_cond : float
            Conditional entropy in bits
        """
        # Joint entropy
        xy = np.column_stack([x, y])
        h_joint = InformationAnalyzer.entropy_discrete(xy)
        
        # Marginal entropy of X
        h_x = InformationAnalyzer.entropy_discrete(x)
        
        # Conditional entropy
        h_cond = h_joint - h_x
        
        return h_cond
    
    @staticmethod
    def entropy_discrete(x: np.ndarray) -> float:
        """
        Calculate Shannon entropy for discrete variable.
        
        H(X) = -Σ p(x) log₂ p(x)
        
        Parameters
        ----------
        x : np.ndarray
            Discrete random variable
            
        Returns
        -------
        h : float
            Entropy in bits
        """
        if x.ndim > 1:
            # For multivariate, convert to unique joint states
            x_tuple = [tuple(row) for row in x]
            unique, counts = np.unique(x_tuple, return_counts=True, axis=0)
        else:
            unique, counts = np.unique(x, return_counts=True)
        
        # Probabilities
        probs = counts / len(x)
        
        # Shannon entropy in bits
        h = entropy(probs, base=2)
        
        return h
    
    @staticmethod
    def channel_capacity_shannon(bandwidth: float, snr: float) -> float:
        """
        Calculate theoretical channel capacity (Shannon-Hartley theorem).
        
        C = W × log₂(1 + SNR)
        
        where:
            C = channel capacity (bits/s)
            W = bandwidth (Hz)
            SNR = signal-to-noise power ratio (linear, not dB)
        
        Parameters
        ----------
        bandwidth : float
            Signal bandwidth in Hz
        snr : float
            Signal-to-noise power ratio (linear)
            
        Returns
        -------
        capacity : float
            Channel capacity in bits/s
        """
        capacity = bandwidth * np.log2(1 + snr)
        return capacity
    
    @staticmethod
    def estimate_signature_capacity(signatures: np.ndarray, 
                                   labels: np.ndarray) -> Dict:
        """
        Estimate information capacity of Ca²⁺ signature system.
        
        Parameters
        ----------
        signatures : np.ndarray
            Feature matrix [n_samples, n_features]
        labels : np.ndarray
            Stimulus labels [n_samples]
            
        Returns
        -------
        metrics : Dict
            Dictionary of information metrics
        """
        classes, counts = np.unique(labels, return_counts=True)
        n_stimuli = len(classes)
        if counts.min() < 2:
            raise ValueError("estimate_signature_capacity needs at least 2 signatures per "
                             "stimulus (k-nearest-neighbour MI estimator)")
        
        # Maximum possible information (uniform distribution)
        max_information = np.log2(n_stimuli)
        
        # Actual mutual information
        mi_values = []
        for feature_idx in range(signatures.shape[1]):
            mi = mutual_info_classif(
                signatures[:, feature_idx].reshape(-1, 1),
                labels,
                random_state=42
            )[0]
            mi_values.append(mi / np.log(2))  # Convert to bits
        
        # Average MI across features
        avg_mi = np.mean(mi_values)
        
        # Information efficiency
        efficiency = avg_mi / max_information if max_information > 0 else 0
        
        metrics = {
            'n_stimuli': n_stimuli,
            'max_information_bits': max_information,
            'avg_mutual_information_bits': avg_mi,
            'max_mutual_information_bits': np.max(mi_values),
            'information_efficiency': efficiency,
            'mi_per_feature': mi_values
        }
        
        return metrics


class PathwayInformationFlow:
    """Analyze information flow through Ca²⁺ signaling pathway."""
    
    def __init__(self):
        self.stages = []
        self.stage_names = []
        
    def add_stage(self, name: str, data: np.ndarray):
        """
        Add a stage to the information flow analysis.
        
        Parameters
        ----------
        name : str
            Stage name (e.g., 'Stimulus', 'Ca_signature', 'CDPK', 'Output')
        data : np.ndarray
            Data for this stage [n_samples, n_features]; non-numeric data
            (labels) is converted to integer codes
        """
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.number):
            # Categorical stage (e.g. stimulus names): sorted integer codes
            _, codes = np.unique(data, return_inverse=True)
            data = codes.reshape(data.shape)
        self.stage_names.append(name)
        self.stages.append(data)
    
    def calculate_flow(self, discretize: bool = True, 
                      n_bins: int = 10) -> Dict:
        """
        Calculate information flow between consecutive stages.
        
        Returns
        -------
        flow_dict : Dict
            Information flow metrics
        """
        if len(self.stages) < 2:
            raise ValueError("Need at least 2 stages for flow analysis")
        
        flow = {}
        
        for i in range(len(self.stages) - 1):
            stage1_name = self.stage_names[i]
            stage2_name = self.stage_names[i + 1]
            
            data1 = self.stages[i]
            data2 = self.stages[i + 1]
            
            # Ensure 1D for MI calculation
            if data1.ndim > 1 and data1.shape[1] > 1:
                # Use PCA to reduce to 1D
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                data1 = pca.fit_transform(data1).flatten()
            else:
                data1 = data1.flatten()
            
            if data2.ndim > 1 and data2.shape[1] > 1:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                data2 = pca.fit_transform(data2).flatten()
            else:
                data2 = data2.flatten()
            
            # Calculate MI
            if discretize:
                mi = InformationAnalyzer.mutual_information_continuous(
                    data1, data2, n_bins=n_bins
                )
            else:
                # Use sklearn's continuous MI estimator
                mi_nats = mutual_info_regression(
                    data1.reshape(-1, 1), 
                    data2,
                    random_state=42
                )[0]
                mi = mi_nats / np.log(2)
            
            flow[f"{stage1_name} → {stage2_name}"] = mi
        
        # Calculate total information preservation
        if len(flow) > 0:
            # Information at start
            h_start = InformationAnalyzer.entropy_discrete(
                KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
                .fit_transform(self.stages[0].reshape(-1, 1))
                .astype(int).flatten()
            )
            
            # Information preserved at end
            mi_total = list(flow.values())[-1]
            
            flow['information_preservation'] = mi_total / h_start if h_start > 0 else 0
            flow['initial_entropy'] = h_start
            flow['final_mutual_information'] = mi_total
        
        return flow


# SignatureDiscriminability lives in discriminability.py; re-exported here.
try:
    from .discriminability import SignatureDiscriminability  # noqa: F401
except ImportError:
    from discriminability import SignatureDiscriminability  # noqa: F401


if __name__ == "__main__":
    # Test code
    print("Testing Information Theory Module\n")
    
    # Test 1: Discrete mutual information
    print("Test 1: Discrete MI")
    x = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    mi = InformationAnalyzer.mutual_information_discrete(x, y)
    print(f"  MI(X;Y) = {mi:.3f} bits\n")
    
    # Test 2: Continuous mutual information
    print("Test 2: Continuous MI")
    x = np.random.randn(100)
    y = x + 0.5 * np.random.randn(100)  # Correlated
    mi = InformationAnalyzer.mutual_information_continuous(x, y, n_bins=10)
    print(f"  MI(X;Y) = {mi:.3f} bits\n")
    
    # Test 3: Channel capacity
    print("Test 3: Shannon-Hartley Channel Capacity")
    bandwidth = 1.0  # 1 Hz (typical Ca2+ oscillation)
    snr = 10  # 10:1 signal-to-noise
    capacity = InformationAnalyzer.channel_capacity_shannon(bandwidth, snr)
    print(f"  Capacity = {capacity:.3f} bits/s")
    print(f"  (Bandwidth = {bandwidth} Hz, SNR = {snr})\n")
    
    # Test 4: Signature discriminability
    print("Test 4: Signature Classification")
    n_samples = 50
    n_classes = 5
    n_features = 9
    
    # Generate synthetic signature data
    signatures = []
    labels = []
    for i in range(n_classes):
        # Each class has different mean
        class_data = np.random.randn(n_samples, n_features) + i * 2
        signatures.append(class_data)
        labels.extend([i] * n_samples)
    
    signatures = np.vstack(signatures)
    labels = np.array(labels)
    
    # Test classification
    results = SignatureDiscriminability.classification_accuracy(
        signatures, labels, classifier='random_forest'
    )
    
    print(f"  Classification Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"  Information Transmitted: {results['information_bits']:.3f} / {results['max_information_bits']:.3f} bits")
    print(f"  ({results['n_samples']} samples, {results['n_classes']} classes)")
