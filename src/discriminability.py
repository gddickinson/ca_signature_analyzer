"""
Signature Discriminability Module
=================================

How well Ca²⁺ signature classes can be told apart: a standardized distance
between class means, and cross-validated classification accuracy.
(Split out of information_theory.py, which re-exports it.)
"""

import warnings
from typing import Dict

import numpy as np


class SignatureDiscriminability:
    """Analyze discriminability of Ca²⁺ signatures."""
    
    @staticmethod
    def pairwise_discrimination(signatures: np.ndarray, 
                               labels: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise discrimination between signature classes.
        
        Distance between class means, each feature scaled by the mean of the
        two classes' standard deviations: ||(mu1 - mu2) / ((sd1 + sd2) / 2)||.
        (Not a Bhattacharyya distance; classes with one sample have sd 0, so
        their distances are scaled by 1e-10 and become very large.)
        
        Parameters
        ----------
        signatures : np.ndarray
            Feature matrix [n_samples, n_features]
        labels : np.ndarray
            Class labels
            
        Returns
        -------
        discrimination_matrix : np.ndarray
            Pairwise discrimination scores [n_classes, n_classes]
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        discrimination = np.zeros((n_classes, n_classes))
        
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i == j:
                    discrimination[i, j] = 0
                    continue
                
                # Get samples for each class
                samples1 = signatures[labels == label1]
                samples2 = signatures[labels == label2]
                
                # Calculate Bhattacharyya distance (simplified)
                # Using Euclidean distance between means normalized by pooled std
                mean1 = np.mean(samples1, axis=0)
                mean2 = np.mean(samples2, axis=0)
                std1 = np.std(samples1, axis=0) + 1e-10
                std2 = np.std(samples2, axis=0) + 1e-10
                
                # Mahalanobis-like distance
                pooled_std = (std1 + std2) / 2
                distance = np.linalg.norm((mean1 - mean2) / pooled_std)
                
                discrimination[i, j] = distance
        
        return discrimination
    
    @staticmethod
    def classification_accuracy(signatures: np.ndarray, 
                               labels: np.ndarray,
                               classifier: str = 'random_forest') -> Dict:
        """
        Test signature discriminability using classification.
        
        Parameters
        ----------
        signatures : np.ndarray
            Feature matrix
        labels : np.ndarray
            Class labels
        classifier : str
            Classifier type ('random_forest', 'svm', 'logistic')
            
        Returns
        -------
        results : Dict
            Classification results
        """
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Select classifier
        if classifier == 'random_forest':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier == 'svm':
            clf = SVC(kernel='rbf', random_state=42)
        elif classifier == 'logistic':
            clf = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        # Cross-validation
        cv_scores = cross_val_score(clf, signatures, labels, cv=5)
        
        # Get predictions for confusion matrix
        predictions = cross_val_predict(clf, signatures, labels, cv=5)
        
        # Confusion matrix
        conf_mat = confusion_matrix(labels, predictions)
        
        # Classification report
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_report(labels, predictions, output_dict=True)
        
        results = {
            'accuracy_mean': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'cv_scores': cv_scores,
            'confusion_matrix': conf_mat,
            'classification_report': report,
            'n_classes': len(np.unique(labels)),
            'n_samples': len(labels)
        }
        
        # Calculate information from accuracy
        # For N equally likely classes, uniform guessing gives 1/N accuracy
        # Information = log2(N) × accuracy (simplified)
        n_classes = len(np.unique(labels))
        max_info = np.log2(n_classes)
        results['information_bits'] = max_info * results['accuracy_mean']
        results['max_information_bits'] = max_info
        
        return results
