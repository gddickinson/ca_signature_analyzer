#!/usr/bin/env python3
"""
Standalone script for analyzing Ca²⁺ signature database.

Usage:
    python analyze_database.py --input signatures.csv --output analysis_results/
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from signature_database import SignatureDatabase, create_example_database
from biophysical_model import CalciumDynamicsModel
from decoder_model import DecoderPanel
from information_theory import (InformationAnalyzer, PathwayInformationFlow,
                                SignatureDiscriminability)
from visualization import SignaturePlotter, IntegratedReport
import utils


def encode_stimuli(labels):
    """Sorted stimulus names and integer codes (sorted order = sklearn's confusion-matrix order)."""
    names = sorted(set(labels))
    return names, np.array([names.index(s) for s in labels])


def main():
    parser = argparse.ArgumentParser(description='Analyze Ca²⁺ signature database')
    
    parser.add_argument('--input', type=str, default=None,
                       help='Input CSV file (if None, uses example database)')
    parser.add_argument('--output', type=str, default='outputs/analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    utils.ensure_directory(output_dir)
    
    # Load database
    if args.input:
        print(f"Loading database from {args.input}...")
        db = SignatureDatabase()
        db.load_from_csv(args.input)
    else:
        print("Using example database...")
        db = create_example_database()
    
    print(f"  Loaded {len(db)} signatures")
    
    if len(db) == 0:
        print("Error: Database is empty")
        return
    
    # Show statistics
    print("\nDatabase Statistics:")
    stats = db.get_summary_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Initialize models
    model = CalciumDynamicsModel()
    decoder_panel = DecoderPanel()
    
    # Analyze each signature
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # 1. Simulate signatures and collect data
    print("\n1. Simulating Ca²⁺ dynamics for each signature...")
    
    t_list = []
    ca_list = []
    labels = []
    decoder_responses_all = []
    
    tracker = utils.ProgressTracker(len(db), "Simulating")
    for i, sig in enumerate(db.signatures):
        # Simulate
        t, sol = model.simulate(sig.stimulus, duration=min(sig.duration, 300), intensity=1.0)
        ca_cyt = sol[:, 0]
        
        # Store
        t_list.append(t)
        ca_list.append(ca_cyt)
        labels.append(sig.stimulus)
        
        # Analyze decoders
        responses = decoder_panel.analyze_signature(t, ca_cyt)
        integrated = [r['integrated_activation'] for r in responses.values()]
        decoder_responses_all.append(integrated)
        
        tracker.update(1)
    
    decoder_responses_all = np.array(decoder_responses_all)
    
    # 2. Generate visualizations
    print("\n2. Generating visualizations...")
    
    # Signature comparison
    print("  - Signature comparison plot")
    SignaturePlotter.plot_signature_comparison(
        t_list[:5],  # First 5
        ca_list[:5],
        [f"{db.signatures[i].signature_id}" for i in range(min(5, len(db)))],
        output_path=str(output_dir / "signature_comparison.png"),
        title="Ca²⁺ Signature Comparison"
    )
    
    # Decoder heatmap
    print("  - Decoder response heatmap")
    SignaturePlotter.plot_decoder_heatmap(
        decoder_responses_all.T,
        decoder_panel.names,
        labels,
        output_path=str(output_dir / "decoder_heatmap.png")
    )
    
    # UMAP projection
    if len(db) >= 5:
        print("  - UMAP projection")
        features = db.get_feature_matrix()
        SignaturePlotter.plot_umap_projection(
            features,
            np.array(labels),
            output_path=str(output_dir / "umap_projection.png")
        )
    
    # 3. Information theory analysis
    print("\n3. Information theory analysis...")
    
    if len(db) >= 5:
        features = db.get_feature_matrix()
        stimuli_array = np.array(labels)
        
        # Encode stimuli (sorted, the order of the confusion-matrix rows)
        unique_stimuli, labels_encoded = encode_stimuli(labels)
        
        # Calculate metrics
        print("  - Calculating mutual information")
        try:
            metrics = InformationAnalyzer.estimate_signature_capacity(features, labels_encoded)
            print("\n  Information Metrics:")
            print(f"    Max possible information: {metrics['max_information_bits']:.3f} bits")
            print(f"    Avg mutual information: {metrics['avg_mutual_information_bits']:.3f} bits")
            print(f"    Information efficiency: {metrics['information_efficiency']:.2%}")
        except ValueError as e:
            metrics = {}
            print(f"    Skipped: {e}")
        
        # Classification accuracy
        if len(db) >= 10:
            print("\n  - Testing classification accuracy")
            results = SignatureDiscriminability.classification_accuracy(
                features, stimuli_array, classifier='random_forest'
            )
            
            print(f"    Classification accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
            print(f"    Information transmitted: {results['information_bits']:.3f} bits")
            
            # Save confusion matrix
            SignaturePlotter.plot_confusion_matrix(
                results['confusion_matrix'],
                unique_stimuli,
                output_path=str(output_dir / "confusion_matrix.png")
            )
    
    # 4. Information flow
    if len(db) >= 5:
        print("\n4. Analyzing information flow...")
        
        pathway = PathwayInformationFlow()
        pathway.add_stage("Stimulus", np.array(labels).reshape(-1, 1))
        pathway.add_stage("Ca_Signature", features)
        pathway.add_stage("CDPK_Response", decoder_responses_all)
        
        flow = pathway.calculate_flow()
        
        print("\n  Information Flow:")
        for key, value in flow.items():
            print(f"    {key}: {value:.3f} bits")
        
        # Plot
        SignaturePlotter.plot_information_flow(
            flow,
            output_path=str(output_dir / "information_flow.png")
        )
    
    # 5. Save summary
    print("\n5. Saving analysis summary...")
    
    summary = {
        'Database Statistics': stats,
        'Information Metrics': metrics if len(db) >= 5 else {},
        'Classification Results': {
            'accuracy': results['accuracy_mean'],
            'n_classes': results['n_classes']
        } if len(db) >= 10 else {},
        'Information Flow': flow if len(db) >= 5 else {}
    }
    
    utils.save_json(summary, str(output_dir / "analysis_summary.json"))
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
