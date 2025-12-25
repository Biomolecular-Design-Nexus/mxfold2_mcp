#!/usr/bin/env python3
"""
UC-002: Deep Learning Model Comparison

This script demonstrates comparing different MXfold2 models on the same
RNA sequences, showcasing the power of deep learning with thermodynamic
integration versus traditional approaches.

Usage:
    python examples/use_case_2_model_comparison.py [--input INPUT_FASTA] [--models MODEL1,MODEL2,...]

Example:
    python examples/use_case_2_model_comparison.py --input examples/data/sample_rna.fa --models Turner,Mix,MixC
"""

import argparse
import os
import sys
import time
from pathlib import Path
import csv

# MXfold2 should be installed in the environment
# No need for local path manipulation

def run_model_comparison(input_file, models, output_file=None):
    """
    Compare multiple MXfold2 models on the same RNA sequences

    Args:
        input_file (str): Path to FASTA file with RNA sequences
        models (list): List of model names to compare
        output_file (str): Optional CSV file to save comparison results

    Returns:
        dict: Comparison results and timing data
    """
    results = []

    try:
        from mxfold2.predict import Predict
        from mxfold2.dataset import FastaDataset
        from torch.utils.data import DataLoader
        import torch

        print(f"MXfold2 Model Comparison")
        print(f"Input: {input_file}")
        print(f"Models: {', '.join(models)}")
        print("=" * 60)

        # Load sequences once
        test_dataset = FastaDataset(input_file)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for model_name in models:
            print(f"\n🧬 Running model: {model_name}")
            start_time = time.time()

            try:
                # Create argument namespace
                class Args:
                    def __init__(self, model_type):
                        self.input = input_file
                        self.model = model_type
                        self.param = ""
                        self.seed = 0
                        self.gpu = -1  # CPU only for compatibility
                        self.use_constraint = False
                        self.result = None
                        self.bpseq = None
                        self.bpp = None

                        # Default hyperparameters
                        self.max_helix_length = 30
                        self.embed_size = 0
                        self.num_filters = None
                        self.filter_size = None
                        self.pool_size = None
                        self.dilation = 0
                        self.num_lstm_layers = 0
                        self.num_lstm_units = 0
                        self.num_transformer_layers = 0
                        self.num_transformer_hidden_units = 2048
                        self.num_transformer_att = 8
                        self.num_hidden_units = None
                        self.num_paired_filters = []
                        self.paired_filter_size = []
                        self.dropout_rate = 0.0
                        self.fc_dropout_rate = 0.0
                        self.num_att = 0
                        self.pair_join = "cat"
                        self.no_split_lr = False

                args = Args(model_name)
                predictor = Predict()
                predictor.test_loader = test_loader

                # Build model
                model, config = predictor.build_model(args)
                predictor.model = model

                # Load default parameters if needed
                if args.param == '' and model_name != 'Turner':
                    # For neural models, try to load default parameters
                    try:
                        from mxfold2 import default_conf
                        if os.path.exists(default_conf):
                            param_path = Path(default_conf).parent / "TrainSetAB.pth"
                            if param_path.exists():
                                p = torch.load(param_path, map_location='cpu')
                                if isinstance(p, dict) and 'model_state_dict' in p:
                                    p = p['model_state_dict']
                                model.load_state_dict(p, strict=False)
                                print(f"  ✓ Loaded pre-trained parameters")
                    except Exception as e:
                        print(f"  ⚠ Could not load pre-trained parameters: {e}")

                # Predict structures (capture output)
                seq_count = 0
                predictions = []

                model.eval()
                with torch.no_grad():
                    for headers, seqs, refs in test_loader:
                        seq_count += len(headers)
                        if model_name == 'Turner':
                            # Turner model (thermodynamic only)
                            scs, preds, bps = model(seqs)
                        else:
                            # Neural models
                            try:
                                scs, preds, bps = model(seqs)
                            except Exception as e:
                                print(f"  ⚠ Model prediction failed: {e}")
                                scs = [0.0] * len(headers)
                                preds = ["." * len(seq) for seq in seqs]
                                bps = [None] * len(headers)

                        for header, seq, sc, pred in zip(headers, seqs, scs, preds):
                            predictions.append({
                                "sequence_id": header,
                                "sequence": seq,
                                "structure": pred,
                                "score": float(sc) if hasattr(sc, 'item') else float(sc)
                            })
                            print(f"  {header}: {pred} ({sc:.1f})")

                elapsed_time = time.time() - start_time

                result = {
                    "model": model_name,
                    "sequences_processed": seq_count,
                    "runtime_seconds": elapsed_time,
                    "predictions": predictions,
                    "status": "success"
                }

                print(f"  ✓ Completed in {elapsed_time:.2f}s ({seq_count} sequences)")

            except Exception as e:
                elapsed_time = time.time() - start_time
                result = {
                    "model": model_name,
                    "sequences_processed": 0,
                    "runtime_seconds": elapsed_time,
                    "predictions": [],
                    "status": "error",
                    "error": str(e)
                }
                print(f"  ✗ Failed: {e}")

            results.append(result)

        # Save comparison results
        if output_file:
            save_comparison_csv(results, output_file)
            print(f"\n📊 Comparison results saved to: {output_file}")

        # Print summary
        print(f"\n📋 SUMMARY")
        print("-" * 40)
        for result in results:
            status = "✓" if result["status"] == "success" else "✗"
            print(f"{status} {result['model']:8s}: {result['runtime_seconds']:6.2f}s ({result['sequences_processed']} seqs)")

        return {"status": "success", "results": results}

    except ImportError as e:
        return {
            "status": "error",
            "error": f"MXfold2 not available: {e}",
            "suggestion": "Install mxfold2 in the legacy environment"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def save_comparison_csv(results, output_file):
    """Save model comparison results to CSV"""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'sequence_id', 'sequence_length', 'structure', 'score', 'runtime_seconds', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            if result['status'] == 'success':
                for pred in result['predictions']:
                    writer.writerow({
                        'model': result['model'],
                        'sequence_id': pred['sequence_id'],
                        'sequence_length': len(pred['sequence']),
                        'structure': pred['structure'],
                        'score': pred['score'],
                        'runtime_seconds': result['runtime_seconds'],
                        'status': result['status']
                    })
            else:
                writer.writerow({
                    'model': result['model'],
                    'sequence_id': 'N/A',
                    'sequence_length': 0,
                    'structure': 'N/A',
                    'score': 0,
                    'runtime_seconds': result['runtime_seconds'],
                    'status': result['status']
                })

def main():
    parser = argparse.ArgumentParser(
        description="Compare different MXfold2 models on RNA sequences"
    )
    parser.add_argument(
        "--input", "-i",
        default="examples/data/test_short.fa",
        help="Input FASTA file with RNA sequences"
    )
    parser.add_argument(
        "--models", "-m",
        default="Turner,Mix",
        help="Comma-separated list of models to compare"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV file for detailed results"
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1

    # Parse models
    models = [m.strip() for m in args.models.split(',')]
    valid_models = ["Turner", "Mix", "MixC", "Zuker", "ZukerC", "ZukerL", "ZukerS"]

    for model in models:
        if model not in valid_models:
            print(f"Error: Invalid model '{model}'. Valid models: {', '.join(valid_models)}")
            return 1

    # Run comparison
    result = run_model_comparison(args.input, models, args.output)

    if result["status"] == "success":
        print(f"\n✓ Model comparison completed successfully!")
        return 0
    else:
        print(f"\n✗ Model comparison failed: {result['error']}")
        return 1

if __name__ == "__main__":
    exit(main())