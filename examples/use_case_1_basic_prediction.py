#!/usr/bin/env python3
"""
UC-001: Basic RNA Secondary Structure Prediction

This script demonstrates the core functionality of MXfold2:
predicting RNA secondary structure from FASTA sequences using
pre-trained neural network models with thermodynamic integration.

Usage:
    python examples/use_case_1_basic_prediction.py [--input INPUT_FASTA] [--model MODEL_TYPE]

Example:
    python examples/use_case_1_basic_prediction.py --input examples/data/sample_rna.fa --model MixC
"""

import argparse
import os
import sys
from pathlib import Path

# MXfold2 should be installed in the environment
# No need for local path manipulation

def predict_rna_structure(input_file, model_type="Turner", output_dir=None, gpu=-1):
    """
    Predict RNA secondary structure using MXfold2

    Args:
        input_file (str): Path to FASTA file with RNA sequences
        model_type (str): Model type ('Turner', 'Mix', 'MixC', 'Zuker', etc.)
        output_dir (str): Optional output directory for detailed results
        gpu (int): GPU device ID (-1 for CPU)

    Returns:
        dict: Prediction results and metadata
    """
    try:
        from mxfold2.predict import Predict
        import torch

        # Create argument namespace similar to CLI
        class Args:
            def __init__(self):
                self.input = input_file
                self.model = model_type
                self.param = ""  # Use default parameters
                self.seed = 0
                self.gpu = gpu
                self.use_constraint = False
                self.result = None
                self.bpseq = output_dir if output_dir else None
                self.bpp = None

                # Model hyperparameters (defaults)
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

        args = Args()
        predictor = Predict()

        print(f"MXfold2 RNA Secondary Structure Prediction")
        print(f"Input: {input_file}")
        print(f"Model: {model_type}")
        print(f"Device: {'GPU' if gpu >= 0 else 'CPU'}")
        print("-" * 50)

        # Run prediction
        predictor.run(args)

        return {
            "status": "success",
            "input_file": input_file,
            "model_type": model_type,
            "output_dir": output_dir,
            "device": "GPU" if gpu >= 0 else "CPU"
        }

    except ImportError as e:
        return {
            "status": "error",
            "error": f"MXfold2 not installed or not in path: {e}",
            "suggestion": "Please install mxfold2 in the legacy environment (env_py37)"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Check input file format and model parameters"
        }

def main():
    parser = argparse.ArgumentParser(
        description="Basic RNA secondary structure prediction using MXfold2"
    )
    parser.add_argument(
        "--input", "-i",
        default="examples/data/test_short.fa",
        help="Input FASTA file with RNA sequences"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["Turner", "Mix", "MixC", "Zuker", "ZukerC", "ZukerL", "ZukerS"],
        default="Turner",
        help="Model type for prediction"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for detailed results (BPSEQ format)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device ID (-1 for CPU)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        print(f"Try using: examples/data/test_short.fa")
        return 1

    # Run prediction
    result = predict_rna_structure(
        input_file=args.input,
        model_type=args.model,
        output_dir=args.output,
        gpu=args.gpu
    )

    if args.verbose:
        print(f"\nResult: {result}")

    if result["status"] == "success":
        print(f"\n✓ Prediction completed successfully!")
        if args.output:
            print(f"  Detailed results saved to: {args.output}")
        return 0
    else:
        print(f"\n✗ Prediction failed: {result['error']}")
        if "suggestion" in result:
            print(f"  Suggestion: {result['suggestion']}")
        return 1

if __name__ == "__main__":
    exit(main())