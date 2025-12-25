#!/usr/bin/env python3
"""
Script: model_comparison.py
Description: Compare multiple MXfold2 models on RNA sequences

Original Use Case: examples/use_case_2_model_comparison.py
Dependencies Removed: Inlined model comparison logic, simplified CSV handling

Usage:
    python scripts/model_comparison.py --input <input_file> --output <output_file>

Example:
    python scripts/model_comparison.py --input examples/data/sample.fa --output results/comparison.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import time

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parent))
from lib.io import load_fasta_simple, save_json, save_csv
from lib.utils import check_mxfold2_available, create_mxfold2_args, timing_context, ensure_directory

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "models": ["Turner", "Mix"],
    "gpu": -1,
    "seed": 0,
    "verbose": True,
    "include_timing": True
}

VALID_MODELS = ["Turner", "Mix", "MixC", "Zuker", "ZukerC", "ZukerL", "ZukerS"]

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_model_comparison(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare multiple MXfold2 models on RNA sequences.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_file: Path to save comparison results (CSV format)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: List of model results
            - comparison: Detailed comparison data
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_model_comparison("input.fa", models=["Turner", "Mix"])
        >>> print(result['results'][0]['model'])
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Validate models
    models = config["models"]
    if isinstance(models, str):
        models = [m.strip() for m in models.split(',')]

    for model in models:
        if model not in VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Valid models: {VALID_MODELS}")

    # Check MXfold2 availability
    mxfold2_status = check_mxfold2_available()
    if not mxfold2_status["available"]:
        return {
            "status": "error",
            "error": f"MXfold2 not available: {mxfold2_status['error']}",
            "suggestion": mxfold2_status.get("suggestion", "Install mxfold2")
        }

    # Import MXfold2 (lazy loading)
    try:
        from mxfold2.predict import Predict
        from mxfold2.dataset import FastaDataset
        from torch.utils.data import DataLoader
        import torch
    except ImportError as e:
        return {
            "status": "error",
            "error": f"Failed to import MXfold2 components: {e}"
        }

    if config.get("verbose"):
        print(f"MXfold2 Model Comparison")
        print(f"Input: {input_file}")
        print(f"Models: {', '.join(models)}")
        print("=" * 60)

    # Load sequences once
    test_dataset = FastaDataset(str(input_file))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []
    comparison_data = []

    for model_name in models:
        if config.get("verbose"):
            print(f"\n🧬 Running model: {model_name}")

        with timing_context(f"model_{model_name}") as timer:
            try:
                # Create predictor args
                args = create_mxfold2_args(
                    model_type=model_name,
                    input_file=str(input_file),
                    gpu=config["gpu"],
                    seed=config["seed"]
                )

                # Initialize predictor
                predictor = Predict()
                predictor.test_loader = test_loader

                # Build model
                model, model_config = predictor.build_model(args)
                predictor.model = model

                # Load pre-trained parameters if available
                if model_name != "Turner" and args.param == "":
                    try:
                        from mxfold2 import default_conf
                        if hasattr(default_conf, '__file__'):
                            param_path = Path(default_conf.__file__).parent / "TrainSetAB.pth"
                            if param_path.exists():
                                p = torch.load(param_path, map_location='cpu')
                                if isinstance(p, dict) and 'model_state_dict' in p:
                                    p = p['model_state_dict']
                                model.load_state_dict(p, strict=False)
                                if config.get("verbose"):
                                    print(f"  ✓ Loaded pre-trained parameters")
                    except Exception as e:
                        if config.get("verbose"):
                            print(f"  ⚠ Could not load pre-trained parameters: {e}")

                # Run predictions
                predictions = []
                seq_count = 0

                model.eval()
                with torch.no_grad():
                    for headers, seqs, refs in test_loader:
                        seq_count += len(headers)
                        try:
                            scs, preds, bps = model(seqs)

                            for header, seq, sc, pred in zip(headers, seqs, scs, preds):
                                prediction = {
                                    "sequence_id": header,
                                    "sequence": seq,
                                    "structure": pred,
                                    "score": float(sc) if hasattr(sc, 'item') else float(sc),
                                    "length": len(seq)
                                }
                                predictions.append(prediction)

                                # Add to comparison data
                                comparison_data.append({
                                    "model": model_name,
                                    "sequence_id": header,
                                    "sequence_length": len(seq),
                                    "structure": pred,
                                    "score": float(sc) if hasattr(sc, 'item') else float(sc),
                                    "runtime_seconds": timer.elapsed if timer.elapsed else 0,
                                    "status": "success"
                                })

                                if config.get("verbose"):
                                    print(f"  {header}: {pred} ({sc:.2f})")

                        except Exception as e:
                            # Handle prediction errors gracefully
                            for header, seq in zip(headers, seqs):
                                prediction = {
                                    "sequence_id": header,
                                    "sequence": seq,
                                    "structure": "." * len(seq),
                                    "score": 0.0,
                                    "length": len(seq),
                                    "error": str(e)
                                }
                                predictions.append(prediction)

                                # Add error to comparison data
                                comparison_data.append({
                                    "model": model_name,
                                    "sequence_id": header,
                                    "sequence_length": len(seq),
                                    "structure": "." * len(seq),
                                    "score": 0.0,
                                    "runtime_seconds": timer.elapsed if timer.elapsed else 0,
                                    "status": "error",
                                    "error": str(e)
                                })

                                if config.get("verbose"):
                                    print(f"  ✗ {header}: Prediction failed - {e}")

                result = {
                    "model": model_name,
                    "sequences_processed": seq_count,
                    "runtime_seconds": timer.elapsed,
                    "predictions": predictions,
                    "status": "success"
                }

                if config.get("verbose"):
                    print(f"  ✓ Completed in {timer.elapsed:.3f}s ({seq_count} sequences)")

            except Exception as e:
                result = {
                    "model": model_name,
                    "sequences_processed": 0,
                    "runtime_seconds": timer.elapsed if timer.elapsed else 0,
                    "predictions": [],
                    "status": "error",
                    "error": str(e)
                }
                if config.get("verbose"):
                    print(f"  ✗ Failed: {e}")

            results.append(result)

    # Save comparison results
    output_path = None
    if output_file:
        output_path = Path(output_file)
        ensure_directory(output_path.parent)

        if str(output_path).endswith('.csv'):
            save_csv(comparison_data, output_path)
        else:
            # Default to JSON
            save_json({
                "results": results,
                "comparison": comparison_data,
                "metadata": config
            }, output_path)

        if config.get("verbose"):
            print(f"\n📊 Comparison results saved to: {output_path}")

    # Print summary
    if config.get("verbose"):
        print(f"\n📋 SUMMARY")
        print("-" * 40)
        total_time = 0
        for result in results:
            status = "✓" if result["status"] == "success" else "✗"
            runtime = result["runtime_seconds"]
            total_time += runtime
            print(f"{status} {result['model']:8s}: {runtime:6.3f}s ({result['sequences_processed']} seqs)")
        print(f"Total time: {total_time:.3f}s")

    return {
        "status": "success",
        "results": results,
        "comparison": comparison_data,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "total_sequences": len(comparison_data) // len(models) if comparison_data else 0,
            "total_runtime": sum(r["runtime_seconds"] for r in results)
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--output', '-o', help='Output file (.csv or .json)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--models', '-m', default='Turner,Mix',
                       help='Comma-separated list of models to compare')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device (-1 for CPU)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Parse models
    models = [m.strip() for m in args.models.split(',')]

    # Override config with CLI args
    overrides = {
        "models": models
    }
    if args.gpu != -1:
        overrides["gpu"] = args.gpu
    if args.verbose:
        overrides["verbose"] = True

    # Run
    try:
        result = run_model_comparison(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        if result["status"] == "success":
            successful_models = sum(1 for r in result['results'] if r['status'] == 'success')
            total_models = len(result['results'])
            total_seqs = result['metadata']['total_sequences']

            print(f"\n✅ Success: Compared {successful_models}/{total_models} models")
            print(f"   Processed {total_seqs} sequences")
            if args.output:
                print(f"   Results saved to: {result['output_file']}")
            print(f"   Total time: {result['metadata']['total_runtime']:.3f}s")
            return 0
        else:
            print(f"\n❌ Error: {result['error']}")
            if "suggestion" in result:
                print(f"   Suggestion: {result['suggestion']}")
            return 1

    except Exception as e:
        print(f"\n❌ Script error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())