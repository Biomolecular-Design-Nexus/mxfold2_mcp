#!/usr/bin/env python3
"""
Script: rna_structure_prediction.py
Description: Predict RNA secondary structures using MXfold2 models

Original Use Case: examples/use_case_1_basic_prediction.py
Dependencies Removed: Inlined Args class, simplified imports

Usage:
    python scripts/rna_structure_prediction.py --input <input_file> --output <output_file>

Example:
    python scripts/rna_structure_prediction.py --input examples/data/sample.fa --output results/pred.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parent))
from lib.io import load_fasta_simple, save_json, save_bpseq
from lib.utils import check_mxfold2_available, create_mxfold2_args, timing_context, ensure_directory

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model": "Turner",
    "gpu": -1,
    "seed": 0,
    "max_helix_length": 30,
    "use_constraint": False,
    "output_formats": ["json", "bpseq"],
    "verbose": True
}

VALID_MODELS = ["Turner", "Mix", "MixC", "Zuker", "ZukerC", "ZukerL", "ZukerS"]

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_rna_structure_prediction(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict RNA secondary structures using MXfold2.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - predictions: List of structure predictions
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_rna_structure_prediction("input.fa", "output.json")
        >>> print(result['predictions'][0]['structure'])
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

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

    # Load input sequences
    if config.get("verbose"):
        print(f"MXfold2 RNA Secondary Structure Prediction")
        print(f"Input: {input_file}")
        print(f"Model: {config['model']}")
        print(f"Device: {'GPU' if config['gpu'] >= 0 else 'CPU'}")
        print("-" * 50)

    predictions = []

    with timing_context("prediction") as timer:
        try:
            # Create predictor args
            args = create_mxfold2_args(
                model_type=config["model"],
                input_file=str(input_file),
                gpu=config["gpu"],
                **{k: v for k, v in config.items() if k not in ["model", "gpu", "verbose", "output_formats"]}
            )

            # Initialize predictor
            predictor = Predict()

            # Load sequences using MXfold2's dataset
            test_dataset = FastaDataset(str(input_file))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # Build model
            model, model_config = predictor.build_model(args)
            predictor.model = model
            predictor.test_loader = test_loader

            # Load pre-trained parameters if available
            if config["model"] != "Turner" and args.param == "":
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
                                print(f"✓ Loaded pre-trained parameters")
                except Exception as e:
                    if config.get("verbose"):
                        print(f"⚠ Could not load pre-trained parameters: {e}")

            # Run predictions
            model.eval()
            with torch.no_grad():
                for headers, seqs, refs in test_loader:
                    try:
                        scs, preds, bps = model(seqs)

                        for header, seq, sc, pred in zip(headers, seqs, scs, preds):
                            prediction = {
                                "sequence_id": header,
                                "sequence": seq,
                                "structure": pred,
                                "score": float(sc) if hasattr(sc, 'item') else float(sc),
                                "length": len(seq),
                                "model": config["model"]
                            }
                            predictions.append(prediction)

                            if config.get("verbose"):
                                print(f"  {header}: {pred} (MFE: {sc:.2f})")

                    except Exception as e:
                        # Handle prediction errors gracefully
                        for header, seq in zip(headers, seqs):
                            prediction = {
                                "sequence_id": header,
                                "sequence": seq,
                                "structure": "." * len(seq),  # No structure predicted
                                "score": 0.0,
                                "length": len(seq),
                                "model": config["model"],
                                "error": str(e)
                            }
                            predictions.append(prediction)

                            if config.get("verbose"):
                                print(f"  ✗ {header}: Prediction failed - {e}")

        except Exception as e:
            return {
                "status": "error",
                "error": f"Prediction failed: {e}",
                "predictions": []
            }

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        ensure_directory(output_path.parent)

        # Determine format from extension
        if str(output_path).endswith('.json'):
            save_json({"predictions": predictions, "metadata": config}, output_path)
        elif str(output_path).endswith('.bpseq'):
            # Save first sequence as BPSEQ (for single sequence files)
            if predictions:
                pred = predictions[0]
                save_bpseq(pred["sequence"], pred["structure"], pred["sequence_id"], output_path)
        else:
            # Default to JSON
            save_json({"predictions": predictions, "metadata": config}, output_path)

    return {
        "status": "success",
        "predictions": predictions,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "processing_time": timer.elapsed,
            "num_sequences": len(predictions)
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
    parser.add_argument('--output', '-o', help='Output file (.json or .bpseq)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--model', '-m', choices=VALID_MODELS, default='Turner',
                       help='Model type for prediction')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device (-1 for CPU)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI args
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.gpu != -1:
        overrides["gpu"] = args.gpu
    if args.verbose:
        overrides["verbose"] = True

    # Run
    try:
        result = run_rna_structure_prediction(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        if result["status"] == "success":
            print(f"\n✅ Success: Processed {result['metadata']['num_sequences']} sequences")
            if args.output:
                print(f"   Output saved to: {result['output_file']}")
            print(f"   Processing time: {result['metadata']['processing_time']:.3f}s")
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