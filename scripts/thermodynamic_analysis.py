#!/usr/bin/env python3
"""
Script: thermodynamic_analysis.py
Description: Analyze RNA structures using Turner thermodynamic parameters

Original Use Case: examples/use_case_3_thermodynamic_analysis.py
Dependencies Removed: Inlined thermodynamic analysis logic, simplified structure analysis

Usage:
    python scripts/thermodynamic_analysis.py --input <input_file> --output <output_file>

Example:
    python scripts/thermodynamic_analysis.py --input examples/data/sample.fa --output results/analysis.json
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parent))
from lib.io import load_fasta_simple, save_json
from lib.utils import check_mxfold2_available, create_mxfold2_args, timing_context, ensure_directory

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model": "Turner",
    "param": "turner2004",
    "gpu": -1,
    "seed": 0,
    "detailed_analysis": True,
    "verbose": True
}

# ==============================================================================
# Inlined Structure Analysis Functions
# ==============================================================================
def analyze_structure_features(sequence: str, structure: str) -> Dict[str, Any]:
    """
    Analyze structural features of RNA secondary structure.
    Inlined from thermodynamic analysis to avoid dependencies.
    """
    features = {
        "length": len(sequence),
        "gc_content": (sequence.count('G') + sequence.count('C')) / len(sequence),
        "base_pairs": 0,
        "stems": 0,
        "loops": 0,
        "bulges": 0,
        "hairpins": 0
    }

    # Count base pairs
    features["base_pairs"] = structure.count('(')

    # Simple structure analysis
    if features["base_pairs"] > 0:
        # Count stems (consecutive base pairs)
        in_stem = False
        stem_length = 0

        for i, char in enumerate(structure):
            if char in '()':
                if not in_stem:
                    features["stems"] += 1
                    in_stem = True
                stem_length += 1
            else:
                if in_stem:
                    in_stem = False
                    if stem_length < 4:  # Short stems might be bulges
                        features["bulges"] += 1

        # Estimate loops and hairpins
        unpaired_regions = structure.replace('(', '').replace(')', '').split('.')
        features["loops"] = len([r for r in unpaired_regions if len(r) > 3])
        features["hairpins"] = structure.count('(') - structure.count(')')  # Simplified

    return features

def calculate_energy_density(score: float, length: int) -> float:
    """Calculate energy density (energy per nucleotide)."""
    if length == 0:
        return 0.0
    return score / length

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_thermodynamic_analysis(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze RNA structures using thermodynamic parameters.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_file: Path to save analysis results (JSON format)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - analyses: List of structure analyses
            - summary: Summary statistics
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_thermodynamic_analysis("input.fa")
        >>> print(result['summary']['avg_energy'])
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

    if config.get("verbose"):
        print(f"MXfold2 Thermodynamic Analysis")
        print(f"Input: {input_file}")
        print(f"Using Turner 2004 parameters")
        print("=" * 50)

    analyses = []

    with timing_context("thermodynamic_analysis") as timer:
        try:
            # Create predictor args for Turner model
            args = create_mxfold2_args(
                model_type="Turner",
                input_file=str(input_file),
                gpu=config["gpu"],
                seed=config["seed"],
                param=config["param"]
            )

            # Initialize predictor
            predictor = Predict()

            # Load sequences
            test_dataset = FastaDataset(str(input_file))
            if len(test_dataset) == 0:
                return {
                    "status": "error",
                    "error": "No sequences found in input file"
                }

            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            predictor.test_loader = test_loader

            # Build thermodynamic model
            model, model_config = predictor.build_model(args)
            predictor.model = model

            if config.get("verbose"):
                print(f"✓ Loaded Turner 2004 thermodynamic parameters")

            # Run predictions and analysis
            seq_count = 0
            model.eval()

            with torch.no_grad():
                for headers, seqs, refs in test_loader:
                    try:
                        scs, preds, bps = model(seqs)

                        for header, seq, sc, pred in zip(headers, seqs, scs, preds):
                            seq_count += 1
                            score = float(sc) if hasattr(sc, 'item') else float(sc)

                            # Detailed structural analysis
                            analysis = {
                                "sequence_id": header,
                                "sequence": seq,
                                "structure": pred,
                                "mfe_energy": score,
                                "energy_density": calculate_energy_density(score, len(seq))
                            }

                            if config.get("detailed_analysis"):
                                features = analyze_structure_features(seq, pred)
                                analysis["structural_features"] = features

                            analyses.append(analysis)

                            if config.get("verbose"):
                                pairs = analysis.get("structural_features", {}).get("base_pairs", pred.count('('))
                                print(f"  {header}: {pairs} bp, MFE: {score:.2f} kcal/mol")

                    except Exception as e:
                        # Handle prediction errors gracefully
                        for header, seq in zip(headers, seqs):
                            seq_count += 1
                            analysis = {
                                "sequence_id": header,
                                "sequence": seq,
                                "structure": "." * len(seq),
                                "mfe_energy": 0.0,
                                "energy_density": 0.0,
                                "error": str(e)
                            }

                            if config.get("detailed_analysis"):
                                features = analyze_structure_features(seq, "." * len(seq))
                                analysis["structural_features"] = features

                            analyses.append(analysis)

                            if config.get("verbose"):
                                print(f"  ✗ {header}: Analysis failed - {e}")

        except Exception as e:
            return {
                "status": "error",
                "error": f"Thermodynamic analysis failed: {e}",
                "analyses": []
            }

    # Calculate summary statistics
    summary = calculate_summary_statistics(analyses)

    # Save results if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        ensure_directory(output_path.parent)

        results = {
            "analyses": analyses,
            "summary": summary,
            "metadata": {
                "input_file": str(input_file),
                "config": config,
                "processing_time": timer.elapsed,
                "num_sequences": len(analyses)
            }
        }

        save_json(results, output_path)

        if config.get("verbose"):
            print(f"\n📊 Analysis results saved to: {output_path}")

    # Print summary
    if config.get("verbose"):
        print(f"\n📋 THERMODYNAMIC ANALYSIS SUMMARY")
        print("-" * 50)
        print(f"Sequences analyzed: {summary['num_sequences']}")
        print(f"Average MFE energy: {summary['avg_energy']:.2f} kcal/mol")
        print(f"Average base pairs: {summary['avg_base_pairs']:.1f}")
        print(f"Average GC content: {summary['avg_gc_content']:.2%}")
        print(f"Processing time: {timer.elapsed:.3f}s")

    return {
        "status": "success",
        "analyses": analyses,
        "summary": summary,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "processing_time": timer.elapsed,
            "num_sequences": len(analyses)
        }
    }

def calculate_summary_statistics(analyses: List[Dict]) -> Dict[str, Any]:
    """Calculate summary statistics from analyses."""
    if not analyses:
        return {"num_sequences": 0}

    valid_analyses = [a for a in analyses if "error" not in a]

    if not valid_analyses:
        return {
            "num_sequences": len(analyses),
            "errors": len(analyses),
            "success_rate": 0.0
        }

    energies = [a["mfe_energy"] for a in valid_analyses]
    energy_densities = [a["energy_density"] for a in valid_analyses]

    base_pairs = []
    gc_contents = []

    for analysis in valid_analyses:
        features = analysis.get("structural_features", {})
        if features:
            base_pairs.append(features.get("base_pairs", 0))
            gc_contents.append(features.get("gc_content", 0))

    return {
        "num_sequences": len(analyses),
        "successful_analyses": len(valid_analyses),
        "errors": len(analyses) - len(valid_analyses),
        "success_rate": len(valid_analyses) / len(analyses),
        "avg_energy": sum(energies) / len(energies) if energies else 0.0,
        "min_energy": min(energies) if energies else 0.0,
        "max_energy": max(energies) if energies else 0.0,
        "avg_energy_density": sum(energy_densities) / len(energy_densities) if energy_densities else 0.0,
        "avg_base_pairs": sum(base_pairs) / len(base_pairs) if base_pairs else 0.0,
        "avg_gc_content": sum(gc_contents) / len(gc_contents) if gc_contents else 0.0
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
    parser.add_argument('--output', '-o', help='Output file (.json)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Include detailed structural analysis')
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
    if args.detailed:
        overrides["detailed_analysis"] = True
    if args.gpu != -1:
        overrides["gpu"] = args.gpu
    if args.verbose:
        overrides["verbose"] = True

    # Run
    try:
        result = run_thermodynamic_analysis(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        if result["status"] == "success":
            summary = result['summary']
            print(f"\n✅ Success: Analyzed {summary['successful_analyses']}/{summary['num_sequences']} sequences")
            if args.output:
                print(f"   Output saved to: {result['output_file']}")
            print(f"   Average MFE: {summary.get('avg_energy', 0):.2f} kcal/mol")
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