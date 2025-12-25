#!/usr/bin/env python3
"""
UC-003: Thermodynamic Analysis with Turner Parameters

This script demonstrates traditional thermodynamic RNA folding using
Turner 2004 parameters, providing a baseline comparison for the
deep learning approaches in MXfold2.

Usage:
    python examples/use_case_3_thermodynamic_analysis.py [--input INPUT_FASTA] [--detailed]

Example:
    python examples/use_case_3_thermodynamic_analysis.py --input examples/data/sample_rna.fa --detailed
"""

import argparse
import os
import sys
import time
from pathlib import Path

# MXfold2 should be installed in the environment
# No need for local path manipulation

def thermodynamic_analysis(input_file, detailed_output=False, save_results=None):
    """
    Analyze RNA structures using thermodynamic parameters

    Args:
        input_file (str): Path to FASTA file with RNA sequences
        detailed_output (bool): Print detailed thermodynamic information
        save_results (str): Optional file to save analysis results

    Returns:
        dict: Analysis results including energies and structures
    """
    try:
        from mxfold2.predict import Predict
        from mxfold2.dataset import FastaDataset
        from torch.utils.data import DataLoader
        import torch

        print(f"MXfold2 Thermodynamic Analysis")
        print(f"Input: {input_file}")
        print(f"Using Turner 2004 parameters")
        print("=" * 50)

        # Create argument namespace for Turner model
        class Args:
            def __init__(self):
                self.input = input_file
                self.model = "Turner"
                self.param = "turner2004"  # Use Turner 2004 parameters
                self.seed = 0
                self.gpu = -1  # CPU only
                self.use_constraint = False
                self.result = save_results
                self.bpseq = None
                self.bpp = None

                # These are not used for Turner model but required by interface
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

        # Load sequences
        test_dataset = FastaDataset(input_file)
        if len(test_dataset) == 0:
            return {
                "status": "error",
                "error": "No sequences found in input file"
            }

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        predictor.test_loader = test_loader

        # Build thermodynamic model
        model, config = predictor.build_model(args)
        predictor.model = model

        print(f"Loaded Turner 2004 thermodynamic parameters")
        print(f"Processing {len(test_dataset)} sequences...")
        print()

        results = []
        total_time = 0

        # Predict structures with detailed analysis
        model.eval()
        with torch.no_grad():
            for headers, seqs, refs in test_loader:
                for header, seq in zip(headers, seqs):
                    start_time = time.time()

                    # Predict structure
                    scs, preds, bps = model([seq])
                    sc = scs[0]
                    pred = preds[0]
                    bp = bps[0] if bps[0] is not None else [0] * (len(seq) + 1)

                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time

                    # Analyze structure
                    analysis = analyze_structure(seq, pred, float(sc))

                    result = {
                        "sequence_id": header,
                        "sequence": seq,
                        "structure": pred,
                        "mfe": float(sc),
                        "length": len(seq),
                        "runtime": elapsed_time,
                        **analysis
                    }
                    results.append(result)

                    # Print results
                    print(f"🧬 {header}")
                    print(f"   Sequence:  {seq}")
                    print(f"   Structure: {pred}")
                    print(f"   MFE:       {sc:.2f} kcal/mol")

                    if detailed_output:
                        print(f"   Length:    {len(seq)} nt")
                        print(f"   Stems:     {analysis['num_stems']}")
                        print(f"   Loops:     {analysis['num_loops']}")
                        print(f"   Pairs:     {analysis['num_pairs']}")
                        print(f"   Bulges:    {analysis['num_bulges']}")
                        print(f"   Runtime:   {elapsed_time:.4f}s")

                    print()

        # Summary statistics
        print("📊 THERMODYNAMIC ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"Total sequences:     {len(results)}")
        print(f"Average MFE:         {sum(r['mfe'] for r in results) / len(results):.2f} kcal/mol")
        print(f"Average length:      {sum(r['length'] for r in results) / len(results):.1f} nt")
        print(f"Average pairs:       {sum(r['num_pairs'] for r in results) / len(results):.1f}")
        print(f"Total runtime:       {total_time:.3f}s")
        print(f"Average per seq:     {total_time / len(results):.4f}s")

        return {
            "status": "success",
            "results": results,
            "summary": {
                "total_sequences": len(results),
                "average_mfe": sum(r['mfe'] for r in results) / len(results),
                "average_length": sum(r['length'] for r in results) / len(results),
                "total_runtime": total_time
            }
        }

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

def analyze_structure(sequence, structure, mfe):
    """
    Analyze structural features of RNA secondary structure

    Args:
        sequence (str): RNA sequence
        structure (str): Dot-bracket structure
        mfe (float): Minimum free energy

    Returns:
        dict: Structural analysis metrics
    """
    analysis = {
        "num_pairs": 0,
        "num_stems": 0,
        "num_loops": 0,
        "num_bulges": 0,
        "gc_content": 0.0,
        "energy_density": 0.0
    }

    # Count base pairs
    analysis["num_pairs"] = structure.count('(')

    # Calculate GC content
    gc_count = sequence.count('G') + sequence.count('C')
    analysis["gc_content"] = gc_count / len(sequence) * 100

    # Energy density (kcal/mol per nucleotide)
    analysis["energy_density"] = mfe / len(sequence)

    # Simple stem counting (consecutive base pairs)
    in_stem = False
    stem_count = 0
    for char in structure:
        if char in '()':
            if not in_stem:
                stem_count += 1
                in_stem = True
        else:
            in_stem = False

    analysis["num_stems"] = stem_count

    # Count loops (unpaired regions between paired regions)
    loop_count = 0
    in_loop = False
    for char in structure:
        if char == '.':
            if not in_loop:
                loop_count += 1
                in_loop = True
        else:
            in_loop = False

    analysis["num_loops"] = loop_count

    # Simple bulge detection (single unpaired bases flanked by pairs)
    bulge_count = 0
    for i in range(1, len(structure) - 1):
        if structure[i] == '.' and structure[i-1] in '()' and structure[i+1] in '()':
            bulge_count += 1

    analysis["num_bulges"] = bulge_count

    return analysis

def main():
    parser = argparse.ArgumentParser(
        description="Thermodynamic RNA secondary structure analysis using Turner parameters"
    )
    parser.add_argument(
        "--input", "-i",
        default="examples/data/test_short.fa",
        help="Input FASTA file with RNA sequences"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed structural analysis"
    )
    parser.add_argument(
        "--save-results", "-s",
        help="Save detailed results to file"
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1

    # Run thermodynamic analysis
    result = thermodynamic_analysis(
        input_file=args.input,
        detailed_output=args.detailed,
        save_results=args.save_results
    )

    if result["status"] == "success":
        print(f"✓ Thermodynamic analysis completed successfully!")
        if args.save_results:
            print(f"  Results saved to: {args.save_results}")
        return 0
    else:
        print(f"✗ Thermodynamic analysis failed: {result['error']}")
        if "suggestion" in result:
            print(f"  Suggestion: {result['suggestion']}")
        return 1

if __name__ == "__main__":
    exit(main())