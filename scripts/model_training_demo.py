#!/usr/bin/env python3
"""
Script: model_training_demo.py
Description: Demonstrate MXfold2 model training workflow

Original Use Case: examples/use_case_4_model_training.py
Dependencies Removed: Inlined training data generation, simplified training demo

Usage:
    python scripts/model_training_demo.py --output <output_dir>

Example:
    python scripts/model_training_demo.py --output results/training_demo --epochs 3
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import os

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parent))
from lib.io import save_json
from lib.utils import check_mxfold2_available, timing_context, ensure_directory, safe_filename

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_type": "Mix",
    "epochs": 3,
    "create_demo_data": True,
    "verbose": True
}

VALID_MODELS = ["Mix", "MixC", "Zuker", "ZukerC"]

# ==============================================================================
# Inlined Demo Data Generation Functions
# ==============================================================================
def create_demo_bpseq_file(filepath: Path, sequence: str, pairs: List[tuple], name: str) -> None:
    """
    Create a BPSEQ file with given sequence and base pairs.
    Inlined from original training script.
    """
    with open(filepath, 'w') as f:
        f.write(f"# {name}\n")
        for i, nt in enumerate(sequence, 1):
            pair_with = 0
            for p1, p2 in pairs:
                if p1 == i:
                    pair_with = p2
                elif p2 == i:
                    pair_with = p1
            f.write(f"{i}\t{nt}\t{pair_with}\n")

def create_demo_training_data(output_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Create demo training dataset for demonstration.
    Inlined from original use case script.
    """
    ensure_directory(output_dir)

    # Demo RNA structures with known secondary structures
    demo_structures = [
        {
            "name": "hairpin1",
            "sequence": "GGGGAAAACCCC",
            "pairs": [(1,12), (2,11), (3,10), (4,9)]
        },
        {
            "name": "stemloop",
            "sequence": "GCGCAAAGCGC",
            "pairs": [(1,11), (2,10), (3,9)]
        },
        {
            "name": "pseudoknot_simple",
            "sequence": "GCGCUGCUGCGC",
            "pairs": [(1,12), (2,11), (3,10), (4,9), (5,8), (6,7)]
        },
        {
            "name": "bulge_loop",
            "sequence": "GGCCUAGGCC",
            "pairs": [(1,10), (2,9), (6,8)]  # Bulge at positions 3-5
        },
        {
            "name": "internal_loop",
            "sequence": "GGGCAAAGCCC",
            "pairs": [(1,11), (2,10), (3,9)]  # Internal loop at 4-8
        }
    ]

    created_files = []

    for struct in demo_structures:
        filepath = output_dir / f"{struct['name']}.bpseq"
        create_demo_bpseq_file(
            filepath,
            struct['sequence'],
            struct['pairs'],
            struct['name']
        )
        created_files.append(str(filepath))

        if verbose:
            print(f"  Created: {struct['name']}.bpseq ({len(struct['sequence'])} nt)")

    # Create training list file
    train_list = output_dir / "train.lst"
    with open(train_list, 'w') as f:
        for file_path in created_files:
            f.write(f"{file_path}\n")

    return {
        "status": "success",
        "train_list": str(train_list),
        "num_files": len(created_files),
        "files": created_files,
        "structures": demo_structures
    }

def simulate_training_process(model_type: str, epochs: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Simulate model training process for demonstration.
    This is a mock implementation since actual training requires large datasets.
    """
    if verbose:
        print(f"  🧠 Simulating {model_type} model training...")
        print(f"     Epochs: {epochs}")

    # Simulate training metrics (mock data)
    training_log = []
    for epoch in range(1, epochs + 1):
        # Mock training metrics
        loss = 10.0 * (0.8 ** epoch) + 0.1  # Decreasing loss
        accuracy = 0.5 + 0.4 * (1 - 0.8 ** epoch)  # Increasing accuracy

        epoch_data = {
            "epoch": epoch,
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4),
            "learning_rate": 0.001
        }
        training_log.append(epoch_data)

        if verbose:
            print(f"     Epoch {epoch:2d}: Loss={loss:.4f}, Acc={accuracy:.3f}")

    return {
        "status": "success",
        "model_type": model_type,
        "epochs": epochs,
        "final_loss": training_log[-1]["loss"],
        "final_accuracy": training_log[-1]["accuracy"],
        "training_log": training_log
    }

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_model_training_demo(
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Demonstrate MXfold2 model training workflow.

    Args:
        output_dir: Directory to save training demo outputs
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - demo_data: Information about created demo data
            - training_result: Training simulation results
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_model_training_demo("output/training")
        >>> print(result['training_result']['final_accuracy'])
    """
    # Setup
    output_dir = Path(output_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    ensure_directory(output_dir)

    # Validate model type
    model_type = config["model_type"]
    if model_type not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model_type}'. Valid models: {VALID_MODELS}")

    if config.get("verbose"):
        print(f"MXfold2 Model Training Demo")
        print(f"Output directory: {output_dir}")
        print(f"Model type: {model_type}")
        print(f"Epochs: {config['epochs']}")
        print("=" * 50)

    results = {}

    with timing_context("training_demo") as timer:
        # Step 1: Create demo training data
        if config.get("create_demo_data"):
            if config.get("verbose"):
                print(f"\n📁 Creating demo training dataset...")

            dataset_dir = output_dir / "training_data"
            demo_data = create_demo_training_data(dataset_dir, config.get("verbose", False))
            results["demo_data"] = demo_data

            if config.get("verbose"):
                print(f"   ✓ Created {demo_data['num_files']} training files")
        else:
            results["demo_data"] = {"status": "skipped"}

        # Step 2: Check MXfold2 availability (for real training)
        mxfold2_status = check_mxfold2_available()
        results["mxfold2_available"] = mxfold2_status["available"]

        if mxfold2_status["available"]:
            if config.get("verbose"):
                print(f"\n✓ MXfold2 is available for actual training")
        else:
            if config.get("verbose"):
                print(f"\n⚠ MXfold2 not available - running simulation mode only")
                print(f"   {mxfold2_status['error']}")

        # Step 3: Simulate training process
        if config.get("verbose"):
            print(f"\n🚀 Starting training simulation...")

        training_result = simulate_training_process(
            model_type=model_type,
            epochs=config["epochs"],
            verbose=config.get("verbose", False)
        )
        results["training_result"] = training_result

        # Step 4: Create training configuration file
        training_config = {
            "model_type": model_type,
            "epochs": config["epochs"],
            "dataset": str(output_dir / "training_data" / "train.lst") if config.get("create_demo_data") else None,
            "parameters": {
                "learning_rate": 0.001,
                "batch_size": 1,
                "optimizer": "Adam",
                "loss_function": "CrossEntropy"
            },
            "demo_mode": True,
            "mxfold2_available": mxfold2_status["available"]
        }

        config_file = output_dir / "training_config.json"
        save_json(training_config, config_file)
        results["config_file"] = str(config_file)

        # Step 5: Save training results
        results_file = output_dir / "training_results.json"
        save_json({
            "training_result": training_result,
            "demo_data": results.get("demo_data", {}),
            "config": training_config,
            "metadata": {
                "processing_time": timer.elapsed,
                "demo_mode": True
            }
        }, results_file)
        results["results_file"] = str(results_file)

    if config.get("verbose"):
        print(f"\n📊 Training demo completed!")
        print(f"   Final accuracy: {training_result['final_accuracy']:.3f}")
        print(f"   Final loss: {training_result['final_loss']:.4f}")
        print(f"   Results saved to: {output_dir}")
        print(f"   Processing time: {timer.elapsed:.3f}s")

    return {
        "status": "success",
        "demo_data": results.get("demo_data", {}),
        "training_result": training_result,
        "output_dir": str(output_dir),
        "config_file": results["config_file"],
        "results_file": results["results_file"],
        "metadata": {
            "config": config,
            "processing_time": timer.elapsed,
            "demo_mode": True,
            "mxfold2_available": mxfold2_status["available"]
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
    parser.add_argument('--output', '-o', required=True, help='Output directory for training demo')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--model', '-m', choices=VALID_MODELS, default='Mix',
                       help='Model type to train')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--no-demo-data', action='store_true',
                       help='Skip demo data creation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI args
    overrides = {
        "model_type": args.model,
        "epochs": args.epochs,
        "create_demo_data": not args.no_demo_data
    }
    if args.verbose:
        overrides["verbose"] = True

    # Run
    try:
        result = run_model_training_demo(
            output_dir=args.output,
            config=config,
            **overrides
        )

        if result["status"] == "success":
            training = result['training_result']
            print(f"\n✅ Success: Training demo completed")
            print(f"   Model: {training['model_type']}")
            print(f"   Epochs: {training['epochs']}")
            print(f"   Final accuracy: {training['final_accuracy']:.3f}")
            print(f"   Output: {result['output_dir']}")
            return 0
        else:
            print(f"\n❌ Error: Training demo failed")
            return 1

    except Exception as e:
        print(f"\n❌ Script error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())