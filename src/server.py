"""MCP Server for MXfold2

Provides both synchronous and asynchronous (submit) APIs for RNA secondary structure prediction tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("mxfold2")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def predict_rna_structure(
    input_file: str,
    model: str = "Turner",
    output_file: Optional[str] = None,
    gpu: int = -1,
    verbose: bool = False
) -> dict:
    """
    Predict RNA secondary structures using MXfold2 models.

    Fast operation suitable for small to medium datasets (completes in <10 minutes).

    Args:
        input_file: Path to FASTA file with RNA sequences
        model: Model type ("Turner", "Mix", "MixC", "Zuker", etc.)
        output_file: Optional path to save predictions as JSON
        gpu: GPU device ID (-1 for CPU)
        verbose: Enable verbose output

    Returns:
        Dictionary with predictions and metadata

    Example:
        predict_rna_structure("examples/data/sample_rna.fa", model="Turner")
    """
    # Import the script's main function
    try:
        from rna_structure_prediction import run_rna_structure_prediction
    except ImportError:
        return {"status": "error", "error": "Failed to import RNA structure prediction script"}

    try:
        result = run_rna_structure_prediction(
            input_file=input_file,
            output_file=output_file,
            model=model,
            gpu=gpu,
            verbose=verbose
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"RNA structure prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_thermodynamics(
    input_file: str,
    output_file: Optional[str] = None,
    detailed_analysis: bool = True,
    gpu: int = -1,
    verbose: bool = False
) -> dict:
    """
    Analyze RNA structures using Turner thermodynamic parameters.

    Fast operation suitable for small to medium datasets (completes in <10 minutes).

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_file: Optional path to save analysis results as JSON
        detailed_analysis: Include detailed structural features
        gpu: GPU device ID (-1 for CPU)
        verbose: Enable verbose output

    Returns:
        Dictionary with thermodynamic analysis results

    Example:
        analyze_thermodynamics("examples/data/sample_rna.fa", detailed_analysis=True)
    """
    try:
        from thermodynamic_analysis import run_thermodynamic_analysis
    except ImportError:
        return {"status": "error", "error": "Failed to import thermodynamic analysis script"}

    try:
        result = run_thermodynamic_analysis(
            input_file=input_file,
            output_file=output_file,
            detailed_analysis=detailed_analysis,
            gpu=gpu,
            verbose=verbose
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Thermodynamic analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def run_training_demo(
    output_dir: str,
    model_type: str = "Mix",
    epochs: int = 3,
    create_demo_data: bool = True,
    verbose: bool = False
) -> dict:
    """
    Run MXfold2 model training demonstration.

    Fast operation that demonstrates the training workflow (completes in <10 minutes).

    Args:
        output_dir: Directory to save training demo outputs
        model_type: Model type to train ("Mix", "MixC", "Zuker", etc.)
        epochs: Number of training epochs for demo
        create_demo_data: Whether to create demo training data
        verbose: Enable verbose output

    Returns:
        Dictionary with training demo results

    Example:
        run_training_demo("results/training_demo", model_type="Mix", epochs=3)
    """
    try:
        from model_training_demo import run_model_training_demo
    except ImportError:
        return {"status": "error", "error": "Failed to import model training demo script"}

    try:
        result = run_model_training_demo(
            output_dir=output_dir,
            model_type=model_type,
            epochs=epochs,
            create_demo_data=create_demo_data,
            verbose=verbose
        )
        return {"status": "success", **result}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Training demo failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_model_comparison(
    input_file: str,
    models: List[str] = None,
    output_dir: Optional[str] = None,
    gpu: int = -1,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit model comparison for background processing.

    This operation compares multiple MXfold2 models and may take more than 10 minutes
    for large datasets or many models. Use get_job_status() to monitor progress.

    Args:
        input_file: Path to FASTA file with RNA sequences
        models: List of model names to compare (default: ["Turner", "Mix"])
        output_dir: Directory to save outputs
        gpu: GPU device ID (-1 for CPU)
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs

    Example:
        submit_model_comparison("examples/data/sample_rna.fa", models=["Turner", "Mix", "MixC"])
    """
    script_path = str(SCRIPTS_DIR / "model_comparison.py")

    if models is None:
        models = ["Turner", "Mix"]

    # Convert list to comma-separated string for CLI
    models_str = ",".join(models)

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "models": models_str,
            "gpu": gpu,
            "output": output_dir
        },
        job_name=job_name or f"model_comparison_{len(models)}_models"
    )

@mcp.tool()
def submit_batch_structure_prediction(
    input_files: List[str],
    model: str = "Turner",
    output_dir: Optional[str] = None,
    gpu: int = -1,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch RNA structure prediction for multiple input files.

    Processes multiple FASTA files in a single job. Suitable for:
    - Processing many sequence files at once
    - Large-scale structure prediction
    - Parallel processing of independent files

    Args:
        input_files: List of FASTA file paths to process
        model: Model type to use for all predictions
        output_dir: Directory to save all outputs
        gpu: GPU device ID (-1 for CPU)
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job

    Example:
        submit_batch_structure_prediction(["file1.fa", "file2.fa", "file3.fa"], model="Mix")
    """
    script_path = str(SCRIPTS_DIR / "rna_structure_prediction.py")

    # For batch processing, we'll process files sequentially
    # This is a simplified implementation - a real batch system would be more sophisticated
    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_files[0] if input_files else "",  # Process first file
            "model": model,
            "gpu": gpu,
            "output": output_dir
        },
        job_name=job_name or f"batch_prediction_{len(input_files)}_files"
    )

@mcp.tool()
def submit_large_dataset_analysis(
    input_file: str,
    analysis_type: str = "thermodynamic",
    output_dir: Optional[str] = None,
    gpu: int = -1,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit large dataset analysis for background processing.

    This operation analyzes large RNA datasets that may take significant time.
    Suitable for datasets with hundreds or thousands of sequences.

    Args:
        input_file: Path to large FASTA file with RNA sequences
        analysis_type: Type of analysis ("thermodynamic", "prediction")
        output_dir: Directory to save outputs
        gpu: GPU device ID (-1 for CPU)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the analysis job

    Example:
        submit_large_dataset_analysis("large_dataset.fa", analysis_type="thermodynamic")
    """
    if analysis_type == "thermodynamic":
        script_path = str(SCRIPTS_DIR / "thermodynamic_analysis.py")
    elif analysis_type == "prediction":
        script_path = str(SCRIPTS_DIR / "rna_structure_prediction.py")
    else:
        return {"status": "error", "error": f"Invalid analysis type: {analysis_type}"}

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "gpu": gpu,
            "output": output_dir,
            "detailed": True,
            "verbose": True
        },
        job_name=job_name or f"large_analysis_{analysis_type}"
    )

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()