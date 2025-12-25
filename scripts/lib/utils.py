"""General utilities for MCP scripts.

Common helper functions extracted from use cases.
"""
import time
import os
from pathlib import Path
from typing import Dict, Any, Union, List


def validate_rna_sequence(sequence: str) -> bool:
    """Validate RNA sequence contains only valid nucleotides."""
    valid_nucleotides = set('AUGC')
    return all(c.upper() in valid_nucleotides for c in sequence.strip())


def get_repo_path() -> Path:
    """Get path to the MXfold2 repository."""
    script_dir = Path(__file__).parent.parent.parent
    return script_dir / "repo" / "mxfold2"


def check_mxfold2_available() -> Dict[str, Any]:
    """Check if MXfold2 is available and return status."""
    try:
        import mxfold2
        from mxfold2.predict import Predict
        return {
            "available": True,
            "version": getattr(mxfold2, '__version__', 'unknown'),
            "predict_class": Predict
        }
    except ImportError as e:
        return {
            "available": False,
            "error": str(e),
            "suggestion": "Install mxfold2: pip install -e repo/mxfold2"
        }


def create_mxfold2_args(model_type: str = "Turner",
                       input_file: str = None,
                       gpu: int = -1,
                       **kwargs) -> object:
    """
    Create MXfold2 argument namespace.
    Inlined from use case scripts to avoid code duplication.
    """
    class Args:
        def __init__(self):
            self.input = input_file
            self.model = model_type
            self.param = kwargs.get('param', "")
            self.seed = kwargs.get('seed', 0)
            self.gpu = gpu
            self.use_constraint = kwargs.get('use_constraint', False)
            self.result = kwargs.get('result', None)
            self.bpseq = kwargs.get('bpseq', None)
            self.bpp = kwargs.get('bpp', None)

            # Model hyperparameters (defaults from original scripts)
            self.max_helix_length = kwargs.get('max_helix_length', 30)
            self.embed_size = kwargs.get('embed_size', 0)
            self.num_filters = kwargs.get('num_filters', None)
            self.filter_size = kwargs.get('filter_size', None)
            self.pool_size = kwargs.get('pool_size', None)
            self.dilation = kwargs.get('dilation', 0)
            self.num_lstm_layers = kwargs.get('num_lstm_layers', 0)
            self.num_lstm_units = kwargs.get('num_lstm_units', 0)
            self.num_transformer_layers = kwargs.get('num_transformer_layers', 0)
            self.num_transformer_hidden_units = kwargs.get('num_transformer_hidden_units', 2048)
            self.num_transformer_att = kwargs.get('num_transformer_att', 8)
            self.num_hidden_units = kwargs.get('num_hidden_units', None)
            self.num_paired_filters = kwargs.get('num_paired_filters', [])
            self.paired_filter_size = kwargs.get('paired_filter_size', [])
            self.dropout_rate = kwargs.get('dropout_rate', 0.0)
            self.fc_dropout_rate = kwargs.get('fc_dropout_rate', 0.0)
            self.num_att = kwargs.get('num_att', 0)
            self.pair_join = kwargs.get('pair_join', "cat")
            self.no_split_lr = kwargs.get('no_split_lr', False)

    return Args()


def timing_context(operation_name: str):
    """Simple timing context manager."""
    class TimingContext:
        def __init__(self, name):
            self.name = name
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time.time() - self.start_time

    return TimingContext(operation_name)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """Convert name to safe filename."""
    import re
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about MXfold2 model types."""
    models = {
        "Turner": {
            "type": "thermodynamic",
            "description": "Classical Turner 2004 parameters",
            "neural": False
        },
        "Mix": {
            "type": "hybrid",
            "description": "Neural network with thermodynamic constraints",
            "neural": True
        },
        "MixC": {
            "type": "hybrid",
            "description": "Mix with additional constraints",
            "neural": True
        },
        "Zuker": {
            "type": "thermodynamic",
            "description": "Zuker algorithm variant",
            "neural": False
        }
    }

    return models.get(model_name, {
        "type": "unknown",
        "description": f"Unknown model: {model_name}",
        "neural": None
    })