# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `rna_structure_prediction.py` | Predict RNA secondary structures | Yes (MXfold2) | `configs/rna_structure_prediction_config.json` |
| `model_comparison.py` | Compare multiple MXfold2 models | Yes (MXfold2) | `configs/model_comparison_config.json` |
| `thermodynamic_analysis.py` | Analyze structures with Turner parameters | Yes (MXfold2) | `configs/thermodynamic_analysis_config.json` |
| `model_training_demo.py` | Demonstrate model training workflow | Partially (demo mode) | `configs/model_training_demo_config.json` |

## Usage

```bash
# Activate environment (prefer mamba over conda)
mamba run -p ./env_py37 python scripts/SCRIPT_NAME.py [options]

# RNA structure prediction
mamba run -p ./env_py37 python scripts/rna_structure_prediction.py --input examples/data/sample.fa --output results/output.json

# Model comparison
mamba run -p ./env_py37 python scripts/model_comparison.py --input examples/data/sample.fa --models Turner,Mix --output results/comparison.csv

# Thermodynamic analysis
mamba run -p ./env_py37 python scripts/thermodynamic_analysis.py --input examples/data/sample.fa --output results/analysis.json --detailed

# Training demo
mamba run -p ./env_py37 python scripts/model_training_demo.py --output results/training --epochs 3

# With custom config
python scripts/rna_structure_prediction.py --input FILE --output FILE --config configs/custom.json
```

## Shared Library

Common functions are in `scripts/lib/`:
- `io.py`: File loading/saving (FASTA, JSON, CSV, BPSEQ)
- `utils.py`: General utilities (validation, timing, model info)

## Dependencies

### Essential (Required)
- `torch` - PyTorch for neural network models
- `mxfold2` - Main MXfold2 package (must be built from repo)

### Standard Library (Inlined)
- `argparse`, `os`, `sys`, `pathlib`, `time`, `csv`, `json`

### MXfold2 Classes (Lazy Loaded)
- `mxfold2.predict.Predict` - Main prediction functionality
- `mxfold2.dataset.FastaDataset` - FASTA file handling
- `torch.utils.data.DataLoader` - Batch processing

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:
```python
from scripts.rna_structure_prediction import run_rna_structure_prediction

# In MCP tool:
@mcp.tool()
def predict_rna_structure(input_file: str, output_file: str = None, model: str = "Turner"):
    return run_rna_structure_prediction(input_file, output_file, config={"model": model})
```

## Configuration

All scripts support JSON configuration files:
```json
{
  "model": "Turner",
  "gpu": -1,
  "verbose": true,
  "output": {
    "format": "json",
    "include_metadata": true
  }
}
```

See `configs/` directory for examples.

## Error Handling

Scripts implement graceful error handling:
- Missing MXfold2: Clear error with installation suggestion
- Invalid inputs: Helpful validation messages
- Prediction failures: Continue processing other sequences
- Missing files: Clear file not found errors

## Testing

All scripts have been tested with example data:
- ✅ `rna_structure_prediction.py`: Works with Turner model
- ⚠️ `model_comparison.py`: Minor timing display bug (functional)
- ⚠️ `thermodynamic_analysis.py`: Turner model outputs all-unpaired (known issue from Step 4)
- ✅ `model_training_demo.py`: Demo mode works perfectly

## Notes

- **Environment**: Scripts require `env_py37` environment with MXfold2 installed
- **Performance**: All scripts run efficiently on CPU
- **Output Formats**: JSON (default), CSV (comparison), BPSEQ (structures)
- **GPU Support**: Available but defaults to CPU for compatibility