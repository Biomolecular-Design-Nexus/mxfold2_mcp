# Configuration Files

This directory contains JSON configuration files for all MCP scripts.

## Configuration Files

| File | Script | Description |
|------|--------|-------------|
| `default_config.json` | All | Global defaults and common settings |
| `rna_structure_prediction_config.json` | `rna_structure_prediction.py` | RNA structure prediction settings |
| `model_comparison_config.json` | `model_comparison.py` | Model comparison parameters |
| `thermodynamic_analysis_config.json` | `thermodynamic_analysis.py` | Thermodynamic analysis options |
| `model_training_demo_config.json` | `model_training_demo.py` | Training demo configuration |

## Usage

### Command Line
```bash
python scripts/SCRIPT_NAME.py --config configs/CONFIG_FILE.json
```

### Programmatic
```python
import json
from scripts.rna_structure_prediction import run_rna_structure_prediction

# Load config
with open('configs/rna_structure_prediction_config.json') as f:
    config = json.load(f)

# Override specific parameters
config['model'] = 'MixC'
config['verbose'] = False

# Run with config
result = run_rna_structure_prediction('input.fa', config=config)
```

## Configuration Structure

### Common Sections
- `_description`: Human-readable description
- `_source`: Original use case script
- `examples`: Usage examples with sample inputs/outputs

### Parameter Sections
- Global settings (gpu, verbose, seed)
- Model-specific parameters
- Output formatting options
- Processing parameters

## Examples

### Basic RNA Prediction
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

### Model Comparison
```json
{
  "models": ["Turner", "Mix", "MixC"],
  "gpu": -1,
  "output": {
    "format": "csv",
    "include_timing": true
  }
}
```

### Thermodynamic Analysis
```json
{
  "model": "Turner",
  "detailed_analysis": true,
  "analysis": {
    "calculate_gc_content": true,
    "analyze_structure_features": true
  }
}
```

## Parameter Override

CLI arguments always override config file values:
```bash
# Config specifies model: "Turner", but CLI overrides to "Mix"
python scripts/rna_structure_prediction.py --config config.json --model Mix
```

## Validation

All configuration files are validated when loaded:
- Required parameters are checked
- Valid model names are enforced
- File paths are validated
- Parameter types are verified

## Tips

1. **Start with defaults**: Use `default_config.json` as a template
2. **Override minimally**: Only specify parameters that differ from defaults
3. **Use examples**: Provided examples show common usage patterns
4. **Test configs**: Validate configurations with small test datasets first