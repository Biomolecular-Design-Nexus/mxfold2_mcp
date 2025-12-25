# MXfold2 MCP

> RNA secondary structure prediction with deep learning and thermodynamic integration via Model Context Protocol (MCP)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The MXfold2 MCP provides RNA secondary structure prediction capabilities through both direct script usage and Model Context Protocol (MCP) integration. This tool combines deep learning models with thermodynamic parameters to accurately predict RNA structures, offering both traditional Turner model approaches and modern neural network-based methods.

### Features
- **Deep Learning Integration**: Neural networks (CNN, LSTM, Transformer) with thermodynamic parameters
- **Multiple Model Types**: Turner, Mix, MixC, Zuker variants for different prediction approaches
- **Thermodynamic Foundation**: Turner 2004 parameters for classical RNA folding
- **Flexible Input/Output**: FASTA input, dot-bracket/BPSEQ output, energy scoring
- **Async Job Management**: Background processing for large datasets with job tracking
- **Comprehensive Analysis**: Structure prediction, model comparison, thermodynamic analysis

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Main conda environment (Python 3.10)
├── env_py37/               # Legacy environment for MXfold2 (Python 3.7)
├── src/
│   ├── server.py           # MCP server
│   └── jobs/               # Job management system
├── scripts/
│   ├── rna_structure_prediction.py     # RNA structure prediction
│   ├── model_comparison.py             # Compare multiple models
│   ├── thermodynamic_analysis.py       # Thermodynamic analysis
│   ├── model_training_demo.py          # Training demonstration
│   └── lib/                            # Shared utilities
├── examples/
│   ├── data/               # Demo FASTA files
│   └── models/             # Pre-trained model files
├── configs/                # Configuration files
├── repo/                   # Original MXfold2 repository
└── reports/                # Documentation and test reports
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+ (for MCP server)
- Python 3.7+ (for MXfold2 compatibility)
- Git (for cloning repositories)

### Create Environments

This project uses a dual environment setup to handle version compatibility:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/mxfold2_mcp

# Determine package manager (prefer mamba over conda)
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi
echo "Using package manager: $PKG_MGR"

# Create main MCP environment (Python 3.10)
$PKG_MGR create -p ./env python=3.10 pip -y
$PKG_MGR run -p ./env pip install loguru click pandas numpy tqdm
$PKG_MGR run -p ./env pip install --force-reinstall --no-cache-dir fastmcp

# Create legacy environment for MXfold2 (Python 3.7)
$PKG_MGR create -p ./env_py37 python=3.7 -y
$PKG_MGR run -p ./env_py37 pip install torch==1.9.0 torchvision==0.10.0 numpy==1.21.0 tqdm pybind11

# Install MXfold2 in legacy environment
$PKG_MGR activate ./env_py37
pip install -e repo/mxfold2
```

### Verify Installation

```bash
# Test main environment
$PKG_MGR run -p ./env python -c "import fastmcp; print('FastMCP available')"

# Test legacy environment
$PKG_MGR run -p ./env_py37 python -c "import torch, mxfold2; print('MXfold2 available')"
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/rna_structure_prediction.py` | Predict RNA secondary structures | See below |
| `scripts/model_comparison.py` | Compare multiple models on same data | See below |
| `scripts/thermodynamic_analysis.py` | Analyze with Turner parameters | See below |
| `scripts/model_training_demo.py` | Training workflow demonstration | See below |

### Script Examples

#### RNA Structure Prediction

```bash
# Activate legacy environment
mamba activate ./env_py37

# Basic prediction with Turner model
python scripts/rna_structure_prediction.py \
  --input examples/data/sample_rna.fa \
  --output results/prediction.json \
  --model Turner

# Use neural model with GPU
python scripts/rna_structure_prediction.py \
  --input examples/data/sample_rna.fa \
  --output results/mix_prediction.json \
  --model Mix \
  --gpu 0
```

**Parameters:**
- `--input, -i`: Path to FASTA file with RNA sequences (required)
- `--output, -o`: Output file path for predictions (default: stdout)
- `--model, -m`: Model type ("Turner", "Mix", "MixC", "Zuker") (default: Turner)
- `--gpu`: GPU device ID (-1 for CPU) (default: -1)
- `--verbose`: Enable verbose output

#### Model Comparison

```bash
# Compare multiple models
python scripts/model_comparison.py \
  --input examples/data/sample_rna.fa \
  --models Turner,Mix,MixC \
  --output results/comparison.csv
```

**Parameters:**
- `--input, -i`: Path to FASTA file with RNA sequences (required)
- `--models, -m`: Comma-separated model names (default: Turner,Mix)
- `--output, -o`: Output file path for comparison results
- `--verbose`: Enable verbose output

#### Thermodynamic Analysis

```bash
# Detailed thermodynamic analysis
python scripts/thermodynamic_analysis.py \
  --input examples/data/sample_rna.fa \
  --output results/analysis.json \
  --detailed
```

**Parameters:**
- `--input, -i`: Path to FASTA file with RNA sequences (required)
- `--output, -o`: Output file path for analysis results
- `--detailed`: Include detailed structural features
- `--verbose`: Enable verbose output

#### Training Demo

```bash
# Run training demonstration
python scripts/model_training_demo.py \
  --output results/training_demo \
  --model Mix \
  --epochs 3
```

**Parameters:**
- `--output, -o`: Directory for training demo outputs (required)
- `--model, -m`: Model type to simulate (default: Mix)
- `--epochs, -e`: Number of training epochs (default: 3)
- `--verbose`: Enable verbose output

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Activate main environment
mamba activate ./env

# Install MCP server for Claude Code
fastmcp install src/server.py --name mxfold2
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add mxfold2 -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "mxfold2": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/mxfold2_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/mxfold2_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from mxfold2?
```

#### Basic Structure Prediction
```
Use predict_rna_structure with input_file @examples/data/sample_rna.fa and model "Turner"
```

#### Model Comparison
```
Use submit_model_comparison to compare Turner, Mix, and MixC models on @examples/data/sample_rna.fa
```

#### Thermodynamic Analysis
```
Use analyze_thermodynamics with input_file @examples/data/sample_rna.fa and detailed_analysis true
```

#### Job Management
```
Submit model comparison for @examples/data/sample_rna.fa comparing Turner and Mix models
Then check the job status
```

#### Batch Processing
```
Process these files in batch:
- @examples/data/sample_rna.fa
- @examples/data/test_short.fa
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sample_rna.fa` | Reference a specific FASTA file |
| `@configs/default_config.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "mxfold2": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/mxfold2_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/mxfold2_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use predict_rna_structure with file examples/data/sample_rna.fa
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict_rna_structure` | Predict RNA secondary structures | `input_file`, `model`, `output_file`, `gpu`, `verbose` |
| `analyze_thermodynamics` | RNA thermodynamic analysis | `input_file`, `output_file`, `detailed_analysis`, `gpu`, `verbose` |
| `run_training_demo` | Model training demonstration | `output_dir`, `model_type`, `epochs`, `create_demo_data`, `verbose` |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_model_comparison` | Compare multiple models | `input_file`, `models`, `output_dir`, `gpu`, `job_name` |
| `submit_batch_structure_prediction` | Process multiple files | `input_files`, `model`, `output_dir`, `gpu`, `job_name` |
| `submit_large_dataset_analysis` | Analyze large datasets | `input_file`, `analysis_type`, `output_dir`, `gpu`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: Basic Structure Prediction

**Goal:** Predict RNA secondary structures using the Turner model

**Using Script:**
```bash
mamba activate ./env_py37
python scripts/rna_structure_prediction.py \
  --input examples/data/sample_rna.fa \
  --output results/example1/prediction.json \
  --model Turner \
  --verbose
```

**Using MCP (in Claude Code):**
```
Use predict_rna_structure to process @examples/data/sample_rna.fa with Turner model and save results to results/example1/
```

**Expected Output:**
- JSON file with structure predictions in dot-bracket notation
- MFE energies for each sequence
- Processing metadata and timing information

### Example 2: Model Comparison Analysis

**Goal:** Compare Turner and Mix models on the same sequences

**Using Script:**
```bash
mamba activate ./env_py37
python scripts/model_comparison.py \
  --input examples/data/sample_rna.fa \
  --models Turner,Mix \
  --output results/example2/comparison.csv \
  --verbose
```

**Using MCP (in Claude Code):**
```
Submit model comparison for @examples/data/sample_rna.fa comparing Turner and Mix models
```

**Expected Output:**
- CSV file with side-by-side model comparisons
- Performance metrics for each model
- Structure differences and energy comparisons

### Example 3: Thermodynamic Analysis

**Goal:** Detailed thermodynamic analysis with structural features

**Using Script:**
```bash
mamba activate ./env_py37
python scripts/thermodynamic_analysis.py \
  --input examples/data/sample_rna.fa \
  --output results/example3/analysis.json \
  --detailed \
  --verbose
```

**Using MCP (in Claude Code):**
```
Use analyze_thermodynamics on @examples/data/sample_rna.fa with detailed_analysis true
```

**Expected Output:**
- Detailed structural feature analysis (base pairs, stems, loops)
- Energy density calculations
- Summary statistics across all sequences

### Example 4: Batch Processing

**Goal:** Process multiple files simultaneously

**Using Script:**
```bash
for f in examples/data/*.fa; do
  mamba run -p ./env_py37 python scripts/rna_structure_prediction.py --input "$f" --output results/batch/
done
```

**Using MCP (in Claude Code):**
```
Submit batch processing for all FASTA files in @examples/data/
```

**Expected Output:**
- Individual prediction files for each input file
- Consolidated processing log and statistics
- Job tracking with completion status

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `sample_rna.fa` | Sample RNA sequences including tRNA, rRNA, hairpins | Any prediction tool |
| `test_short.fa` | Short test sequences for quick testing | Any prediction tool |

### Sample Data Details
- **sample_rna.fa**: Contains diverse RNA types for comprehensive testing
- **test_short.fa**: Contains 2 short sequences (~30-50 nucleotides) for rapid testing

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `default_config.json` | Global defaults for all scripts | gpu, models, paths, output formats |
| `rna_structure_prediction_config.json` | RNA prediction settings | model selection, output options |
| `model_comparison_config.json` | Model comparison parameters | model lists, comparison metrics |
| `thermodynamic_analysis_config.json` | Thermodynamic analysis options | detailed analysis flags |
| `model_training_demo_config.json` | Training demo configuration | training parameters, demo settings |

### Config Example

```json
{
  "_description": "Configuration for RNA structure prediction",
  "model": "Turner",
  "gpu": -1,
  "output_format": "json",
  "include_metadata": true,
  "examples": {
    "basic": {
      "input": "examples/data/sample_rna.fa",
      "output": "results/prediction.json"
    }
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environments
mamba create -p ./env python=3.10 -y
mamba create -p ./env_py37 python=3.7 -y
# Follow installation steps above
```

**Problem:** Import errors in legacy environment
```bash
# Verify MXfold2 installation
mamba activate ./env_py37
python -c "import torch, mxfold2; print('Success')"

# Reinstall if needed
pip install -e repo/mxfold2
```

**Problem:** FastMCP import errors
```bash
# Reinstall fastmcp in main environment
mamba activate ./env
pip install --force-reinstall --no-cache-dir fastmcp
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove mxfold2
claude mcp add mxfold2 -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
mamba activate ./env
python -c "
from src.server import mcp
print(list(mcp.list_tools().keys()))
"
```

**Problem:** Wrong environment being used
```bash
# Verify correct Python interpreter
which python  # Should point to ./env/bin/python for MCP server
```

### Script Issues

**Problem:** MXfold2 not found
```bash
# Check MXfold2 installation in legacy environment
mamba activate ./env_py37
python -c "import mxfold2; print(mxfold2.__file__)"

# Install if missing
cd repo/mxfold2
pip install -e .
```

**Problem:** Model files not found
```bash
# Check model files exist
ls -la examples/models/
# Should show TrainSetAB.conf and TrainSetAB.pth

# Copy from repo if missing
cp repo/mxfold2/mxfold2/models/* examples/models/
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job metadata
cat jobs/<job_id>/metadata.json
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

**Problem:** Jobs accumulating on disk
```bash
# Clean old job directories (optional)
find jobs/ -type d -name "????????" -mtime +7 -exec rm -rf {} \;
```

---

## Development

### Running Tests

```bash
# Activate main environment
mamba activate ./env

# Run all tests
python tests/run_integration_tests.py

# Test specific functionality
python tests/test_mcp_server.py
python tests/test_tools_functionality.py
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
mamba activate ./env
fastmcp dev src/server.py

# Test with MCP inspector
npx @anthropic/mcp-inspector src/server.py
```

### Performance Monitoring

```bash
# Monitor job processing
tail -f jobs/*/job.log

# Check disk usage
du -sh jobs/ results/
```

---

## License

This project builds upon the MXfold2 software. Please refer to the original license terms.

## Credits

Based on [MXfold2](https://github.com/mxfold/mxfold2) - RNA secondary structure prediction using deep learning with thermodynamic integration.

### References
- Sato K, Akiyama M, Sakakibara Y. RNA secondary structure prediction using deep learning with thermodynamic integration. Nature Communications. 2021.
- Turner DH, Mathews DH. NNDB: the nearest neighbor parameter database for predicting stability of nucleic acid secondary structure. Nucleic Acids Research. 2010.