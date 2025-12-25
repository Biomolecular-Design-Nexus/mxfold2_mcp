# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: mxfold2
- **Version**: 1.0.0
- **Created Date**: 2024-12-24
- **Server Path**: `src/server.py`
- **Dependencies**: fastmcp, loguru, MXfold2

## Architecture

The MXfold2 MCP server provides both synchronous and asynchronous APIs for RNA secondary structure analysis:

- **Synchronous Tools**: Fast operations completing in <10 minutes
- **Submit Tools**: Long-running operations for batch processing and complex analyses
- **Job Management**: Full job lifecycle management for async operations

## Job Management Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_job_status` | Check job progress and status | job_id: str |
| `get_job_result` | Get completed job results | job_id: str |
| `get_job_log` | View job execution logs | job_id: str, tail: int = 50 |
| `cancel_job` | Cancel running job | job_id: str |
| `list_jobs` | List all jobs with optional filter | status: Optional[str] |

### Job Status Values
- **pending**: Job submitted but not yet started
- **running**: Job currently executing
- **completed**: Job finished successfully
- **failed**: Job terminated with error
- **cancelled**: Job terminated by user request

## Synchronous Tools (Fast Operations < 10 min)

### predict_rna_structure

**Description**: Predict RNA secondary structures using MXfold2 models
**Source Script**: `scripts/rna_structure_prediction.py`
**Estimated Runtime**: 30 seconds - 2 minutes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_file | str | Yes | - | Path to FASTA file with RNA sequences |
| model | str | No | "Turner" | Model type ("Turner", "Mix", "MixC", "Zuker", etc.) |
| output_file | str | No | None | Optional path to save predictions as JSON |
| gpu | int | No | -1 | GPU device ID (-1 for CPU) |
| verbose | bool | No | False | Enable verbose output |

**Example:**
```
Use predict_rna_structure with input_file "examples/data/sample_rna.fa" and model "Turner"
```

**Returns:**
```json
{
  "status": "success",
  "predictions": [
    {
      "sequence_id": "seq1",
      "sequence": "GGGAAACCC",
      "structure": "(((...)))",
      "score": -2.4,
      "length": 9,
      "model": "Turner"
    }
  ],
  "metadata": {
    "processing_time": 0.523,
    "num_sequences": 1
  }
}
```

---

### analyze_thermodynamics

**Description**: Analyze RNA structures using Turner thermodynamic parameters
**Source Script**: `scripts/thermodynamic_analysis.py`
**Estimated Runtime**: 1-5 minutes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_file | str | Yes | - | Path to FASTA file with RNA sequences |
| output_file | str | No | None | Optional path to save analysis as JSON |
| detailed_analysis | bool | No | True | Include detailed structural features |
| gpu | int | No | -1 | GPU device ID (-1 for CPU) |
| verbose | bool | No | False | Enable verbose output |

**Example:**
```
Use analyze_thermodynamics with input_file "examples/data/sample_rna.fa" and detailed_analysis true
```

**Returns:**
```json
{
  "status": "success",
  "analyses": [
    {
      "sequence_id": "seq1",
      "mfe_energy": -2.4,
      "energy_density": -0.267,
      "structural_features": {
        "base_pairs": 3,
        "stems": 1,
        "gc_content": 0.667
      }
    }
  ],
  "summary": {
    "avg_energy": -2.4,
    "avg_base_pairs": 3.0,
    "success_rate": 1.0
  }
}
```

---

### run_training_demo

**Description**: Demonstrate MXfold2 model training workflow
**Source Script**: `scripts/model_training_demo.py`
**Estimated Runtime**: 1-3 minutes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| output_dir | str | Yes | - | Directory to save training demo outputs |
| model_type | str | No | "Mix" | Model type ("Mix", "MixC", "Zuker", etc.) |
| epochs | int | No | 3 | Number of training epochs for demo |
| create_demo_data | bool | No | True | Whether to create demo training data |
| verbose | bool | No | False | Enable verbose output |

**Example:**
```
Use run_training_demo with output_dir "results/training_demo" and model_type "Mix"
```

---

## Submit Tools (Long Operations > 10 min)

### submit_model_comparison

**Description**: Compare multiple MXfold2 models on same dataset
**Source Script**: `scripts/model_comparison.py`
**Estimated Runtime**: 2-15 minutes
**Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_file | str | Yes | - | Path to FASTA file with RNA sequences |
| models | List[str] | No | ["Turner", "Mix"] | List of model names to compare |
| output_dir | str | No | None | Directory to save outputs |
| gpu | int | No | -1 | GPU device ID (-1 for CPU) |
| job_name | str | No | auto | Custom job name for tracking |

**Example:**
```
Use submit_model_comparison with input_file "examples/data/sample_rna.fa" and models ["Turner", "Mix", "MixC"]
```

**Returns:**
```json
{
  "status": "submitted",
  "job_id": "abc123",
  "message": "Job submitted. Use get_job_status('abc123') to check progress."
}
```

---

### submit_batch_structure_prediction

**Description**: Process multiple sequence files in parallel
**Estimated Runtime**: Variable based on file count
**Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_files | List[str] | Yes | - | List of FASTA file paths to process |
| model | str | No | "Turner" | Model type to use for all predictions |
| output_dir | str | No | None | Directory to save all outputs |
| gpu | int | No | -1 | GPU device ID (-1 for CPU) |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Use submit_batch_structure_prediction with input_files ["file1.fa", "file2.fa", "file3.fa"] and model "Mix"
```

---

### submit_large_dataset_analysis

**Description**: Analyze large datasets that require extended processing time
**Estimated Runtime**: >10 minutes for large datasets
**Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_file | str | Yes | - | Path to large FASTA file |
| analysis_type | str | No | "thermodynamic" | Type ("thermodynamic", "prediction") |
| output_dir | str | No | None | Directory to save outputs |
| gpu | int | No | -1 | GPU device ID (-1 for CPU) |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Use submit_large_dataset_analysis with input_file "large_dataset.fa" and analysis_type "thermodynamic"
```

---

## Workflow Examples

### Quick Prediction (Sync)
```
1. Use predict_rna_structure with input_file "examples/data/sample_rna.fa"
   → Returns predictions immediately (30 sec - 2 min)
```

### Model Comparison (Submit API)
```
1. Submit: Use submit_model_comparison with input_file "examples/data/sample_rna.fa"
           and models ["Turner", "Mix", "MixC"]
   → Returns: {"job_id": "abc123", "status": "submitted"}

2. Check: Use get_job_status with job_id "abc123"
   → Returns: {"status": "running", "started_at": "2024-12-24T10:15:30"}

3. Result: Use get_job_result with job_id "abc123"
   → Returns: Full comparison results when completed
```

### Batch Processing
```
1. Submit: Use submit_batch_structure_prediction with
           input_files ["seq1.fa", "seq2.fa", "seq3.fa"]
   → Processes all files in a single job

2. Monitor: Use get_job_log with job_id to see processing progress

3. Results: Use get_job_result when completed for all file results
```

### Large Dataset Analysis
```
1. Submit: Use submit_large_dataset_analysis with
           input_file "dataset_1000_sequences.fa"
   → Handles large datasets efficiently

2. Track: Use get_job_status periodically to monitor progress

3. Cancel: Use cancel_job if needed to terminate early
```

## Error Handling

All tools return structured error responses:

```json
{
  "status": "error",
  "error": "Description of what went wrong",
  "suggestion": "Helpful tip to fix the issue (when available)"
}
```

Common error scenarios:
- **File not found**: Input file doesn't exist
- **Invalid model**: Model name not recognized
- **MXfold2 unavailable**: MXfold2 not installed or configured
- **Job not found**: Invalid job_id provided
- **GPU error**: GPU device not available

## Installation Requirements

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Install MCP dependencies
pip install fastmcp loguru

# Ensure MXfold2 is installed
pip install -e repo/mxfold2
```

## Development and Testing

```bash
# Run tests
env/bin/python tests/test_mcp_server.py

# Start development server
env/bin/python -m fastmcp dev src/server.py

# Test with MCP inspector
npx @anthropic/mcp-inspector src/server.py
```