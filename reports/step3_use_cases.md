# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2024-12-24
- **Filter Applied**: RNA secondary structure prediction, thermodynamic regularization, deep learning-based structure prediction
- **Python Version**: 3.7+ (detected from repository)
- **Environment Strategy**: Dual (main: Python 3.10, legacy: Python 3.7)
- **Repository**: MXfold2 - RNA secondary structure prediction using deep learning with thermodynamic integration

## Use Cases

### UC-001: Basic RNA Secondary Structure Prediction
- **Description**: Predict RNA secondary structures from FASTA sequences using pre-trained neural network models
- **Script Path**: `examples/use_case_1_basic_prediction.py`
- **Complexity**: Simple
- **Priority**: High
- **Environment**: `./env_py37` (for MXfold2 execution)
- **Source**: `mxfold2/predict.py`, README.md examples

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_file | file | FASTA file with RNA sequences | --input, -i |
| model_type | string | Model type (Turner, Mix, MixC, Zuker, etc.) | --model, -m |
| output_dir | directory | Optional output directory for detailed results | --output, -o |
| gpu | integer | GPU device ID (-1 for CPU) | --gpu |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| structures | text | Dot-bracket notation structures |
| energies | float | Minimum free energies (kcal/mol) |
| bpseq_files | files | Optional BPSEQ format files |

**Example Usage:**
```bash
python examples/use_case_1_basic_prediction.py --input examples/data/test_short.fa --model MixC --output results/
```

**Example Data**: `examples/data/test_short.fa`, `examples/data/sample_rna.fa`

---

### UC-002: Deep Learning Model Comparison
- **Description**: Compare multiple MXfold2 models on the same RNA sequences, showcasing different approaches
- **Script Path**: `examples/use_case_2_model_comparison.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py37`
- **Source**: `mxfold2/predict.py`, multiple model types analysis

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_file | file | FASTA file with RNA sequences | --input, -i |
| models | list | Comma-separated model names | --models, -m |
| output_file | file | CSV file for comparison results | --output, -o |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| comparison_csv | file | Detailed model comparison results |
| runtime_analysis | data | Performance metrics per model |
| structure_comparison | data | Side-by-side structure predictions |

**Example Usage:**
```bash
python examples/use_case_2_model_comparison.py --input examples/data/sample_rna.fa --models Turner,Mix,MixC --output comparison.csv
```

**Example Data**: `examples/data/sample_rna.fa`

---

### UC-003: Thermodynamic Analysis with Turner Parameters
- **Description**: Analyze RNA structures using traditional thermodynamic parameters (Turner 2004) as baseline
- **Script Path**: `examples/use_case_3_thermodynamic_analysis.py`
- **Complexity**: Medium
- **Priority**: Medium
- **Environment**: `./env_py37`
- **Source**: `mxfold2/predict.py` (Turner model), `param_turner2004.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_file | file | FASTA file with RNA sequences | --input, -i |
| detailed | flag | Show detailed structural analysis | --detailed, -d |
| save_results | file | Save detailed results to file | --save-results, -s |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| structures | text | Thermodynamic structure predictions |
| analysis | data | Structural features (stems, loops, pairs) |
| energies | data | MFE and energy density calculations |
| summary_stats | data | Dataset-wide statistics |

**Example Usage:**
```bash
python examples/use_case_3_thermodynamic_analysis.py --input examples/data/sample_rna.fa --detailed --save-results analysis.txt
```

**Example Data**: `examples/data/sample_rna.fa`

---

### UC-004: Custom Model Training Demonstration
- **Description**: Demonstrate training custom MXfold2 models on RNA datasets (educational/demo version)
- **Script Path**: `examples/use_case_4_model_training.py`
- **Complexity**: Complex
- **Priority**: Medium
- **Environment**: `./env_py37`
- **Source**: `mxfold2/train.py`, training configuration analysis

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| dataset_dir | directory | Directory with training data | --dataset, -d |
| model_type | string | Model type to train (Mix, MixC, etc.) | --model, -m |
| epochs | integer | Number of training epochs | --epochs, -e |
| output_dir | directory | Directory for trained model | --output, -o |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| trained_model | file | Trained model parameters (.pth) |
| config_file | file | Model configuration (.conf) |
| training_log | data | Training progress and metrics |
| demo_dataset | files | Generated demo training files |

**Example Usage:**
```bash
python examples/use_case_4_model_training.py --dataset examples/data/training --model MixC --epochs 3 --output models/trained
```

**Example Data**: Auto-generated demo BPSEQ files in specified dataset directory

---

## Summary

| Metric | Count |
|--------|-------|
| Total Found | 4 |
| Scripts Created | 4 |
| High Priority | 2 |
| Medium Priority | 2 |
| Low Priority | 0 |
| Demo Data Copied | ✅ |

## Demo Data Index

| Source | Destination | Description |
|--------|-------------|-------------|
| `repo/mxfold2/mxfold2/models/TrainSetAB.conf` | `examples/models/TrainSetAB.conf` | Pre-trained model configuration |
| `repo/mxfold2/mxfold2/models/TrainSetAB.pth` | `examples/models/TrainSetAB.pth` | Pre-trained model weights (3.2MB) |
| Created | `examples/data/sample_rna.fa` | Sample RNA sequences including tRNA, rRNA, hairpins |
| Created | `examples/data/test_short.fa` | Short test sequences for quick testing |

## Key Features Identified

### Core MXfold2 Capabilities
1. **Deep Learning Integration**: Neural networks (CNN, LSTM, Transformer) with thermodynamic parameters
2. **Multiple Model Types**: Turner, Mix, MixC, Zuker variants for different approaches
3. **Thermodynamic Foundation**: Turner 2004 parameters for classical RNA folding
4. **Flexible Input/Output**: FASTA input, dot-bracket/BPSEQ output, energy scoring

### Use Case Coverage
- ✅ **RNA Secondary Structure Prediction**: Core functionality covered
- ✅ **Thermodynamic Regularization**: Turner model and parameter integration
- ✅ **Deep Learning-Based Structure Prediction**: Neural network models (Mix, MixC)
- ✅ **Model Comparison**: Comparative analysis capabilities
- ✅ **Custom Training**: Educational demonstration of training workflow

### Technical Implementation
- **Environment Compatibility**: Scripts handle both environments appropriately
- **Error Handling**: Graceful degradation when MXfold2 not available
- **Documentation**: Comprehensive usage examples and parameter documentation
- **Modularity**: Each use case is standalone and independently executable

## Validation Status
- [x] All scripts created with proper imports and error handling
- [x] Demo data prepared and accessible
- [x] Model files copied to examples directory
- [x] Scripts documented with usage examples
- [x] Environment compatibility addressed
- [x] Use cases match filtering criteria (RNA structure prediction + deep learning + thermodynamics)