# Multi-task Thermometer System (MTS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive system for temperature scaling and calibration of language models using the Thermometer approach. This system implements a multi-task learning framework for improving model calibration through temperature scaling.

## 🎯 Overview

The Multi-task Thermometer System (MTS) is designed to improve the calibration of language models through temperature scaling. It implements the Thermometer model architecture that learns to predict optimal temperature parameters for better calibration.

### Key Features

- **5-shot Learning**: Implements proper 5-shot learning for MMLU dataset
- **Temperature Scaling**: Learns optimal temperature parameters for calibration
- **Multi-task Architecture**: Handles multiple subjects and domains
- **Modular Design**: Clean separation of concerns with modular components
- **Easy Configuration**: YAML-based configuration system
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Access to MMLU dataset
- Language model (e.g., LLaMA-2-7b-chat)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/MTS.git
   cd MTS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements-core.txt
   ```

3. **Quick start with default settings**
   ```bash
   python scripts/quick_start.py
   ```

### Step-by-Step Usage

1. **Data Preprocessing**
   ```bash
   python scripts/preprocess_data.py \
     --input_dir /path/to/mmlu/dataset \
     --output_dir outputs/data_5shot
   ```

2. **Feature Extraction**
   ```bash
   python scripts/extract_features.py \
     --model_path /path/to/language/model \
     --data_dir outputs/data_5shot \
     --output_dir outputs/features_5shot \
     --batch_size 1
   ```

3. **Model Training**
   ```bash
   python scripts/train_models.py \
     --features_dir outputs/features_5shot \
     --output_dir outputs/results_5shot \
     --num_epochs 20
   ```

## 📁 Project Structure

```
MTS/
├── src/                    # Core algorithm modules
│   ├── __init__.py        # Package initialization
│   ├── models.py          # Model definitions (Thermometer, etc.)
│   ├── training.py        # Training logic and evaluation
│   ├── evaluation.py      # Comprehensive evaluation tools
│   └── utils.py           # Utilities and configuration
├── data/                  # Data processing modules
│   ├── processing/        # Data preprocessing
│   │   ├── preprocessor.py   # MMLU data preprocessing
│   │   └── __init__.py      # Package initialization
│   ├── features/          # Feature extraction
│   │   ├── extractor.py     # LM feature extraction
│   │   └── __init__.py      # Package initialization
│   ├── utils/             # Data utilities
│   │   ├── data_utils.py   # Common data utilities
│   │   └── __init__.py      # Package initialization
│   └── README.md          # Data module documentation
├── scripts/               # Execution scripts
│   ├── run_pipeline.py    # Full pipeline execution
│   ├── quick_start.py     # Quick start script
│   ├── preprocess_data.py # Data preprocessing
│   ├── extract_features.py # Feature extraction
│   └── train_models.py    # Model training
├── config/                # Configuration files
│   ├── default.yaml       # Default configuration
│   └── sample_config.yaml  # Sample configuration
├── tests/                 # Test files
├── outputs/               # Output directory
├── requirements.txt       # Full dependencies
├── requirements-core.txt  # Core dependencies
├── requirements-dev.txt  # Development dependencies
└── README.md             # This file
```

## ⚙️ Configuration

The system uses YAML configuration files for easy customization. Key configuration sections:

### Data Configuration
```yaml
data:
  input_dir: "/path/to/mmlu/dataset"
  processed_dir: "data/processed"
  features_dir: "data/features"
  max_length: 1024
```

### Model Configuration
```yaml
model:
  path: "/path/to/language/model"
  device: "auto"
  batch_size: 1
```

### Training Configuration
```yaml
training:
  learning_rate: 1e-4
  weight_decay: 1e-5
  num_epochs: 20
  batch_size: 64
  hidden_dim: 256
```

## 🎯 Methodology

### 5-shot Learning Format

The system implements proper 5-shot learning by:

1. **Data Redistribution**: 
   - Original test set → Training set
   - Original validation set → Test set
   - Original dev set → 5-shot examples

2. **Prompt Construction**:
   ```
   The following are multiple choice questions (with answers) about [subject].
   
   [Example 1 with answer]
   [Example 2 with answer]
   [Example 3 with answer]
   [Example 4 with answer]
   [Example 5 with answer]
   
   [Current question]
   A. [Choice A]
   B. [Choice B]
   C. [Choice C]
   D. [Choice D]
   Answer:
   ```

### Thermometer Model

The Thermometer model learns to predict optimal temperature parameters:

1. **Feature Extractor**: Processes hidden states from language models
2. **Temperature Head**: Predicts temperature scaling parameters
3. **Classifier**: Provides auxiliary classification task
4. **Training**: Joint optimization of classification and calibration objectives

## 📊 Data Flow

```
Raw MMLU Data → Data Processing → Processed Data → Feature Extraction → Features → Model Training → Results
     ↓                    ↓                   ↓                  ↓           ↓          ↓
  Arrow files      JSONL files      5-shot format      Hidden states   Pickle files   JSON reports
```

## 📊 Evaluation

The system evaluates three approaches:

1. **Vanilla Baseline**: Direct prediction without temperature scaling
2. **Oracle Baseline**: Optimal temperature scaling with perfect knowledge
3. **Thermometer Model**: Learned temperature scaling

### Metrics

- **Accuracy**: Standard classification accuracy
- **ECE (Expected Calibration Error)**: Calibration quality
- **Brier Score**: Probabilistic prediction accuracy
- **NLL (Negative Log Likelihood)**: Model confidence

## 🔧 Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src
```

### Code Quality

```bash
# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/

# Type checking
mypy src/
```

## 📈 Results

Expected performance on MMLU dataset:

| Method       | Accuracy | ECE    | Temperature |
|--------------|----------|--------|-------------|
| Vanilla      | ~0.45    | ~0.15  | 1.0         |
| Oracle       | ~0.45    | ~0.05  | Optimal     |
| Thermometer  | ~0.45    | ~0.08  | Learned     |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MMLU dataset for providing the benchmark
- Hugging Face for the transformers library
- PyTorch team for the deep learning framework
- Original Thermometer paper authors for the methodology

## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

## 🔗 Related Links

- [Paper Reference](https://arxiv.org/abs/...)
- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)