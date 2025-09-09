# Data Processing Module

This directory contains all data processing related modules for MTS.

## Structure

```
data/
├── processing/           # Data preprocessing
│   ├── preprocessor.py   # Main preprocessing logic
│   └── __init__.py      # Package initialization
├── features/            # Feature extraction
│   ├── extractor.py     # Feature extraction logic
│   └── __init__.py      # Package initialization
├── utils/              # Data utilities
│   ├── data_utils.py   # Common data utilities
│   └── __init__.py      # Package initialization
├── README.md           # This file
└── raw/                # Raw data (empty, user should populate)
```

## Usage

### Data Preprocessing
```python
from data.processing.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(input_dir="/path/to/mmlu", output_dir="data/processed")
preprocessor.process_all_subjects()
```

### Feature Extraction
```python
from data.features.extractor import FeatureExtractor

extractor = FeatureExtractor(model_path="/path/to/model")
extractor.extract_features(
    data_path="data/processed/train.jsonl",
    dev_path="data/processed/dev_by_subject.json",
    output_path="data/features/train_features.pkl"
)
```

### Data Utilities
```python
from data.utils.data_utils import DataValidator, DataStats, PathManager

# Validate data structure
validator = DataValidator()
validator.validate_mmlu_structure("/path/to/mmlu")

# Calculate statistics
stats = DataStats.calculate_data_stats("data/processed")
DataStats.print_data_stats(stats)

# Manage paths
path_manager = PathManager()
features_dir = path_manager.get_features_dir()
```

## Data Flow

1. **Raw Data** → **Preprocessing** → **Processed Data**
2. **Processed Data** → **Feature Extraction** → **Features**
3. **Features** → **Model Training** → **Results**

## File Formats

- **Input**: Arrow files (MMLU dataset)
- **Processed**: JSONL files (train.jsonl, val.jsonl, test.jsonl)
- **Features**: Pickle files (train_features.pkl, val_features.pkl, test_features.pkl)
- **Metadata**: JSON files (config.json, dev_by_subject.json)