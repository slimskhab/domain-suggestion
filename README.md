# Domain Name Suggestion LLM Evaluation Framework

A comprehensive framework for building, evaluating, and iteratively improving fine-tuned LLMs for domain name suggestions with emphasis on systematic evaluation, edge case discovery, and model improvement cycles.

## Project Overview

This project implements a complete pipeline for:
- Synthetic dataset creation for domain name generation
- Model fine-tuning and iteration
- LLM-as-a-Judge evaluation framework
- Edge case discovery and analysis
- Safety guardrails for inappropriate content
- Systematic model improvement cycles

## Project Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                           # Installation script
├── config/                            # Configuration files
│   ├── model_config.yaml             # Model configuration
│   └── evaluation_config.yaml        # Evaluation settings
├── data/                              # Data directory
│   ├── synthetic_dataset/            # Generated datasets
│   ├── evaluation_data/              # Evaluation datasets
│   └── edge_cases/                   # Discovered edge cases
├── models/                            # Model checkpoints
│   ├── baseline/                     # Initial model
│   ├── improved/                     # Improved versions
│   └── evaluation/                   # Evaluation models
├── src/                              # Source code
│   ├── data_generation/              # Dataset creation
│   ├── model_training/               # Model fine-tuning
│   ├── evaluation/                   # Evaluation framework
│   ├── edge_case_discovery/          # Edge case analysis
│   └── safety/                       # Safety guardrails
├── notebooks/                        # Jupyter notebooks
│   ├── 01_dataset_creation.ipynb    # Dataset generation
│   ├── 02_baseline_model.ipynb      # Baseline model training
│   ├── 03_evaluation_framework.ipynb # Evaluation setup
│   ├── 04_edge_case_analysis.ipynb  # Edge case discovery
│   ├── 05_model_improvement.ipynb   # Iterative improvement
│   └── 06_final_evaluation.ipynb    # Final results
├── tests/                            # Unit tests
├── api/                              # Optional API deployment
└── reports/                          # Technical reports
    └── technical_report.md           # Final technical report
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd domain-name-llm-evaluation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file with your API keys
cp .env.example .env
# Edit .env with your actual API keys
```

### Quick Start

1. **Generate synthetic dataset**:
```bash
python src/data_generation/create_dataset.py
```

2. **Train baseline model**:
```bash
python src/model_training/train_baseline.py
```

3. **Run evaluation framework**:
```bash
python src/evaluation/run_evaluation.py
```

4. **Discover edge cases**:
```bash
python src/edge_case_discovery/discover_edge_cases.py
```

5. **Improve model iteratively**:
```bash
python src/model_training/improve_model.py
```

## Key Features

### 1. Synthetic Dataset Creation
- Diverse business types and complexity levels
- Systematic dataset generation methodology
- Quality control and validation

### 2. Model Development & Iteration
- Baseline model fine-tuning
- Multiple improvement strategies (LoRA, full fine-tuning)
- Hyperparameter optimization
- Model versioning and checkpointing

### 3. LLM-as-a-Judge Evaluation Framework
- Automated evaluation using LLM-as-a-judge
- Systematic scoring methodology
- Quality assessment metrics

### 4. Edge Case Discovery & Analysis
- Systematic failure mode discovery
- Categorized failure analysis
- Measurable improvement tracking

### 5. Safety Guardrails
- Content filtering for inappropriate requests
- Safety testing and validation

## Model Requirements

- **Domain Name Generator**: Open source LLM (Llama, Mistral, etc.)
- **LLM-as-a-Judge**: Third-party API models or fine-tuned open-source models
- **Reproducibility**: Clear setup instructions and version tracking

## Evaluation Metrics

- Domain name quality score
- Relevance to business type
- Creativity and uniqueness
- Safety compliance
- Edge case handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub.
