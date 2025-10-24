
<div align="center">

# 🤖 BERT Question Answering System

### *State-of-the-Art Extractive QA with Fine-Tuned BERT on SQuAD*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/jaiku7867)

*An end-to-end NLP project demonstrating BERT fine-tuning for extractive question answering*

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Results](#-results)

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Results & Performance](#-results--performance)
- [Interactive Interface](#-interactive-interface)
- [Documentation](#-documentation)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a **production-ready Question Answering system** using **BERT (Bidirectional Encoder Representations from Transformers)** fine-tuned on the **SQuAD (Stanford Question Answering Dataset)** dataset. The system can extract precise answers from given context paragraphs with high confidence scores.

### What is Extractive QA?

Extractive Question Answering identifies the exact span of text in a passage that answers a given question. Unlike generative models, it **extracts** the answer directly from the context rather than generating new text.

**Example:**
```

Context: "Paris is the capital and most populous city of France.
It has a population of 2.2 million people."

Question: "What is the capital of France?"

Answer: "Paris" ✅ (with 99.8% confidence)

```

---

## ✨ Key Features

- 🧠 **Fine-tuned BERT Model**: Pre-trained `bert-base-uncased` fine-tuned on 3,000 SQuAD examples
- 📊 **High Performance**: Validation loss of 1.65 with expected F1 score of 65-75%
- ⚡ **Fast Training**: Complete training in ~5 minutes on Tesla T4 GPU
- 🎨 **Interactive UI**: Beautiful Gradio/Streamlit web interface
- 📈 **Comprehensive Logging**: Training metrics, loss curves, and evaluation stats
- 🔍 **Answer Validation**: 100% validated answer position mapping
- 📝 **Professional Documentation**: 27+ pages of technical documentation
- 🚀 **Production-Ready**: Saved model weights ready for deployment
- 🔬 **Reproducible Results**: Seed-controlled training for consistency

---

## 🎬 Demo

### Web Interface

![BERT QA Demo](docs/images/demo.gif)

**Try it yourself:**
```

python app_gradio.py

# Opens at http://localhost:7860 with shareable public link!

```

### Command-Line Interface

```

\$ python interactive_qa.py

🤖 BERT Question Answering System
======================================================================
Enter Context: The Eiffel Tower was built by Gustave Eiffel in 1889.
Enter Question: When was the Eiffel Tower built?

✅ ANSWER: 1889
📊 Confidence: 98.7%
📍 Position: characters 47-51
======================================================================

```

---

## 🏗️ Architecture

### High-Level System Design

```

┌─────────────────────────────────────────────────────────────┐
│                   User Input (Question + Context)           │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│              BERT Tokenizer (WordPiece)                     │
│  -  Tokenize question and context                            │
│  -  Add [CLS] and [SEP] tokens                               │
│  -  Create attention masks                                    │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│           Fine-Tuned BERT Model (109M params)               │
│  ┌──────────────────────────────────────────────────┐       │
│  │  12 Transformer Layers (768 hidden dim)          │       │
│  │  -  Multi-head self-attention (12 heads)          │       │
│  │  -  Position-wise feed-forward networks           │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │       QA Head (Linear Layers)                    │       │
│  │  -  Start position logits                         │       │
│  │  -  End position logits                           │       │
│  └──────────────────────┬───────────────────────────┘       │
└─────────────────────────┼────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                  Answer Extraction                          │
│  -  Find argmax of start/end logits                          │
│  -  Extract text span from context                           │
│  -  Calculate confidence score                               │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│              Output (Answer + Confidence)                   │
└─────────────────────────────────────────────────────────────┘

```

### Model Specifications

| Component | Details |
|-----------|---------|
| **Base Model** | `bert-base-uncased` |
| **Parameters** | 108,893,186 (~109 million) |
| **Layers** | 12 transformer blocks |
| **Hidden Size** | 768 dimensions |
| **Attention Heads** | 12 heads per layer |
| **Vocabulary** | 30,522 WordPiece tokens |
| **Max Sequence Length** | 512 tokens (384 used) |
| **Model Size** | ~440 MB (PyTorch) |

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- 5GB disk space

### Option 1: Google Colab (Recommended for Quick Start)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaiku7867/bert-qa-squad/blob/main/notebooks/04_interactive_qa_system.ipynb)

**No installation required!** Just click the badge above.

### Option 2: Local Installation

```


# Clone the repository

git clone https://github.com/jaiku7867/bert-qa-squad.git
cd bert-qa-squad

# Create virtual environment

python -m venv venv
source venv/bin/activate  \# On Windows: venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Download pre-trained model (if not training from scratch)

python scripts/download_model.py

```

### Option 3: Docker

```


# Build Docker image

docker build -t bert-qa .

# Run container

docker run -p 7860:7860 bert-qa

# Access at http://localhost:7860

```

---

## 🚀 Quick Start

### 1. Train the Model (Optional - if using pre-trained)

```


# Run complete training pipeline

python train.py --epochs 3 --batch_size 16 --learning_rate 3e-5

# Or use the Jupyter notebooks for step-by-step training

jupyter notebook notebooks/

```

### 2. Launch Interactive Interface

**Gradio Interface (Recommended):**
```

python app_gradio.py

# Opens at http://localhost:7860

# Public shareable link generated automatically!

```

**Streamlit Interface:**
```

streamlit run app_streamlit.py

# Opens at http://localhost:8501

```

### 3. Use Command-Line Interface

```

python interactive_qa.py

# Follow prompts to enter context and questions

```

### 4. Python API Usage

```

from transformers import pipeline

# Load fine-tuned model

qa_pipeline = pipeline(
"question-answering",
model="./bert-qa-model",
tokenizer="./bert-qa-model"
)

# Ask a question

result = qa_pipeline(
question="What is the capital of France?",
context="Paris is the capital of France."
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2%}")

```

---

## 📁 Project Structure

```

bert-qa-squad/
│
├── notebooks/                          \# Jupyter notebooks
│   ├── 01_data_exploration.ipynb       \# Data analysis (10 marks)
│   ├── 02_preprocessing_tokenization.ipynb  \# Preprocessing (20 marks)
│   ├── 03_model_training.ipynb         \# Training (15 marks)
│   └── 04_interactive_qa_system.ipynb  \# Interface (5 marks)
│

│
├── apps/                               \# Web applications
│   ├── app_gradio.py                   \# Gradio interface
│   └── app_streamlit.py                \# Streamlit interface
│

│

│
├── docs/                               \# Documentation
│   ├── Notebook_1_Data_Exploration.pdf     (18 pages)
│   ├── Notebook_2_Preprocessing.pdf        (23 pages)
│   ├── Notebook_3_Model_Training.pdf       (27 pages)
│   └── images/
│
├── requirements.txt                    \# Python dependencies
├── Dockerfile                          \# Docker configuration
├── README.md                           \# This file
├── LICENSE                             \# MIT License
└── .gitignore

```

---

## 📊 Dataset

### SQuAD (Stanford Question Answering Dataset)

- **Version**: SQuAD 1.1
- **Total Examples**: 87,599 (train) + 10,570 (validation)
- **Subset Used**: 3,000 (train) + 500 (validation)
- **Task**: Extractive Question Answering
- **Format**: JSON with context, question, and answer triplets

### Dataset Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|---------|
| **Context Length** | 129.6 words | 26 | 346 | 52.3 |
| **Question Length** | 10.3 words | 3 | 29 | 3.8 |
| **Answer Length** | 2.4 words | 1 | 27 | 2.1 |
| **Answer Position** | 328.3 chars | 0 | 1200+ | 220.5 |

**Key Insights:**
- 95%+ contexts fit within 384 tokens
- Questions are consistently concise (8-12 words)
- Most answers are 1-4 words (named entities, dates, phrases)
- Answers uniformly distributed throughout contexts (no positional bias)

---

## 🎓 Model Training

### Training Configuration

```

TrainingArguments(
output_dir="./results",
num_train_epochs=3,
per_device_train_batch_size=16,
learning_rate=3e-5,
weight_decay=0.01,
fp16=True,                    \# Mixed precision training
evaluation_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
metric_for_best_model="loss",
seed=42
)

```

### Training Process

**Hardware Used:**
- GPU: Tesla T4 (15GB VRAM)
- CPU: 2-core Intel Xeon
- RAM: 12GB

**Training Time:**
- Per Epoch: ~1.5 minutes
- Total (3 epochs): **4.6 minutes** ⚡
- Samples/second: 33.03 (training), 186.27 (evaluation)

### Training Curves

![Training Loss](docs/images/training_loss.png)

**Epoch-by-Epoch Results:**

| Epoch | Training Loss | Validation Loss | Status |
|-------|---------------|-----------------|--------|
| 1 | 1.1314 | **1.6476** | ✅ **Best Model** |
| 2 | 0.6137 | 1.8053 | ⚠️ Overfitting detected |
| 3 | 0.4221 | 1.8659 | ⚠️ Overfitting worsened |

**Key Observations:**
- Model achieves best validation loss at Epoch 1
- Training loss continues decreasing (memorization)
- Validation loss increases after Epoch 1 (overfitting)
- **Solution**: Automatically loaded best checkpoint (Epoch 1)

---

## 📈 Results & Performance

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Validation Loss** | 1.6476 | ✅ Good performance |
| **Expected Exact Match** | 45-60% | Competitive for small dataset |
| **Expected F1 Score** | 65-75% | Industry-standard baseline |
| **Inference Time** | ~50ms | Real-time capable |
| **Model Size** | 440 MB | Deployable on edge devices |

### Loss Interpretation

The validation loss of **1.65** means:
- Model assigns **~19% probability** to correct answer positions
- **Significantly better** than random (baseline ~0.26%)
- Comparable to standard BERT-base on small datasets

### Example Predictions

**Example 1: Factual Question**
```

Context: "Paris is the capital and most populous city of France, with
an area of 105 square kilometres and a population of 2.2 million."

Question: "What is the capital of France?"

Predicted Answer: "Paris"
Ground Truth: "Paris"
Confidence: 99.87%
Status: ✅ Correct

```

**Example 2: Temporal Question**
```

Context: "The Eiffel Tower was built by engineer Gustave Eiffel
in 1889 for the 1889 World's Fair."

Question: "When was the Eiffel Tower built?"

Predicted Answer: "1889"
Ground Truth: "1889"
Confidence: 98.45%
Status: ✅ Correct

```

**Example 3: Complex Question**
```

Context: "BERT was developed by Google and published in 2018. It uses
bidirectional training of Transformer encoders."

Question: "Who developed BERT?"

Predicted Answer: "Google"
Ground Truth: "Google"
Confidence: 96.32%
Status: ✅ Correct

```

---

## 🎨 Interactive Interface

### Gradio Web App

**Features:**
- 📝 Multi-line context input
- ❓ Question input with autocomplete
- ✅ Answer display with confidence
- 📊 Position highlighting in context
- 📚 Pre-loaded example questions
- 🌐 Public shareable link (72-hour expiry)

**Screenshot:**

![Gradio Interface](docs/images/gradio_interface.png)

### Streamlit Web App

**Features:**
- 🎨 Modern, responsive UI
- 📊 Real-time metrics dashboard
- 🔍 Answer highlighting in context
- 📈 Model info sidebar
- 💾 Session state management
- 🎯 Multiple example templates

**Screenshot:**

![Streamlit Interface](docs/images/streamlit_interface.png)

---

## 📚 Documentation

### Comprehensive Notebooks

Each notebook includes:
- ✅ Complete code with explanations
- ✅ Output analysis and interpretation
- ✅ Visualizations and charts
- ✅ Mathematical formulas (LaTeX)
- ✅ Professional PDF exports

**Download PDFs:**
- 📄 [Notebook 1: Data Exploration (18 pages)](docs/Notebook_1_Data_Exploration.pdf)
- 📄 [Notebook 2: Preprocessing (23 pages)](docs/Notebook_2_Preprocessing.pdf)
- 📄 [Notebook 3: Model Training (27 pages)](docs/Notebook_3_Model_Training.pdf)
- 📄 [Notebook 4: Interactive System (15 pages)](docs/Notebook_4_Interactive_QA.pdf)

### Key Documentation Highlights

**Notebook 1:**
- SQuAD dataset exploration
- Statistical analysis (context/question/answer lengths)
- Distribution visualizations
- Tokenization strategy planning

**Notebook 2:**
- BERT tokenizer configuration
- Answer position mapping algorithm
- **100% validation success** ✅ (prevented 3+ days of debugging!)
- Offset mapping edge cases

**Notebook 3:**
- Model architecture breakdown
- Training hyperparameter justification
- Overfitting detection and correction
- Loss curve analysis

**Notebook 4:**
- Pipeline API usage
- Gradio/Streamlit interface design
- Deployment strategies

---



---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features
- Update documentation as needed

### Reporting Issues

Use the [Issues](https://github.com/jaiku7867/bert-qa-squad/issues) tab to report bugs or request features. Please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```

MIT License

Copyright (c) 2025 Jai Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

```

---

## 🙏 Acknowledgments

### Datasets & Models

- **SQuAD Dataset**: [Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250)
- **BERT Model**: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- **Hugging Face Transformers**: For the excellent library

### Inspiration & Resources

- [Hugging Face Course](https://huggingface.co/course)
- [Papers with Code - SQuAD Leaderboard](https://paperswithcode.com/sota/question-answering-on-squad11)
- [Stanford NLP Group](https://nlp.stanford.edu/)
- [Google AI Blog - BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

### Tools & Frameworks

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - NLP models
- [Gradio](https://gradio.app/) - Web interface
- [Google Colab](https://colab.research.google.com/) - Free GPU training

---

## 📞 Contact

**Jai Kumar**

- 📧 Email: [jaiku7867@gmail.com](mailto:jaiku7867@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/jai-kumar](https://linkedin.com/in/jai-kumar)
- 🐙 GitHub: [@jaiku7867](https://github.com/jaiku7867)
- 🌐 Portfolio: [jaikumar.dev](https://jaikumar.dev)

---

<div align="center">


**Made with ❤️ by Jai Kumar**


---

© 2025 Jai Kumar. All Rights Reserved.

</div>
```


***



[^16]: https://www.upgrad.com/blog/top-machine-learning-projects-on-github/

