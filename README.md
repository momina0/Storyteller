# Neural Storyteller - Image Captioning with Seq2Seq

A deep learning project that generates natural language descriptions for images using a Sequence-to-Sequence architecture with LSTM networks.

## 📋 Overview

This project implements an image captioning system that converts visual content into descriptive text. The model uses a pre-trained ResNet50 for feature extraction and a Seq2Seq architecture with LSTM for caption generation.

## ✨ Features

- **Dual GPU Training**: Optimized for Kaggle T4 x2 GPUs with DataParallel
- **Efficient Feature Extraction**: Pre-cached image features using ResNet50
- **Advanced Decoding**: Greedy search and beam search caption generation
- **Comprehensive Evaluation**: BLEU, METEOR, ROUGE, Precision, Recall, F1-score
- **Interactive Demo**: Gradio web interface for real-time caption generation
- **Cloud Deployment**: Ready-to-deploy on Hugging Face Spaces

## 🗂️ Dataset

**Flickr30k**: A dataset containing 31,783 images with 5 captions each (158,915 total captions)

- Training: 80%
- Validation: 10%
- Testing: 10%

## 🏗️ Model Architecture

### Encoder

- **Input**: 2048-dim feature vectors from ResNet50
- **Architecture**: Single linear layer projecting to 512-dim hidden space
- **Output**: Encoded image representation

### Decoder

- **Embedding Layer**: 256-dim word embeddings
- **LSTM**: 1-layer LSTM with 512 hidden units
- **Vocabulary**: ~3000-5000 words (frequency threshold: 5)
- **Special Tokens**: `<pad>`, `<start>`, `<end>`, `<unk>`

### Caption Generation

- **Greedy Search**: Fast, deterministic decoding
- **Beam Search**: Better quality with configurable beam width (default: 5)

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd genasses/1
```

2. Install dependencies:

```bash
pip install torch torchvision tqdm pillow pandas scikit-learn nltk rouge-score gradio matplotlib
```

3. Download the Flickr30k dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr30k)

## 📊 Training

### On Kaggle

1. Upload the notebook `neural_storyteller_base.ipynb` to Kaggle
2. Add the Flickr30k dataset to your notebook
3. Enable GPU T4 x2 accelerator
4. Run all cells sequentially

### Training Configuration

```python
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
MAX_CAPTION_LENGTH = 50
VOCAB_THRESHOLD = 5
```

### Training Pipeline

1. **Feature Extraction** (~10-15 min): Extract and cache ResNet50 features
2. **Vocabulary Building**: Process captions and build vocabulary
3. **Model Training** (~2-3 hours): Train Seq2Seq model
4. **Evaluation**: Calculate metrics on test set
5. **Deployment**: Generate deployment files for Hugging Face Spaces

## 📈 Evaluation Metrics

The model is evaluated using multiple metrics:

- **BLEU-1/2/3/4**: N-gram overlap between predicted and ground truth captions
- **METEOR**: Considers synonyms and stemming
- **ROUGE-1/2/L**: Recall-oriented metrics for text generation
- **Token-level**: Precision, Recall, F1-score

## 🎯 Usage

### Generate Caption for Single Image

```python
import torch
from PIL import Image

# Load model and vocabulary
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Load and process image
image = Image.open('path/to/image.jpg')
# ... (preprocessing steps)

# Generate caption
caption = model.beam_search(features, vocab)
print(f"Caption: {caption}")
```

### Interactive Demo

```bash
# Run locally
python -c "import gradio as gr; from app import demo; demo.launch()"
```

## 🌐 Deployment

### Deploy to Hugging Face Spaces

The notebook includes automatic deployment cells:

1. Run cell 29: Save vocabulary
2. Run cell 30: Install huggingface_hub
3. Run cell 31: Create deployment files
4. Run cell 32: Upload to Hugging Face Spaces

Enter your Hugging Face token and space name when prompted.

Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

## 📁 Project Structure

```
.
├── neural_storyteller_base.ipynb   # Main training notebook
├── best_model.pth                  # Trained model weights
├── vocab.pkl                       # Vocabulary object
├── flickr30k_features.pkl          # Cached image features
├── app.py                          # Gradio deployment app
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Key Components

### Files Generated During Training

- `flickr30k_features.pkl`: Pre-extracted ResNet50 features (~1-2 GB)
- `best_model.pth`: Model checkpoint with lowest validation loss
- `vocab.pkl`: Vocabulary mapping for encoding/decoding

### Deployment Files

- `app.py`: Standalone Gradio application
- `requirements.txt`: Package dependencies for deployment
- `README.md`: Hugging Face Space description
- find it on: https://huggingface.co/spaces/momina0/stoory
## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
Pillow>=9.0.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
nltk>=3.8
rouge-score>=0.1.2
tqdm>=4.65.0
matplotlib>=3.7.0
```

## 🎓 Model Details

- **Total Parameters**: ~10-15M (depending on vocabulary size)
- **Training Time**: ~2-3 hours on Kaggle T4 x2
- **Inference Speed**: ~0.5-1 sec per image (CPU), ~0.1 sec (GPU)
- **Model Size**: ~50-100 MB

## 🏆 Results

Expected performance metrics (varies by training run):

- BLEU-4: 0.15-0.25
- METEOR: 0.20-0.30
- ROUGE-L: 0.35-0.45

## 🔍 Troubleshooting

### DataParallel Loading Error

If you get a "module." prefix error when loading the model, the notebook automatically handles this by stripping the prefix during deployment.

### Out of Memory

- Reduce `BATCH_SIZE` from 64 to 32 or 16
- Reduce `MAX_CAPTION_LENGTH` from 50 to 30

### Slow Training

- Ensure GPU is enabled in Kaggle
- Pre-extract features before training (cell 3)
- Use fewer epochs for testing

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Implement attention mechanism
- Add more sophisticated beam search
- Experiment with different architectures (Transformers, CLIP)
- Improve vocabulary preprocessing
- Add data augmentation

## 📄 License

This project is for educational purposes. Please cite the Flickr30k dataset if you use this work.

## 🙏 Acknowledgments

- **Dataset**: Flickr30k by Peter Young et al.
- **Framework**: PyTorch
- **Platform**: Kaggle Notebooks
- **Deployment**: Hugging Face Spaces
- **Pre-trained Model**: ResNet50 from torchvision

## 📧 Contact

For questions or issues, please open an issue in this repository.

---

**Built with ❤️ using PyTorch and Gradio**

<img width="1910" height="893" alt="image" src="https://github.com/user-attachments/assets/69396d82-b69c-4292-acb1-4fb775297a0b" />

