# 🧠 Deep Learning Computer Vision

Two end-to-end deep learning projects built with PyTorch — covering image classification from scratch and real-world object detection on autonomous driving data.

---

## 📌 Projects Overview

| Task | Problem | Dataset | Best Result |
|---|---|---|---|
| Task 1 | Multi-class Image Classification | MNIST | 99.27% accuracy |
| Task 2 | Object Detection | KITTI Raw | mAP@0.5: 0.3166 |

---

## 🔢 Task 1 — CNN from Scratch on MNIST

Built a Convolutional Neural Network entirely from scratch using PyTorch and ran **5 systematic architectural experiments** to understand how design choices affect model behaviour — not just accuracy numbers, but training dynamics, overfitting risk, and convergence speed.

### Architecture Experiments

| Model | Test Accuracy | Key Change |
|---|---|---|
| Baseline CNN | 99.10% | 2 conv layers, ReLU, MaxPool |
| Wider CNN ⭐ | **99.27%** | 64/128 filters |
| Deeper CNN | 99.27% | 3 conv layers |
| BatchNorm CNN | 99.16% | Batch Normalisation |
| Dropout CNN | 99.26% | Dropout p=0.5 |
| LeakyReLU CNN | 99.03% | LeakyReLU α=0.1 |

**Best model: Wider CNN — 99.27% test accuracy**

The difference between a good model and a great one is rarely one big decision — it's a series of small, justified ones.

### Key Findings
- Increasing width (64/128 filters) and depth (3 conv layers) both improved accuracy
- Batch Normalisation provided the fastest early convergence
- Dropout showed the smallest train-test gap — best regularisation behaviour
- LeakyReLU offered no benefit over ReLU on this relatively simple dataset

### How to Run
```bash
# Open in Google Colab
# Runtime → Change runtime type → T4 GPU
# Run all cells top to bottom
```
Open `mnist_cnn/Task1_CNN_MNIST.ipynb` in Google Colab.

---

## 🚗 Task 2 — Fine-Tuning Faster R-CNN on KITTI

Fine-tuned a pre-trained **Faster R-CNN with ResNet-50 FPN backbone** (pre-trained on COCO) to detect **Cars, Pedestrians, and Cyclists** in real-world urban driving footage from the KITTI raw dataset.

### Dataset
- **18 video sequences** from a forward-facing camera on a car driving through German city streets
- Annotations in XML tracklet format with 3D bounding boxes projected to 2D using calibration matrices
- **Split at sequence level** (not frame level) to prevent data leakage from temporally similar frames

| Split | Sequences | Frames |
|---|---|---|
| Train | VideoOne, VideoFour, VideoFive, VideoSix, VideoSeven | 1,246 |
| Validation | VideoEight | 114 |
| Test | Video9 | 270 |

### Experiments

| Experiment | Strategy | Car AP | Ped AP | Cyclist AP | mAP@0.5 |
|---|---|---|---|---|---|
| Exp A | Full backbone freeze | 0.4823 | 0.1245 | 0.1876 | 0.2648 |
| Exp B | Partial freeze (layers 3-4 trainable) | 0.5134 | 0.1589 | 0.2213 | 0.2979 |
| Exp C ⭐ | Partial freeze + temporal averaging (T=3) | **0.5267** | **0.1734** | **0.2498** | **0.3166** |

**Best model: Exp C — Temporal Frame Averaging with Partial Freeze**

### Key Findings

**Freezing strategy matters more than most tutorials suggest.**
Unfreezing deeper backbone layers (3 and 4) allowed the model to adapt its higher-level feature representations to the driving domain — something a fully frozen backbone simply cannot do.

**Data splitting is not just a technicality.**
With sequential video data, splitting at the frame level leaks nearly identical frames into training and test sets, inflating results and hiding true generalisation ability. Sequence-level splitting is non-negotiable.

**Temporal context is underrated.**
Averaging 3 consecutive frames before passing them to the detector gave the model subtle motion cues that measurably improved detection of smaller, moving objects like pedestrians and cyclists — without any architectural changes.

### How to Run
```bash
# Install dependencies
pip install torch torchvision torchmetrics matplotlib pillow scikit-learn

# Update VIDEOS_ROOT in the script to your dataset path
# Line 24: VIDEOS_ROOT = r'your\path\to\videos'

# Run
python kitti_detection/task2_kitti_detection.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange)
![Torchvision](https://img.shields.io/badge/Torchvision-0.27-orange)

| Library | Purpose |
|---|---|
| PyTorch | Model building and training |
| Torchvision | Pre-trained models, transforms |
| Matplotlib | Learning curves, visualisation |
| NumPy | Numerical operations |
| Pillow | Image loading and resizing |
| Scikit-learn | Confusion matrix, classification report |
| XML ElementTree | KITTI tracklet XML parsing |

---

## 📁 Repository Structure

```
deep-learning-computer-vision/
│
├── mnist_cnn/
│   └── Task1_CNN_MNIST.ipynb       # Full Task 1 notebook (Google Colab)
│
├── kitti_detection/
│   └── task2_kitti_detection.py    # Full Task 2 script (local)
│
├── .gitignore                       # Excludes large model files
└── README.md
```

---

## 📖 References

- LeCun, Y. et al. (1998) Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), pp. 2278–2324.
- Ren, S. et al. (2015) Faster R-CNN: Towards real-time object detection with region proposal networks. *NeurIPS*, 28.
- He, K. et al. (2016) Deep residual learning for image recognition. *CVPR*, pp. 770–778.
- Ioffe, S. and Szegedy, C. (2015) Batch normalization. *ICML*, pp. 448–456.
- Srivastava, N. et al. (2014) Dropout. *Journal of Machine Learning Research*, 15(1), pp. 1929–1958.
- Geiger, A. et al. (2013) Vision meets robotics: The KITTI dataset. *International Journal of Robotics Research*, 32(11).
- Lin, T.Y. et al. (2017) Feature pyramid networks for object detection. *CVPR*, pp. 2117–2125.

---

## 👤 Author

**Tharuka Premasiri**
Software Engineer | Deep Learning | Computer Vision

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/tharukapremasiri)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/tharukapremasiri)

---
