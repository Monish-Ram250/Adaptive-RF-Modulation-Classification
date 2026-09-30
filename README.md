Sure bro. Since you **don't have a license**, I've removed all license references. This version is detailed, professional, and ready to paste directly into `README.md`.

````markdown
# Adaptive RF Modulation Classification

<p align="center">
  <b>Adaptive RF Modulation Classification Using Residual Denoising and Raw–Denoised Signal Fusion with Residual CNN–Transformer–Attention Learning</b>
</p>

<p align="center">
  Research Project • ICCIS 2026 • Paper ID: 925
</p>

---

## 📌 Overview

Automatic Modulation Classification (AMC) is an important task in wireless communication and intelligent RF signal analysis. It aims to identify the modulation scheme of a received RF signal without requiring prior knowledge of the transmitter.

However, noise present in received RF signals can make modulation classification more difficult, especially at lower Signal-to-Noise Ratio (SNR) levels.

This project proposes an adaptive RF signal recognition pipeline that combines:

- Residual Denoising Autoencoder (RDAE)
- Raw–Denoised Signal Fusion
- Residual 1D Convolutional Neural Network (CNN)
- Transformer Encoder
- Attention Mechanism
- SNR Feature Branch

The proposed approach first learns to suppress noise using a Residual Denoising Autoencoder. Instead of discarding the original raw signal after denoising, the raw and denoised representations are combined through weighted fusion. The resulting representation is then processed by a Residual CNN–Transformer–Attention classifier for modulation recognition.

---

# 📄 Research Paper

### Paper Title

**Adaptive RF Modulation Classification Using Residual Denoising and Raw–Denoised Signal Fusion with Residual CNN–Transformer–Attention Learning**

### Paper ID

**925**

### Author

**Akula Monish Ram**

### Affiliation

**Lovely Professional University**

### Conference

**8th International Conference on Communication and Intelligent Systems (ICCIS 2026)**

### Conference Venue

**BITS Pilani, K K Birla Goa Campus, Goa, India**

### Conference Date

**September 26–27, 2026**

### Publication Status

- ✅ Paper accepted
- ✅ Camera-ready manuscript submitted
- ✅ Research paper presented at ICCIS 2026
- ⏳ Proceedings publication pending

The paper was accepted for presentation at ICCIS 2026 and was subsequently presented at the conference.

> **Note:** The manuscript included in this repository is the final/camera-ready manuscript associated with the conference submission. The paper should not be considered officially published until the conference proceedings are released by the publisher.

---

# 🎯 Research Objective

The primary objective of this work is to improve RF modulation classification under noisy signal conditions.

The proposed system focuses on three main ideas:

1. **Noise suppression** using a Residual Denoising Autoencoder.
2. **Information preservation** by combining the original raw signal with the denoised representation.
3. **Robust feature learning** using Residual CNN, Transformer Encoder, and Attention mechanisms.

The overall objective is to allow the classifier to learn both local and long-range characteristics of RF signals while retaining useful information from the original noisy representation.

---

# 🏗️ Overall Architecture

The complete pipeline is:

```text
CLEAN RF SIGNAL
       │
       ▼
   Add AWGN
       │
       ▼
NOISY / RAW SIGNAL
       │
       ▼
┌─────────────────────────────┐
│          RDAE               │
│                             │
│ Encoder Stem                │
│      ↓                      │
│ 3 Residual Encoder Blocks   │
│      ↓                      │
│ Residual Bottleneck         │
│      ↓                      │
│ 2 Residual Decoder Blocks   │
│      ↓                      │
│ Residual Head               │
└─────────────┬───────────────┘
              │
              ▼
     Residual Correction
              │
              ▼
 Xdenoised = Xraw + R(Xraw)
              │
              ▼
      DENOISED SIGNAL
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
   RAW SIGNAL   DENOISED
        │           │
        └─────┬─────┘
              ▼
      WEIGHTED FUSION
              │
              ▼
 Xfused = 0.40 Xdenoised
        + 0.60 Xraw
              │
              ▼
 Raw (2×128) + Fused (2×128)
              │
              ▼
       FINAL 4×128 INPUT
              │
              ▼
┌─────────────────────────────┐
│       CLASSIFIER            │
│                             │
│ Residual CNN Block 1        │
│         ↓                   │
│ Residual CNN Block 2        │
│         ↓                   │
│ Residual CNN Block 3        │
│         ↓                   │
│ Transformer Encoder 1       │
│         ↓                   │
│ Transformer Encoder 2       │
│         ↓                   │
│ Attention Layer             │
└─────────────┬───────────────┘
              │
              ├──────── SNR Branch
              │
              ▼
       Feature Concatenation
              │
              ▼
          Classifier
              │
              ▼
   MODULATION CLASS PREDICTION
````

---

# 📡 1. RF Signal Representation

The project works with RF signals represented using **In-phase (I)** and **Quadrature (Q)** components.

Each original signal contains:

```text
2 × 128
```

where:

* 2 → I and Q channels
* 128 → samples per signal

---

# 🌫️ 2. AWGN Noise Generation

To evaluate the robustness of the system under noisy conditions, noise is introduced into the clean RF signal.

The noisy signal can be represented as:

$$
X_{raw}=X_{clean}+N_{AWGN}
$$

where:

* \(X_{clean}\) = clean RF signal
* \(N_{AWGN}\) = Additive White Gaussian Noise
* \(X_{raw}\) = noisy/raw RF signal

AWGN provides a controlled way to evaluate model performance at different SNR levels.

The clean signal is retained as the target for the denoising stage.

---

# 🧹 3. Residual Denoising Autoencoder (RDAE)

The first major component is the **Residual Denoising Autoencoder**.

The purpose of the RDAE is to learn a residual correction that transforms the noisy/raw RF signal into a representation closer to the clean signal.

Instead of directly learning:

```text
Noisy Signal → Clean Signal
```

the RDAE learns:

```text
Noisy Signal → Residual Correction
```

The final output is:

$$
X_{denoised}=X_{raw}+R(X_{raw})
$$

where \(R(X_{raw})\) represents the learned residual correction.

---

## RDAE Architecture

The RDAE consists of:

### Encoder

* Encoder Stem
* Residual Encoder Block 1
* Residual Encoder Block 2
* Residual Encoder Block 3

### Bottleneck

* Residual Bottleneck Block

### Decoder

* Upsampling Stage 1
* Residual Decoder Block 1
* Upsampling Stage 2
* Residual Decoder Block 2
* Upsampling Stage 3

### Output

* Residual Head
* Residual correction
* Residual addition with the original input

### Residual Block Count

```text
3 Residual Encoder Blocks
+
1 Residual Bottleneck Block
+
2 Residual Decoder Blocks
=
6 Residual Blocks
```

---

# 🔗 4. Residual Connections in RDAE

Residual connections are used to preserve useful information from the original signal while allowing the network to learn the required correction.

The main residual formulation is:

$$
X_{denoised}=X_{raw}+R(X_{raw})
$$

The raw signal is therefore used as the base signal, while the neural network learns the correction.

This also provides a direct path for information and gradients through the residual blocks.

---

# 🔀 5. Raw–Denoised Signal Fusion

After the RDAE produces the denoised signal, the original raw signal is **not discarded**.

Instead, both representations are combined using weighted fusion:

$$
X_{fused}=0.40X_{denoised}+0.60X_{raw}
$$

Therefore:

* Denoised signal contribution = **40%**
* Raw signal contribution = **60%**

The purpose is to retain information from both representations.

The raw signal is used twice in the overall pipeline, but for different purposes:

### First use

Inside the RDAE:

$$
X_{denoised}=X_{raw}+R(X_{raw})
$$

Purpose:

**Residual reconstruction / denoising**

### Second use

During fusion:

$$
X_{fused}=0.40X_{denoised}+0.60X_{raw}
$$

Purpose:

**Preserving complementary raw signal information**

---

# 🧩 6. Final Classifier Input

The original raw representation has:

```text
2 × 128
```

The fused representation also has:

```text
2 × 128
```

They are concatenated:

```text
Raw Signal
2 × 128
     +
Fused Signal
2 × 128
     ↓
Final Input
4 × 128
```

Therefore, the classifier receives a **4-channel, 128-sample input**.

---

# 🧠 7. Residual 1D CNN

The classifier begins with three Residual 1D CNN blocks.

The CNN is responsible for extracting local patterns from the RF signal representation.

---

## Residual CNN Block 1

```text
Input Channels  : 4
Output Channels : 64
Kernel Size     : 7
Dropout         : 0.15
Pooling         : Yes
```

Contains:

* Conv1D
* Batch Normalization
* ReLU
* Conv1D
* Batch Normalization
* Residual/Shortcut Connection
* ReLU
* Max Pooling

---

## Residual CNN Block 2

```text
Input Channels  : 64
Output Channels : 128
Kernel Size     : 5
Dropout         : 0.20
Pooling         : Yes
```

Contains:

* Conv1D
* Batch Normalization
* ReLU
* Conv1D
* Batch Normalization
* Residual/Shortcut Connection
* ReLU
* Max Pooling

---

## Residual CNN Block 3

```text
Input Channels  : 128
Output Channels : 256
Kernel Size     : 3
Dropout         : 0.25
Pooling         : No
```

Contains:

* Conv1D
* Batch Normalization
* ReLU
* Conv1D
* Batch Normalization
* Residual/Shortcut Connection
* ReLU

---

## CNN Summary

```text
3 Residual 1D CNN Blocks
        ↓
2 Conv1D Layers per Block
        ↓
6 Conv1D Layers Total
```

| Component               | Value |
| ----------------------- | ----: |
| Residual CNN Blocks     |     3 |
| Conv1D Layers per Block |     2 |
| Total Conv1D Layers     |     6 |
| Final CNN Channels      |   256 |
| Max Pooling Layers      |     2 |

---

# 🔭 8. Transformer Encoder

After the CNN feature extraction stage, the feature representation is passed to the Transformer encoder.

The Transformer helps capture relationships across the learned sequence representation.

The model contains **2 Transformer Encoder blocks**.

---

## Transformer Configuration

| Parameter                  | Value |
| -------------------------- | ----: |
| Transformer Encoder Blocks |     2 |
| Embedding Dimension        |   256 |
| Attention Heads            |     8 |
| Dimension per Head         |    32 |
| Feed-Forward Dimension     |   512 |
| Dropout                    |  0.25 |

The embedding dimension is:

$$
d_{model}=256
$$

With 8 attention heads:

$$
256/8=32
$$

Therefore, each attention head operates on a **32-dimensional representation**.

---

# 🎯 9. Attention Layer

After the two Transformer encoder blocks, a custom attention layer is used to generate a weighted feature representation.

The attention mechanism follows:

```text
256
 ↓
Linear 256 → 128
 ↓
Tanh
 ↓
Linear 128 → 1
 ↓
Softmax
 ↓
Weighted Feature Aggregation
 ↓
256-D Context Vector
```

The resulting context representation has **256 dimensions**.

---

# 📡 10. SNR Feature Branch

The SNR value is processed separately through a small neural network.

```text
SNR
 ↓
Linear 1 → 16
 ↓
ReLU
 ↓
Linear 16 → 32
```

The SNR branch produces a:

```text
32-dimensional feature vector
```

---

# 🔗 11. Feature Concatenation

The attention output and SNR features are combined.

```text
Attention Features = 256
SNR Features       = 32
                     ↓
               Concatenation
                     ↓
                   288
```

Therefore, the final feature vector given to the classification head is:

$$
256+32=288
$$

---

# 🏷️ 12. Final Classification Head

The final classifier consists of:

```text
288
 ↓
Linear 288 → 256
 ↓
ReLU
 ↓
Dropout 0.40
 ↓
Linear 256 → 128
 ↓
ReLU
 ↓
Dropout 0.30
 ↓
Linear 128 → 11
 ↓
Modulation Prediction
```

The final layer produces predictions for **11 modulation classes**.

---

# 📊 Dataset

The experiments use the **RadioML 2016.10a** dataset.

### Dataset Information

| Property              |            Value |
| --------------------- | ---------------: |
| Total Samples         |          220,000 |
| Modulation Classes    |               11 |
| Samples per Signal    |              128 |
| Signal Representation |              I/Q |
| SNR Range             | −20 dB to +18 dB |
| Experimental Focus    |      SNR ≥ −8 dB |

The dataset provides a controlled benchmark for evaluating automatic modulation classification under different noise conditions.

---

# ⚙️ Implementation

The project is implemented using **Python and PyTorch**.

### Main Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

### Training Environment

* Google Colab
* NVIDIA Tesla T4 GPU

---

# 🏋️ Training Configuration

| Parameter        | Configuration                           |
| ---------------- | --------------------------------------- |
| Framework        | PyTorch                                 |
| Optimizer        | AdamW                                   |
| Learning Rate    | 5 × 10⁻⁴                                |
| Batch Size       | 256                                     |
| Epochs           | 50                                      |
| Weight Decay     | 1 × 10⁻⁴                                |
| Dropout          | 0.10 / 0.15 / 0.20 / 0.25 / 0.30 / 0.40 |
| Mixed Precision  | AMP                                     |
| Gradient Scaling | GradScaler                              |
| Scheduler        | ReduceLROnPlateau                       |

---

# ⚡ Automatic Mixed Precision

Automatic Mixed Precision (AMP) is used during training to improve computational efficiency.

The training process uses a mixture of lower and higher precision operations where appropriate.

GradScaler is used to maintain numerical stability during mixed-precision training by scaling the loss and gradients before the optimizer update.

The model parameters are ultimately updated using the gradients through the AdamW optimizer.

---

# 📈 Main Results

The proposed model achieved the following results:

| Metric               |              Result |
| -------------------- | ------------------: |
| Accuracy             |          **87.20%** |
| Macro Precision      |            **0.89** |
| Macro Recall         |            **0.87** |
| Macro F1             |            **0.87** |
| Weighted F1          |            **0.87** |
| Trainable Parameters |          **~1.69M** |
| Inference Time       |   **5.75 ms/frame** |
| Approx. Throughput   | **~174 frames/sec** |

The inference measurement was obtained using an NVIDIA Tesla T4 GPU.

---

# 🧪 Ablation Study

An ablation study was conducted to evaluate the effect of different architectural components.

| Model                                      |   Accuracy |
| ------------------------------------------ | ---------: |
| Baseline CNN                               |     74.10% |
| CNN–BiLSTM–Attention                       |     81.42% |
| CNN–BiLSTM–Attention + Fusion              |     86.63% |
| Proposed Residual CNN–Transformer + Fusion | **87.20%** |

The ablation results show the performance progression from the baseline architecture to the proposed architecture.

---

# 🖼️ Results and Visualizations

The `results/` directory contains the main visual outputs from the experiments.

Expected files include:

```text
results/
│
├── confusion_matrix.png
├── accuracy_vs_snr.png
├── denoising_results.png
└── ablation_results.png
```

### Confusion Matrix

Shows the classification performance across the modulation classes.

### Accuracy vs. SNR

Shows how classification performance changes with different SNR levels.

### Denoising Results

Shows examples of clean, noisy, and reconstructed RF signals.

### Ablation Results

Shows the comparison between the baseline and different architectural configurations.

---

# 📁 Repository Structure

```text
Adaptive-RF-Modulation-Classification/
│
├── README.md
│
├── RF_Modulation_Classification.ipynb
│
├── ICCIS_2026_Final_Manuscript_Paper_925.pdf
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_vs_snr.png
│   ├── denoising_results.png
│   └── ablation_results.png
│
└── requirements.txt
```

---

# 🚀 How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/Monish-Ram250/Adaptive-RF-Modulation-Classification.git
```

## 2. Navigate to the Repository

```bash
cd Adaptive-RF-Modulation-Classification
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Prepare the Dataset

The RadioML 2016.10a dataset is **not included in this repository**.

Obtain the dataset separately and configure the dataset path according to the notebook.

## 5. Open the Notebook

Open:

```text
RF_Modulation_Classification.ipynb
```

The notebook contains the implementation and experimental workflow.

---

# ⚠️ Limitations

The current study has several limitations.

### 1. AWGN-focused evaluation

The current experiments primarily focus on AWGN-based noise conditions.

### 2. Real-world wireless channels

Real wireless environments can contain additional effects such as:

* Rayleigh fading
* Frequency offsets
* Phase offsets
* Hardware impairments
* Other channel effects

These are not fully evaluated in the current study.

### 3. Over-the-Air Validation

The current work does not include complete over-the-air validation using SDR hardware.

### 4. Dataset Scope

The experiments focus on RadioML 2016.10a.

Additional datasets can provide further validation.

---

# 🔮 Future Work

Potential future directions include:

* Evaluation under Rayleigh fading
* Additional wireless channel models
* Hardware impairment modeling
* Software Defined Radio (SDR) validation
* Over-the-air experiments
* Hardware-in-the-loop testing
* Evaluation on RadioML 2018.01a
* Model compression
* Quantization
* Deployment on resource-constrained hardware
* Evaluation using real-world RF signals

---

# 📄 Conference Presentation

This research was presented at:

**8th International Conference on Communication and Intelligent Systems (ICCIS 2026)**

### Conference Details

```text
Conference : ICCIS 2026
Paper ID   : 925
Venue      : BITS Pilani, K K Birla Goa Campus
Date       : September 26–27, 2026
Author     : Akula Monish Ram
Affiliation: Lovely Professional University
```

### Current Status

```text
✅ Research completed
✅ Paper submitted
✅ Paper accepted
✅ Camera-ready manuscript submitted
✅ Conference presentation completed
⏳ Proceedings publication pending
```

---

# 📑 Manuscript

The final/camera-ready manuscript associated with this research is included in this repository:

```text
ICCIS_2026_Final_Manuscript_Paper_925.pdf
```

The manuscript contains the detailed methodology, experiments, results, analysis, and conclusions associated with this work.

---

# 👨‍💻 Author

**Akula Monish Ram**

B.Tech Computer Science and Engineering
Lovely Professional University

---

# 🙏 Acknowledgement

I would like to express my sincere gratitude to my research mentor and faculty members for their continuous guidance, support, and valuable feedback throughout this research work.

I also thank the organizers of **ICCIS 2026** for providing the opportunity to present this research and engage in technical discussions with researchers and academicians from different backgrounds.

---

# 📚 Citation

If you reference this research, please cite the conference paper as:

```text
Akula Monish Ram,
"Adaptive RF Modulation Classification Using Residual Denoising
and Raw–Denoised Signal Fusion with Residual CNN–Transformer–
Attention Learning,"
ICCIS 2026, Paper ID 925.
```

---

# 📌 Project Status

**Accepted & Presented at ICCIS 2026**

This repository contains the implementation, experimental notebook, results, visualizations, and final manuscript associated with the research work.

The paper has been accepted and presented at ICCIS 2026. The official proceedings publication information will be added once the publisher releases the proceedings.

---

## ⭐ Keywords

```text
Automatic Modulation Classification
RF Signal Processing
Radio Frequency
RDAE
Residual Denoising
Denoising Autoencoder
Raw-Denoised Fusion
1D CNN
Residual CNN
Transformer Encoder
Attention Mechanism
SNR
RadioML 2016.10a
Deep Learning
Wireless Communication
Signal Classification
```

```

### One important recommendation

Since you're **not adding a license**, this README deliberately has **no `LICENSE` section** and does not claim that the code is freely reusable.

Also, I kept the publication wording as **“Accepted & Presented — Proceedings publication pending”** rather than saying the paper is already published. That is the safest wording for your current status.
```
