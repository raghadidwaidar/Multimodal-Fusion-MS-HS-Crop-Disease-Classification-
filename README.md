# Multimodal-Fusion-MS-HS-Crop-Disease-Classification-

# Automated Multimodal Crop Disease Diagnosis from Multimodal Remote Sensing Imagery

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Timm](https://img.shields.io/badge/timm-NeuralNetwork-blue)](https://github.com/huggingface/pytorch-image-models)
[![Competition](https://img.shields.io/badge/ICPR_2026-Competition-orange)](https://www.icpr2026.org/)

An advanced, end-to-end Deep Learning framework developed for the **ICPR 2026 Automated Multimodal Crop Disease Diagnosis Competition**. This repository implements a dual-branch late-fusion architecture featuring **Gated Cross-Modal Fusion** to dynamically weigh and integrate Multispectral (MS) and Hyperspectral (HS) remote sensing imagery for highly accurate crop pathogen classification (**Rust**, **Health**, and **Other**).

---

## ── Architecture Overview ──────────────────────────────────────────

The framework utilizes a specialized dual-branch topology to extract asynchronous spatial-spectral features before projecting them into a dynamically gated shared subspace:

MS  (5ch,  64×64)  ──► BandAdapter ──► EfficientNet-B0 ──► CBAM ──► Pool ──► ms_feat (1280)│Gated Cross-Modal Fusion│HS  (125ch, 32×32)                                                             │→ bilinear upsample → (125ch, 64×64) ──► SEResNet18 ──────────────► hs_feat (512)│Concat → MLP → 3 classes
### Key Technical Highlights:
* **Multispectral Branch:** Custom 5-to-3 channel `BandAdapter` combined with an `EfficientNet-B0` backbone and `CBAM` (Convolutional Block Attention Module) for advanced spatial-structural processing.
* **Hyperspectral Branch:** Custom modified `SEResNet18` accepting dense $125$-band cubes, reinforced with `Squeeze-and-Excitation (SE)` blocks to capture fine-grained chemical-spectral dependencies.
* **Gated Cross-Modal Fusion:** A learnable Contextual Softmax Gating mechanism that dynamically generates routing scalars ($g_{ms}, g_{hs}$) to weight modalities based on environmental noise or spatial quality.
* **Robust Regularization:** Integrates synchronized multimodal spatial augmentations, `Mixup` training ($\alpha=0.2$), `Label Smoothing` ($0.1$), and multi-perspective `Test-Time Augmentation (TTA)`.

---

## ── Repository Structure ───────────────────────────────────────────

── Experimental Setup & Hyperparameters ──────────────────────────HyperparameterOperational ValueInput ModalitiesMS: $5 \times 64 \times 64$ | HS: $$125 \times 32 \times 3$ (Upsampled to $$64 \times 6$)Batch Size16Optimization AlgorithmAdamW ($\beta_1=0.9, \beta_2=0.999$, Weight Decay = $10^{-4}$)Learning Rate ScheduleInitial $3 \times 10^{-4}$ decaying via Cosine Annealing to $10^{-6}$Gradient RegularizationMax Norm Clipping = 1.0Loss ModificationsCross-Entropy with 0.1 Label Smoothing + 50% Probable MixupValidation StrategyStratified 5-Fold Cross-Validation (Patience = 10 epochs)Inference Configuration5-Transform Test-Time Augmentation (TTA)── Empirical Results & Performance Analysis ──────────────────────1. Cross-Validation ResultsThe model demonstrated high stability and generalization across independent geographical partitions under Stratified 5-Fold Validation:Mean Validation Accuracy: 67.45% ± 2.43%Best Single Fold (Fold 2): 71.57% (Optimal checkpoint extracted at Epoch 28).2. Quantitative Testing & TTA UpliftThe optimal Fold 2 model checkpoint was evaluated against an independent, out-of-fold test set consisting of 90 paired samples:Standard Single-Pass Inference: 62.22% AccuracyTTA-Augmented Inference: 70.00% AccuracyNet Empirical TTA Improvement: +7.78%3. Per-Class Metrics (TTA-Augmented)Plaintext              precision    recall  f1-score   support

        Rust     0.6410    0.8333    0.7246        30
      Health     0.6087    0.4667    0.5283        30
       Other     0.8571    0.8000    0.8276        30

    accuracy                         0.7000        90
   macro avg     0.7023    0.7000    0.6935        90
4. Critical Error AnalysisPathological Protection: The network prioritized high sensitivity for active fungal spread, pushing Rust Recall to 83.33% (25/30 true positives detected).Open Challenges: A persistent cross-modal misclassification vector remains within the Health baseline class, where early physiological plant stress or micro-scale illumination shadows mimic the structural-spectral fingerprints of early-onset rust. TTA multi-perspective voting successfully mitigated part of this overlap, raising true positive detection from 12 to 14 samples.── Installation & Usage ──────────────────────────────────────────1. Clone the RepositoryBashgit clone [https://github.com/raghadidwaidar/Automated-Multimodal-Crop-Disease-Diagnosis.git](https://github.com/raghadidwaidar/Automated-Multimodal-Crop-Disease-Diagnosis.git)
cd Automated-Multimodal-Crop-Disease-Diagnosis
2. Install RequirementsBashpip install -r requirements.txt
3. Execution (Training & Evaluation)To trigger the complete Stratified 5-Fold Cross-Validation training loop:Bashpython train.py
To run independent test set evaluation utilizing the saved optimal model checkpoints with full TTA support:Bashpython evaluate.py
── Citation & Acknowledgements ────────────────────────────────────This work was compiled specifically for the automated agricultural diagnostics tracks at the International Conference on Pattern Recognition (ICPR 2026). We thank the Kaggle competition hosts for sourcing and curating the multimodal remote sensing dataset.
