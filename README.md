# NeurIPS 2025 Ariel Data Challenge – Solution Overview

This repository contains a solution for the **NeurIPS 2025 Ariel Data Challenge**, titled *“Advancing Exoplanetary Signal Extraction for the Ariel Space Telescope.”*  
In this competition, participants extract extremely faint exoplanetary signals (atmospheric spectra) from noisy simulated telescope data and provide uncertainty estimates.  
Official challenge links:
- https://www.kaggle.com/competitions/ariel-data-challenge-2025


---

## 🧠 Methodology and Core Techniques

This solution uses a hybrid pipeline of signal processing and deep learning to clean the data, detect transits, predict exoplanet spectra, and estimate uncertainties.

### 🔹 Savitzky–Golay Filtering
- Smooths the time-series using local polynomial fits.
- Preserves transit shapes while reducing high-frequency noise.

### 🔹 Moving Average Smoothing
- Suppresses spikes or instrument noise by averaging neighbors.
- Used together with SG-filter for robust signal cleaning.

### 🔹 Phase Detection & Polynomial Fitting
- Detects transit start/end points using gradient changes.
- Fits polynomial to out-of-transit baseline.
- Optimizes transit depth scale (*s*) using Nelder–Mead.

### 🔹 ResNet-Based Signal Prediction
- ResNet model predicts:
  - Coarse transit depth (first channel)
  - Full transmission spectrum (282 channels)
- Inputs include: polynomial signal, stellar metadata (Rs, i)
- Ensemble average improves robustness.

### 🔹 Linear Regression for Sigma Estimation
- Uses std-dev of predictions to estimate uncertainty (σ).
- Model: `linear_model_use2_to_sigma.joblib`
- Same σ applied to all channels for each planet.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `trainanpy.ipynb` | Model training on cleaned training data |
| `resnet-for-airs-finetuned.ipynb` | Final pipeline for test-time prediction |
| `data.py` | Polynomial-based transit model and preprocessing |
| `linear_model_use2_to_sigma.joblib` | Linear model to predict sigma from prediction std-dev |

---

## ⚙️ Installation and Dependencies

```bash
pip install torch numpy pandas scipy scikit-learn astropy tqdm pqdm matplotlib
```

Python 3.9+ recommended. GPU is highly recommended for model training and inference.

---

## 🚀 How to Use

### 1. Download Dataset
From the [Kaggle challenge page](https://www.kaggle.com/competitions/ariel-data-challenge-2025) and place in a folder:
```python
ROOT_PATH = "/your/local/data/path"
```

### 2. Preprocessing & Feature Extraction
Run notebooks directly — preprocessing (SG filter, moving average, polynomial model) is built-in.

### 3. Train Models (optional)
```python
# Inside trainanpy.ipynb
# Trains ResNet models + saves .pth weights
```

### 4. Run Inference
```python
# Inside resnet-for-airs-finetuned.ipynb
# Loads pretrained weights + predicts spectrum and sigma
```

Ensure:
- `linear_model_use2_to_sigma.joblib` is in root dir
- model weights are correctly referenced or trained

---

## 📤 Output Format

Final output: `submission.csv`

| planet_id | wl_1 | ... | wl_283 | sigma_1 | ... | sigma_283 |
|-----------|------|-----|--------|---------|-----|-----------|
| 1103775 | 0.016 | ... | 0.0159 | 0.000275 | ... | 0.000275 |

- `wl_*`: transit depth (atmospheric spectrum)
- `sigma_*`: corresponding uncertainty (σ)

---

## 🔁 Notes

- Code was developed for Kaggle kernel, paths may need adjusting.
- Memory-heavy: inference with CPU may be slow; GPU strongly recommended.
- Predictions depend on ResNet weights — for reproducibility, ensure correct versions are used.

---

## 📌 Conclusion

This repo provides a complete solution to the NeurIPS 2025 Ariel Data Challenge using classical signal processing and modern deep learning. It’s extensible to future exoplanet missions like ESA's Ariel, and may support further research in denoising astrophysical time-series.
