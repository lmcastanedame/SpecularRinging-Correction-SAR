# 🛰️ SWOT-SAR-Superresolution

**Improving SWOT SAR images by specular ringing correction with superresolution learning.**  
This project proposes a hybrid pipeline combining spectral subband filtering and deep learning-based superresolution to suppress specular ringing artifacts in SWOT SAR imagery.

---

## 📌 Project Overview

SWOT (Surface Water and Ocean Topography) provides high-resolution SAR data but suffers from **specular ringing artifacts**, especially near strong reflectors like coastlines. These artifacts degrade the scientific quality of the data.

This project aims to:
- Reduce specular ringing via frequency-domain filtering.
- Enhance image resolution through a trained deep neural network (NAFNet).
- Evaluate the quality of reconstructions both quantitatively and visually.

---

## 🧠 Method Summary

1. **Spectral Subband Filtering**
   - Perform 2D FFT on complex SAR images.
   - Select subbands around the peak energy region.
   - Combine filtered outputs using strategies like `min_energy`, `median`, and `AND`.

2. **Superresolution (NAFNet)**
   - Train a single-channel NAFNet model.
   - LR inputs: filtered subband-combined images.
   - HR targets: MERLIN-processed SWOT images.
   - Patch-based training with overlap for stability and diversity.

---

## 🗂️ Repository Structure

```bash
.
├── MERLIN/                    
├── Papers/                   
├── SAE_SuperResolution/                  
│   ├── data/
│      ├── SR
│      ├── Test/
│         ├── LR
│         ├── HR
│      ├── Train/
│         ├── LR
│         ├── HR
│   ├── metrics/
│   ├── model/
│   └── nafnet/
│      ├── arch_util.py
│      ├── local_arch.py
│      └── NAFNet_arch.py
│   ├── metrics.ipynb
│   ├── mvalab_v2.py
│   ├── test_nafnet.py
│   └── train_nafnet.py
├── Subband images analysis/                
│   ├── dataset/
│      ├── test/
│   ├── No_specular_ringing/
│   ├── filter_subband_combination.ipynb
│   ├── filter_subband_combination.py
│   └── mvalab_v2.py
├── README.md
└── report.pdf
```

🚀 How to Run

1. Subband Filtering

        python subband_filtering.py  # Processes full dataset or single image

2. Train NAFNet

        python train_nafnet.py

3. Test NAFNet

        python test_nafnet.py

⸻

❗ Limitations
- Small dataset: only 11 SWOT scenes available.
- HR targets (MERLIN) still contain ringing.
- SR model sometimes reintroduces artifacts.

⸻

🔍 Future Work
- Use clean synthetic HR targets for training.
- Try attention-based models (e.g. SwinIR).
- Add artifact-aware loss functions (e.g. perceptual or adversarial).
- Explore learnable subband selection.

⸻

👩‍💻 Authors
- Laura Manuela Castañeda Medina
- Daniel Felipe Torres Robles
M2 MVA Masters' Course, 2025

⸻

📄 License

MIT License – feel free to use, modify, and cite.
