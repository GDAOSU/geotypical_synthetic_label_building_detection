# Synthetic Data Matters: Re-training with Geo-typical Synthetic Labels for Building Detection

[![Paper](https://img.shields.io/badge/IEEE_TGRS-Paper-blue.svg)](https://arxiv.org/abs/2507.16657)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-yellow.svg)](https://huggingface.co/datasets/GDAOSU/Geotypical_Synthetic_Dataset)
[![Organization](https://img.shields.io/badge/OSU-GDA_Group-red.svg)](https://ceg.osu.edu/labs/gda)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository provides the dataset information for the paper: **"Synthetic Data Matters: Re-training with Geo-typical Synthetic Labels for Building Detection"**, published in *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, 2025.

---

## 🚀 Dataset Access (Hugging Face)

Due to the large file size (~4GB), all data (images, masks, and metadata) are hosted on **Hugging Face Datasets**.

### [👉 Click here to access the Dataset on Hugging Face](https://huggingface.co/datasets/GDAOSU/Geotypical_Synthetic_Dataset)

**The dataset includes:**
- **High-fidelity Synthetic Data**: Generated via geo-typical procedural modeling for domain adaptation.
- **Real-world Target Data**: Urban tiles from cities like Chicago, Columbus (OSU), and more.
- **Binary Masks**: Pixel-level annotations for building extraction (Mapping: 2 -> Building).
- **Metadata**: Normalization statistics and class mapping defined in `info.json`.

---

## 📖 Citation

If you use this dataset or the findings from our research, please cite our IEEE TGRS paper:

```bibtex
@article{song2025synthetic,
  title={Synthetic Data Matters: Re-training with Geo-typical Synthetic Labels for Building Detection},
  author={Song, Shuang and Tang, Yang and Qin, Rongjun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}