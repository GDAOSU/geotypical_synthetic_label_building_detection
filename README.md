# Synthetic Data Matters: Re-training with Geo-typical Synthetic Labels for Building Detection

[![Paper](https://img.shields.io/badge/IEEE_TGRS-Paper-blue.svg)](https://arxiv.org/abs/2507.16657)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-yellow.svg)](https://huggingface.co/datasets/GDAOSU/Geotypical_Synthetic_Dataset)
[![Organization](https://img.shields.io/badge/OSU-GDA_Group-red.svg)](https://u.osu.edu/qin.324/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the dataset presented in our **IEEE TGRS 2025** paper: *"Synthetic Data Matters: Re-training with Geo-typical Synthetic Labels for Building Detection"*. 

Our work introduces a high-fidelity geotypical synthetic dataset specifically designed for building extraction and domain adaptation in remote sensing.

---

## 🚀 Dataset Access

The dataset (~4GB) is hosted on **Hugging Face Datasets** for stable distribution and version control.

### [👉 Click here to access the Dataset](https://huggingface.co/datasets/GDAOSU/Geotypical_Synthetic_Dataset)

---

## 📊 Dataset Specifications

### 1. Data Content & Format
- **Task**: Binary Semantic Segmentation (Building extraction).
- **Resolution**: 512 x 512 pixels.
- **Imagery**: RGB 3-channel remote sensing tiles.
- **Masks**: Single-channel PNG files.

### 2. Normalization Statistics (from `info.json`)
If you are using these images for training, we recommend using the following calculated statistics for normalization:
- **Mean (RGB)**: `[101.069, 103.695, 97.969]`
- **Std (RGB)**: `[42.538, 41.296, 39.941]`

### 3. Label Mapping & Palette
The raw masks use specific pixel values that should be mapped as follows for standard training:

| Original Mask Value | Mapped Training Label | Description | Hex Color |
|---------------------|-----------------------|-------------|-----------|
| 0                   | 255 (Ignore)          | Unknown     | #903C08   |
| 1                   | 0                     | Background  | #903C08   |
| 2                   | 1                     | **Building**| #FFFF00   |

---

## 📂 Repository Structure

When you download and extract `building_dataset.zip`, you will find the following structure:

```text
building_dataset/
├── imgs/          # Original RGB remote sensing tiles
├── masks/         # Segmentation masks (suffix: 'seg')
├── info.json      # Metadata: mapping, mean, std, and palette
├── train.txt      # List of filenames for training
├── val.txt        # List of filenames for validation
└── label.txt      # Master list of all label files
```

---

## 🛠️ Usage & Download

### Using `huggingface-cli` (Recommended)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download GDAOSU/Geotypical_Synthetic_Dataset --include "building_dataset.zip" --local-dir ./
```

### Using Python
```python
from huggingface_hub import hf_hub_download
import zipfile

# Download
path = hf_hub_download(
    repo_id="GDAOSU/Geotypical_Synthetic_Dataset", 
    filename="building_dataset.zip", 
    repo_type="dataset"
)

# Extract
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall("./data")
```

---

## 📖 Citation

If you find this dataset or research useful, please cite:

```bibtex
@article{song2025synthetic,
  title={Synthetic Data Matters: Re-training with Geo-typical Synthetic Labels for Building Detection},
  author={Song, Shuang and Tang, Yang and Qin, Rongjun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  doi={10.1109/TGRS.2025.3593864},
  publisher={IEEE}
}
```

---

## 🏫 About GDAOSU
This work is part of the research at the **Geospatial Data Analytics Group (GDAOSU)**, Department of Civil, Environmental and Geodetic Engineering, **The Ohio State University**.

**Contact**: [Tang, Yang](mailto:tang.1693@osu.edu) (Associate Researcher)