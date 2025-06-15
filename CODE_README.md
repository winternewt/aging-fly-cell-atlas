# 🧬 Aging Fly Cell Atlas (AFCA) - Processing Code Repository

> **Processing pipeline to convert the Aging Fly Cell Atlas into HuggingFace-compatible dataset format**

This repository contains the complete processing pipeline to transform the original H5AD files from the [Aging Fly Cell Atlas study](https://www.science.org/doi/10.1126/science.adg0934) into optimized, analysis-ready parquet files suitable for machine learning and longevity research.

## 📊 Dataset Overview

**Original Study**: [Lu et al., Science 2023](https://www.science.org/doi/10.1126/science.adg0934)  
**Interactive Atlas**: [hongjielilab.org/afca](https://hongjielilab.org/afca/)  
**GEO Repository**: [GSE218661](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218661)  
**Processing Code**: [TBD - This Repository]

## 🎯 Key Features

This dataset contains comprehensive single-nucleus RNA sequencing data from aging Drosophila melanogaster:

- **Cell Types**: 78 distinct cell types (40 head, 38 body regions)
- **Age Groups**: Multiple timepoints (2d, 5d, 15d) covering developmental stages
- **Sex Stratification**: Male and female flies analyzed separately
- **Rich Annotations**: AFCA, FCA, and broad cell type classifications
- **Pre-computed Embeddings**: PCA (150D), t-SNE, and UMAP coordinates

## 🛠️ Environment Setup

This project uses **uv** for Python environment management with Python 3.9.22.

### Prerequisites
- Python 3.9.22 (exactly)
- uv package manager
- ~5 GB free disk space

### Installation

```bash
# Clone repository
git clone [TBD-repo-url]
cd aging-fly-cell-atlas

# Create environment and install dependencies  
uv sync

# Activate environment
source .venv/bin/activate
```

### Dependencies

Key packages from `pyproject.toml`:

```toml
[project]
name = "aging-fly-cell-atlas"
version = "0.1.0"
requires-python = "==3.9.22"
dependencies = [
    "cellhint>=1.0.0",
    "celltypist>=1.6.3", 
    "GEOparse>=2.0.4",
    "huggingface-hub>=0.33.0",
    "jax[cpu]>=0.4.30",
    "matplotlib>=3.9.4",
    "mygene>=3.2.2",
    "numpy>=2.0.2",
    "pandas>=2.3.0",
    "pyarrow>=20.0.0",
    "pybiomart>=0.2.0",
    "scanpy>=1.10.3",
    "scvi>=0.6.8",
    "seaborn>=0.13.2",
    "torch>=2.7.1",
    "typer>=0.16.0",
]
```

## 📂 Repository Structure

```
aging-fly-cell-atlas/
├── pyproject.toml              # Project dependencies (uv)
├── README.md                   # Main dataset documentation  
├── CODE_README.md              # This processing guide
├── scripts/                    # Processing pipeline scripts
│   ├── 01_data_retrieval.py       # Download from GEO
│   ├── 02_data_exploration.py     # Data exploration and metadata discovery
│   ├── 03_data_processing.py      # Convert H5AD to parquet (planned)
│   └── 04_validation.py          # Quality control checks (planned)
├── data/                       # Raw data
│   ├── raw/                    # Original H5AD files
│   └── metadata/               # Annotation files
├── processed/                  # Parquet files for HF
└── notebooks/                  # Analysis examples (planned)
    ├── data_exploration.ipynb     # Dataset exploration examples
    ├── aging_analysis.ipynb       # Age-related analysis
    └── ml_examples.ipynb          # Machine learning applications
```

## 🔄 Processing Pipeline

### Step 1: Data Download
```bash
# Download original H5AD files from GEO GSE218661
python3 scripts/01_data_retrieval.py
```

### Step 2: Data Exploration
```bash
# Explore datasets and discover metadata structure
python3 scripts/02_data_exploration.py
```

### Step 3: H5AD to Parquet Conversion (planned)
```bash
# Convert H5AD files to HuggingFace-compatible parquet format
python3 scripts/03_data_processing.py
```

### Step 4: Data Validation (planned)
```bash
# Validate processed data integrity
python3 scripts/04_validation.py
```

## 📋 Data Processing Details

### Original Data Format
- **Source**: GEO GSE218661 H5AD files
- **Total Size**: 3.1 GB
- **Cell Count**: 566,254 single nuclei
- **Gene Count**: 15,992 features

### Processed Data Format
- **Format**: Parquet files optimized for pandas/polars
- **Compression**: Snappy compression for fast I/O
- **Schema**: HuggingFace datasets compatible
- **Total Size**: ~2-3 GB (compressed)

### File Structure (planned)
```
aging_fly_expression.parquet                     # Expression matrix (566K × 16K)
aging_fly_sample_metadata.parquet                # Cell annotations and QC metrics
aging_fly_feature_metadata.parquet               # Gene information  
aging_fly_projection_X_umap.parquet              # UMAP coordinates (2D)
aging_fly_projection_X_pca.parquet               # PCA embedding (150D)
aging_fly_projection_X_tsne.parquet              # t-SNE coordinates (2D)
aging_fly_unstructured_metadata.json             # Processing metadata
```

## 🧪 Quality Control

The processing pipeline includes comprehensive quality control:

- **Current QC Metrics**: n_genes_by_counts, total_counts, pct_counts_mt
- **Cell Type Annotations**: AFCA (78 types), FCA, and broad classifications
- **Age Validation**: Multiple developmental timepoints (2d, 5d, 15d)
- **Sex Stratification**: Male and female samples verified
- **Embedding Quality**: Pre-computed PCA (150D), t-SNE, and UMAP

## 📊 Usage Examples

### Loading Data
```python
from datasets import load_dataset

# Load expression data
dataset = load_dataset("longevity-gpt/aging-fly-cell-atlas", "expression")
expression_df = dataset["train"].to_pandas()

# Load metadata
metadata_dataset = load_dataset("longevity-gpt/aging-fly-cell-atlas", "sample_metadata") 
metadata_df = metadata_dataset["train"].to_pandas()
```

### Basic Analysis
```python
import pandas as pd
import scanpy as sc

# Load for scanpy analysis
adata_head = sc.read_h5ad("data/afca_head.h5ad")
adata_body = sc.read_h5ad("data/afca_body.h5ad")

# Basic stats
print(f"Head cells: {adata_head.n_obs:,}")
print(f"Body cells: {adata_body.n_obs:,}")
print(f"Genes: {adata_head.n_vars:,}")
print(f"Age groups: {sorted(adata_head.obs['age'].unique())}")
print(f"Head cell types: {adata_head.obs['afca_annotation'].nunique()}")
print(f"Body cell types: {adata_body.obs['afca_annotation'].nunique()}")
```

## 🔬 Scientific Applications

This dataset enables research in:

- **Aging Biology**: Transcriptomic changes across lifespan
- **Cell Type Dynamics**: Age-related cellular composition changes  
- **Biomarker Discovery**: Aging-associated gene expression signatures
- **Machine Learning**: Age prediction models and feature selection
- **Comparative Biology**: Cross-species aging mechanisms
- **Drug Discovery**: Aging intervention target identification

## 📖 Citation

If you use this dataset, please cite both the original study and this processing work:

**Original Study:**
```bibtex
[TBD - Add proper citation for Lu et al. Science 2023]
```

**This Dataset:**
```bibtex  
[TBD - Add citation for processed dataset]
```

## 📄 License

This dataset is released under the CC BY 4.0 license, consistent with the original study's data sharing policy.

## 🤝 Contributing

Contributions to improve the processing pipeline are welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📞 Contact

For questions about this processing pipeline or dataset:
- **Issues**: [GitHub Issues](https://github.com/[TBD]/aging-fly-cell-atlas/issues)
- **Original Study**: Contact authors of Lu et al., Science 2023

---

**Last Updated**: [TBD]  
**Processing Version**: [TBD] 