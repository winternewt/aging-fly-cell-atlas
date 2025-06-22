---
license: cc-by-4.0
tags:
- longevity
- aging
- drosophila
- single-cell-rna-seq
- fly-aging
- cellular-aging
- 10x-genomics
- aging-atlas
- model-organism
pretty_name: "Aging Fly Cell Atlas (AFCA) - Drosophila melanogaster Head Dataset"
size_categories:
- 100K<n<1M
language:
- en

# Dataset configurations - Complete head and body datasets
configs:
- config_name: default
  data_files:
    - split: head_expression
      path: "aging_fly_head_expression.parquet"
    - split: head_sample_metadata
      path: "aging_fly_head_sample_metadata.parquet"
    - split: head_feature_metadata
      path: "aging_fly_head_feature_metadata.parquet"
    - split: head_projection_pca
      path: "aging_fly_head_projection_X_pca.parquet"
    - split: head_projection_tsne
      path: "aging_fly_head_projection_X_tsne.parquet"  
    - split: head_projection_umap
      path: "aging_fly_head_projection_X_umap.parquet"
    - split: body_expression
      path: "aging_fly_body_expression.parquet"
    - split: body_sample_metadata
      path: "aging_fly_body_sample_metadata.parquet"
    - split: body_feature_metadata
      path: "aging_fly_body_feature_metadata.parquet"
    - split: body_projection_pca
      path: "aging_fly_body_projection_X_pca.parquet"
    - split: body_projection_tsne
      path: "aging_fly_body_projection_X_tsne.parquet"
    - split: body_projection_umap
      path: "aging_fly_body_projection_X_umap.parquet"

- config_name: metadata_json
  data_files:
    - split: head_unstructured_metadata
      path: "aging_fly_head_unstructured_metadata.json"
    - split: body_unstructured_metadata
      path: "aging_fly_body_unstructured_metadata.json"
---

# 🧬 Aging Fly Cell Atlas (AFCA) - Complete Dataset

> **Comprehensive single-nucleus transcriptomic atlas of aging in Drosophila melanogaster covering both head and body tissues for longevity research and machine learning applications**

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Aging%20Fly%20Cell%20Atlas-blue)](https://huggingface.co/datasets/longevity-gpt/aging-fly-cell-atlas)
[![Paper](https://img.shields.io/badge/📖%20Paper-Science%202023-red)](https://www.science.org/doi/10.1126/science.adg0934)
[![Data Portal](https://img.shields.io/badge/🌐%20AFCA%20Portal-Interactive-green)](https://hongjielilab.org/afca/)
[![License](https://img.shields.io/badge/📄%20License-CC%20BY%204.0-yellow)](https://creativecommons.org/licenses/by/4.0/)

## 📊 Dataset Overview

**Original Study**: [Lu et al., Science 2023](https://www.science.org/doi/10.1126/science.adg0934)  
**Interactive Atlas**: [hongjielilab.org/afca](https://hongjielilab.org/afca/)  
**GEO Repository**: [GSE218661](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218661)  
**Processing Repository**: [github.com/winternewt/aging-fly-cell-atlas](https://github.com/winternewt/aging-fly-cell-atlas) - Complete data processing pipeline

This dataset provides the most comprehensive single-nucleus transcriptomic atlas of aging in _Drosophila melanogaster_, covering the entire organism across the lifespan. The Aging Fly Cell Atlas (AFCA) enables unprecedented insights into cellular aging, longevity mechanisms, and age-related disease processes.

### Key Features (Complete Dataset)
- **566,273 single nuclei** from both head and body tissues
- **78 distinct cell types** with detailed annotations (40 head + 38 body types)
- **Multiple age timepoints**: 5, 30, 50, 70 days across lifespan
- **Sex-stratified data**: Male and female flies analyzed separately  
- **Rich annotations**: AFCA, FCA, and broad cell type classifications
- **Pre-computed embeddings**: PCA, t-SNE, and UMAP coordinates for both tissues
- **Quality control metrics**: Comprehensive QC data for all cells

---

## 🗂️ Dataset Structure

The processed AFCA complete dataset contains optimized parquet files ready for HuggingFace:

```
processed/
# HEAD TISSUE
├── aging_fly_head_expression.parquet        # Head expression matrix (962MB)
├── aging_fly_head_sample_metadata.parquet   # Head cell metadata (5.6MB)
├── aging_fly_head_feature_metadata.parquet  # Gene annotations (220KB)
├── aging_fly_head_projection_X_pca.parquet  # Head PCA embeddings (258MB)
├── aging_fly_head_projection_X_umap.parquet # Head UMAP coordinates (5.8MB)
├── aging_fly_head_projection_X_tsne.parquet # Head t-SNE coordinates (5.8MB)
├── aging_fly_head_unstructured_metadata.json # Head processing metadata

# BODY TISSUE  
├── aging_fly_body_expression.parquet        # Body expression matrix (916MB)
├── aging_fly_body_sample_metadata.parquet   # Body cell metadata (5.5MB)
├── aging_fly_body_feature_metadata.parquet  # Gene annotations (220KB)
├── aging_fly_body_projection_X_pca.parquet  # Body PCA embeddings (85MB)
├── aging_fly_body_projection_X_umap.parquet # Body UMAP coordinates (5.6MB)
└── aging_fly_body_projection_X_tsne.parquet # Body t-SNE coordinates (5.6MB)
└── aging_fly_body_unstructured_metadata.json # Body processing metadata
```

### Data Dimensions (Complete Dataset)
- **Cells**: 566,273 single nuclei (289,981 head + 276,273 body)
- **Genes**: ~16,000 protein-coding and non-coding genes  
- **Cell Types**: 78 distinct cell types (40 head + 38 body)
- **Ages**: Multiple timepoints (5, 30, 50, 70 days across lifespan)
- **Sexes**: Male and female flies
- **Annotations**: 3 levels (AFCA, FCA, and broad classifications)
- **File Size**: 2.2GB total (optimized parquet format)

---

## 🔬 Biological Context

### Aging Phenotypes Captured
- **Fat body expansion**: Multinuclear cells via amitosis-like division
- **Muscle sarcopenia**: Loss of flight and skeletal muscle mass
- **Metabolic changes**: Altered lipid homeostasis and energy metabolism  
- **Ribosomal decline**: Universal decrease in protein synthesis machinery
- **Mitochondrial dysfunction**: Reduced oxidative phosphorylation

### Cell Type Diversity
- **Neurons**: Cholinergic, GABAergic, glutamatergic, monoaminergic
- **Glia**: Astrocytes, ensheathing, cortex, surface glia
- **Specialized**: Photoreceptors, Kenyon cells, peptidergic neurons
- **Non-neural**: Fat body, muscle, hemocytes, reproductive cells

### Aging Features Quantified
1. **Cell composition changes** - Which cell types expand/contract with age
2. **Differential gene expression** - Age-related transcriptional changes  
3. **Cell identity maintenance** - Stability of cell type markers
4. **Expressed gene diversity** - Changes in transcriptional complexity

---

## 🚀 Quick Start

### Loading the Dataset

```python
from datasets import load_dataset
import pandas as pd

# Load the complete AFCA dataset from HuggingFace
dataset = load_dataset("longevity-db/aging-fly-cell-atlas")

# Access HEAD tissue data
head_expression = dataset['head_expression'].to_pandas()
head_metadata = dataset['head_sample_metadata'].to_pandas()
head_features = dataset['head_feature_metadata'].to_pandas()
head_pca = dataset['head_projection_pca'].to_pandas()
head_umap = dataset['head_projection_umap'].to_pandas()

# Access BODY tissue data
body_expression = dataset['body_expression'].to_pandas()
body_metadata = dataset['body_sample_metadata'].to_pandas()
body_features = dataset['body_feature_metadata'].to_pandas()
body_pca = dataset['body_projection_pca'].to_pandas()
body_umap = dataset['body_projection_umap'].to_pandas()

print(f"Head dataset: {head_expression.shape[0]:,} cells × {head_expression.shape[1]:,} genes")
print(f"Body dataset: {body_expression.shape[0]:,} cells × {body_expression.shape[1]:,} genes")
print(f"Total cells: {head_expression.shape[0] + body_expression.shape[0]:,}")

# Combine datasets if needed
import pandas as pd
combined_metadata = pd.concat([
    head_metadata.assign(tissue='head'),
    body_metadata.assign(tissue='body')
], ignore_index=True)

print(f"Cell types: {combined_metadata['afca_annotation'].nunique()}")
print(f"Ages available: {sorted(combined_metadata['age'].unique())}")
```

### Aging Analysis Example

```python
import numpy as np
from scipy.stats import ttest_ind

# Compare young vs old flies in head tissue
young_mask = sample_metadata['age'].isin(['5', '30'])
old_mask = sample_metadata['age'].isin(['50', '70'])

young_cells = sample_metadata[young_mask].index
old_cells = sample_metadata[old_mask].index

# Calculate differential expression
young_expr = expression.loc[young_cells].mean()
old_expr = expression.loc[old_cells].mean()

# Find age-related genes (top fold changes)
log2fc = np.log2((old_expr + 1e-9) / (young_expr + 1e-9))
top_aging_genes = log2fc.abs().nlargest(10)

print("Top age-related genes (by fold change):")
for gene, fc in top_aging_genes.items():
    direction = "↑" if log2fc[gene] > 0 else "↓"
    print(f"  {gene}: {direction} {abs(fc):.2f} log2FC")

# Cell composition changes across ages
young_composition = sample_metadata[young_mask]['afca_annotation'].value_counts(normalize=True)
old_composition = sample_metadata[old_mask]['afca_annotation'].value_counts(normalize=True)

print(f"\nAge group sizes: Young={len(young_cells):,}, Old={len(old_cells):,}")
print("\nCell types with biggest age-related changes:")
composition_changes = (old_composition / young_composition).fillna(0).sort_values(ascending=False)
print(composition_changes.head(5))
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Age-colored UMAP for head tissue
plt.figure(figsize=(12, 8))
scatter = plt.scatter(umap_coords.iloc[:, 0], umap_coords.iloc[:, 1], 
                     c=sample_metadata['age'].astype('category').cat.codes, 
                     cmap='viridis', s=0.5, alpha=0.6)
plt.colorbar(scatter, label='Age')
plt.title('Aging Fly Head Atlas - UMAP colored by age')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# Cell type composition across ages
age_composition = sample_metadata.groupby(['age', 'afca_annotation']).size().unstack(fill_value=0)
age_composition_norm = age_composition.div(age_composition.sum(axis=1), axis=0)

plt.figure(figsize=(12, 6))
age_composition_norm.plot(kind='bar', stacked=True)
plt.title('Head Cell Type Composition Changes During Aging')
plt.xlabel('Age (days)')
plt.ylabel('Proportion of cells')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Top cell types by age
top_cell_types = sample_metadata['afca_annotation'].value_counts().head(10).index
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, cell_type in enumerate(top_cell_types):
    mask = sample_metadata['afca_annotation'] == cell_type
    subset_coords = umap_coords[mask]
    
    axes[i].scatter(umap_coords.iloc[:, 0], umap_coords.iloc[:, 1], 
                   c='lightgray', s=0.1, alpha=0.3)
    axes[i].scatter(subset_coords.iloc[:, 0], subset_coords.iloc[:, 1], 
                   s=0.5, alpha=0.8)
    axes[i].set_title(f'{cell_type} ({mask.sum():,} cells)')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

---

## 📈 Key Findings & Applications

### Major Discoveries
1. **Cell-type-specific aging rates**: Different tissues age at different speeds
2. **Fat body multinucleation**: Novel mechanism of cellular aging
3. **Conserved ribosomal decline**: Universal aging signature across cell types
4. **Aging clocks**: High-accuracy age prediction from single-cell transcriptomes
5. **Sex differences**: Distinct aging patterns between male and female flies

### Research Applications
- **Longevity research**: Identify pro-longevity targets and mechanisms
- **Aging clocks**: Develop biomarkers of biological aging
- **Disease modeling**: Understand age-related pathological processes  
- **Drug discovery**: Screen anti-aging interventions at cellular resolution
- **Comparative aging**: Cross-species aging studies with mammals

### Machine Learning Use Cases
- **Age prediction**: Train aging clocks on single-cell data
- **Cell type classification**: Identify cell states and transitions
- **Trajectory analysis**: Model aging dynamics and cellular transitions
- **Biomarker discovery**: Find molecular signatures of healthy aging
- **Drug response prediction**: Model intervention effects on aging

---

## 🛠️ Data Processing & Repository

**Want to understand how this dataset was created?** The complete data processing pipeline is available in our GitHub repository:

**📜 Processing Repository**: [github.com/winternewt/aging-fly-cell-atlas](https://github.com/winternewt/aging-fly-cell-atlas)

**Key Processing Scripts**:
- ✅ **Data Retrieval**: [`01_data_retrieval.py`](https://github.com/winternewt/aging-fly-cell-atlas/blob/main/scripts/01_data_retrieval.py) - Automated GEO download and metadata extraction
- ✅ **Data Processing**: [`03_data_processing.py`](https://github.com/winternewt/aging-fly-cell-atlas/blob/main/scripts/03_data_processing.py) - H5AD to HuggingFace parquet conversion
- ✅ **Upload Script**: [`04_upload_to_huggingface.py`](https://github.com/winternewt/aging-fly-cell-atlas/blob/main/scripts/04_upload_to_huggingface.py) - HuggingFace dataset upload

**🔧 Technical Features**:
- Memory-efficient processing for large datasets (~290K cells)
- Automated data retrieval from GEO with comprehensive metadata
- Quality control validation and error handling
- HuggingFace-optimized file formats (parquet)
- Comprehensive logging and progress tracking

```bash
# Reproduce this dataset from scratch
git clone https://github.com/winternewt/aging-fly-cell-atlas
cd aging-fly-cell-atlas
python3 scripts/01_data_retrieval.py  # Download from GEO
python3 scripts/03_data_processing.py # Process to HF format
```

*This transparency enables reproducibility and helps researchers understand data transformations applied to the original study data.*

---

## 🔗 Related Resources

### Original Data & Tools
- **AFCA Portal**: [hongjielilab.org/afca](https://hongjielilab.org/afca) - Interactive data exploration
- **CELLxGENE**: [cellxgene.cziscience.com](https://cellxgene.cziscience.com/) - Online visualization
- **GEO Repository**: [GSE218661](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218661) - Raw sequencing data
- **Original Processing Code**: Available on [Zenodo](https://doi.org/10.5281/zenodo.7853649)

### Companion Datasets
- **Fly Cell Atlas (FCA)**: Young fly reference atlas
- **Mouse Aging Cell Atlas**: Cross-species aging comparisons
- **Human Brain Aging**: Comparative aging studies

---

## 📖 Citation

If you use this dataset in your research, please cite the original publication:

```bibtex
@article{lu2023aging,
  title={Aging Fly Cell Atlas identifies exhaustive aging features at cellular resolution},
  author={Lu, Tzu-Chiao and Brbi{\'c}, Maria and Park, Ye-Jin and Jackson, Tyler and Chen, Jiaye and Kolluru, Sai Saroja and Qi, Yanyan and Katheder, Nadja Sandra and Cai, Xiaoyu Tracy and Lee, Seungjae and others},
  journal={Science},
  volume={380},
  number={6650},
  pages={eadg0934},
  year={2023},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.adg0934}
}
```

**HuggingFace Dataset Citation**:
```bibtex
@dataset{aging_fly_cell_atlas_2024,
  title={Aging Fly Cell Atlas - HuggingFace Dataset},
  author={Longevity Genomics Consortium},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/longevity-gpt/aging-fly-cell-atlas}
}
```

---

## 🤝 Contributing & Support

### Getting Help
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join community discussions for usage questions  
- **Documentation**: Complete processing pipeline in `CODE_README.md`

### Data Processing
This dataset was processed from the original H5AD files using an optimized pipeline:
- Quality control and filtering
- Normalization and batch correction  
- Dimensionality reduction (PCA, UMAP, t-SNE, scVI)
- Cell type annotation and validation
- Age-related analysis and aging clock development

See `CODE_README.md` for complete processing documentation.

---

## 📄 License & Usage

**License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) - Free to use with attribution

**Data Usage Guidelines**:
- ✅ Research and commercial use permitted
- ✅ Modification and redistribution allowed  
- ✅ Academic and educational use encouraged
- 📝 **Attribution required**: Cite original Lu et al. Science 2023 paper

**Ethical Considerations**:
- Animal research conducted under institutional oversight
- Data sharing approved by original authors
- No human subjects or sensitive personal information

---

**🔬 Ready for aging research • 🧬 Comprehensively annotated • 💻 ML-optimized • 📊 Cross-species relevant**

---

## 📥 Data Retrieval & Processing

This dataset was programmatically retrieved from **GEO GSE218661** using our automated pipeline:

### Automated Download
```bash
# Clone the repository
git clone https://github.com/your-repo/aging-fly-cell-atlas
cd aging-fly-cell-atlas

# Activate environment (uv project)
source .venv/bin/activate

# Run data retrieval script
python scripts/01_data_retrieval.py
```

### What the Script Does
1. **Extracts GEO Metadata**: Downloads comprehensive metadata for all 72 samples
2. **Downloads H5AD Files**: Automatically finds and downloads processed h5ad files from GEO
3. **Processes Data**: Decompresses files and extracts cell/gene statistics
4. **Organizes Structure**: Places files in clean directory structure
5. **Generates Metadata**: Creates detailed JSON and CSV metadata files

### Retrieved Files
- **`afca_head.h5ad`**: 289,981 head cells with full annotations and embeddings
- **`afca_body.h5ad`**: 276,273 body cells with full annotations and embeddings
- **Metadata**: Complete sample information, processing logs, and statistics

### Data Quality
- ✅ **Complete Age Series**: 5d, 30d, 50d, 70d timepoints
- ✅ **Sex-Stratified**: Male and female samples
- ✅ **Rich Annotations**: FCA and AFCA cell type annotations
- ✅ **Embeddings Included**: PCA, t-SNE, and UMAP coordinates
- ✅ **Quality Metrics**: Cell counts, gene counts, mitochondrial percentages

---
