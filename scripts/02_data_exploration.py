#!/usr/bin/env python3
"""
Phase 2: Data Exploration & Setup
Aging Fly Cell Atlas (AFCA)

This script loads and inspects the afca_body.h5ad and afca_head.h5ad files to understand:
- Dataset dimensions and structure for both body and head atlases
- Cell type annotations and tissue organization
- Age-related information (developmental stages and aging timepoints)
- Sequencing method and technical factors
- Available embeddings and projections
- Quality control metrics
- Tissue-specific features and metadata structure
"""

import scanpy as sc
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json

# Suppress scanpy warnings for cleaner output
warnings.filterwarnings('ignore')
sc.settings.verbosity = 1

def explore_fly_dataset(dataset_name: str, file_path: Path):
    """Load and comprehensively explore a fly cell atlas dataset."""
    
    print(f"🪰 EXPLORING {dataset_name.upper()} DATASET")
    print("=" * 60)
    
    # Load the dataset
    print(f"📂 Loading {dataset_name} dataset...")
    
    if not file_path.exists():
        print(f"❌ Error: Dataset not found at {file_path}")
        return None, None
    
    try:
        adata = sc.read_h5ad(file_path)
        print(f"✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None
    
    # Dataset Overview
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   Shape: {adata.shape} (cells × genes)")
    
    file_size_gb = file_path.stat().st_size / (1024**3)
    print(f"   File size: {file_size_gb:.1f} GB")
    print(f"   Data type: {type(adata.X)}")
    
    if hasattr(adata.X, 'nnz'):
        sparsity = (1 - adata.X.nnz / (adata.n_obs * adata.n_vars)) * 100
        print(f"   Sparsity: {sparsity:.1f}% zeros")
    
    # Cell Type and Tissue Annotations
    print(f"\n🏷️  CELL TYPE & TISSUE ANNOTATIONS")
    
    # Look for common cell type annotation columns
    cell_type_columns = [col for col in adata.obs.columns if any(keyword in col.lower() 
                        for keyword in ['cell_type', 'celltype', 'annotation', 'cluster', 'type'])]
    
    if cell_type_columns:
        print(f"   Found {len(cell_type_columns)} cell type related columns:")
        for col in cell_type_columns:
            n_types = adata.obs[col].nunique()
            print(f"   ✅ {col}: {n_types} unique types")
            
            # Show top 10 most abundant types
            top_types = adata.obs[col].value_counts().head(10)
            print(f"      Top 10: {list(top_types.index)}")
    else:
        print(f"   ⚠️  No obvious cell type columns found")
        print(f"   Available obs columns: {list(adata.obs.columns)}")
    
    # Tissue/Organ Structure (specific to body vs head)
    print(f"\n🦋 TISSUE/ORGAN STRUCTURE")
    tissue_columns = [col for col in adata.obs.columns if any(keyword in col.lower() 
                     for keyword in ['tissue', 'organ', 'compartment', 'region'])]
    
    if tissue_columns:
        for col in tissue_columns:
            tissues = adata.obs[col].value_counts()
            print(f"   ✅ {col}: {len(tissues)} unique tissues/regions")
            for tissue, count in tissues.items():
                print(f"      {tissue}: {count:,} cells")
    else:
        print(f"   📝 No specific tissue columns found - may be single-tissue dataset")
    
    # Age/Developmental Stage Analysis
    print(f"\n⏰ AGE/DEVELOPMENTAL STAGE ANALYSIS")
    age_columns = [col for col in adata.obs.columns if any(keyword in col.lower() 
                  for keyword in ['age', 'stage', 'day', 'development', 'time'])]
    
    if age_columns:
        for col in age_columns:
            print(f"   ✅ {col}:")
            age_counts = adata.obs[col].value_counts().sort_index()
            for age_val, count in age_counts.items():
                print(f"      {age_val}: {count:,} cells")
            
            # Analyze age distribution
            if len(age_counts) > 1:
                print(f"   📊 Age range: {min(age_counts.index)} to {max(age_counts.index)}")
    else:
        print(f"   ⚠️  No obvious age/developmental columns found")
    
    # Sex/Genetic Background
    print(f"\n🧬 GENETIC FACTORS")
    genetic_columns = [col for col in adata.obs.columns if any(keyword in col.lower() 
                      for keyword in ['sex', 'gender', 'genotype', 'strain', 'background'])]
    
    for col in genetic_columns:
        if col in adata.obs.columns:
            values = adata.obs[col].value_counts()
            print(f"   ✅ {col}: {list(values.index)} (counts: {list(values.values)})")
    
    # Technical Factors & Sample Structure
    print(f"\n🔬 TECHNICAL FACTORS & SAMPLE STRUCTURE")
    
    # Look for batch, sample, and technical columns
    tech_columns = [col for col in adata.obs.columns if any(keyword in col.lower() 
                   for keyword in ['batch', 'sample', 'donor', 'replicate', 'library', 'lane'])]
    
    for col in tech_columns:
        n_unique = adata.obs[col].nunique()
        print(f"   ✅ {col}: {n_unique} unique values")
        if n_unique <= 10:
            unique_vals = sorted(adata.obs[col].unique())
            print(f"      Values: {unique_vals}")
    
    # Quality Control Metrics
    print(f"\n📊 QUALITY CONTROL METRICS")
    qc_columns = [col for col in adata.obs.columns if any(keyword in col.lower() 
                 for keyword in ['n_genes', 'n_counts', 'total_counts', 'n_umi', 'mito', 'ribo', 'doublet', 'scrublet'])]
    
    for col in qc_columns:
        if adata.obs[col].dtype in ['int64', 'float64']:
            stats = adata.obs[col].describe()
            print(f"   ✅ {col}:")
            print(f"      Range: {stats['min']:.1f} - {stats['max']:.1f}")
            print(f"      Mean ± Std: {stats['mean']:.1f} ± {stats['std']:.1f}")
        else:
            print(f"   ✅ {col}: {adata.obs[col].value_counts().to_dict()}")
    
    # Gene Metadata Structure
    print(f"\n🧬 GENE METADATA STRUCTURE")
    print(f"   Gene metadata columns ({len(adata.var.columns)}): {list(adata.var.columns)}")
    
    # Look for gene symbols, IDs, biotypes
    gene_info_columns = ['gene_id', 'gene_symbol', 'gene_name', 'biotype', 'chromosome']
    for col in gene_info_columns:
        matching_cols = [c for c in adata.var.columns if col.lower() in c.lower()]
        if matching_cols:
            print(f"   ✅ Gene {col}: {matching_cols[0]}")
            if 'symbol' in col.lower() or 'name' in col.lower():
                print(f"      Sample genes: {list(adata.var[matching_cols[0]][:10])}")
    
    # Available Dimensionality Reductions
    print(f"\n📈 DIMENSIONALITY REDUCTIONS & EMBEDDINGS")
    print(f"   Available projections (.obsm): {len(adata.obsm)} found")
    
    for proj_key in adata.obsm.keys():
        shape = adata.obsm[proj_key].shape
        print(f"   ✅ {proj_key}: {shape}")
    
    if not adata.obsm:
        print(f"   📝 No pre-computed embeddings found")
    
    # Additional Matrices (.layers)
    if adata.layers:
        print(f"\n📊 EXPRESSION LAYERS")
        for layer_key in adata.layers.keys():
            print(f"   ✅ {layer_key}: {type(adata.layers[layer_key])}")
    
    # Sequencing Method Detection
    print(f"\n🔬 SEQUENCING METHOD DETECTION")
    
    # Analyze gene count distribution to infer method
    if 'n_genes' in adata.obs.columns or any('gene' in col.lower() for col in adata.obs.columns):
        gene_col = next((col for col in adata.obs.columns if 'n_genes' in col.lower()), None)
        if not gene_col:
            gene_col = next((col for col in adata.obs.columns if 'gene' in col.lower()), None)
        
        if gene_col and adata.obs[gene_col].dtype in ['int64', 'float64']:
            mean_genes = adata.obs[gene_col].mean()
            print(f"   📊 Mean genes per cell: {mean_genes:.0f}")
            
            if mean_genes > 5000:
                print(f"   🔬 Likely method: Smart-seq2 or full-length")
            else:
                print(f"   🔬 Likely method: 10X or droplet-based")
    
    # Dataset Quality Assessment
    print(f"\n✅ DATASET QUALITY ASSESSMENT")
    
    quality_checks = {
        'Expression data': adata.X is not None,
        'Cell metadata': len(adata.obs.columns) > 5,
        'Gene metadata': len(adata.var.columns) > 0,
        'Reasonable cell count': adata.n_obs > 1000,
        'Reasonable gene count': adata.n_vars > 5000,
        'Has embeddings': len(adata.obsm) > 0
    }
    
    all_good = True
    for check, status in quality_checks.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check}")
        if not status:
            all_good = False
    
    # Fly-specific Research Value
    print(f"\n🪰 FLY RESEARCH VALUE ASSESSMENT")
    
    research_features = [
        f"✅ Drosophila model organism (well-characterized genetics)",
        f"✅ {dataset_name.title()} tissue focus (organ-specific biology)",
        f"✅ Large scale ({adata.n_obs:,} cells, {adata.n_vars:,} genes)",
        f"✅ Single-cell resolution (cellular heterogeneity)"
    ]
    
    # Add aging-specific features if detected
    if age_columns:
        research_features.append("✅ Aging/developmental stages (temporal dynamics)")
    
    for feature in research_features:
        print(f"   {feature}")
    
    print(f"\n💡 RESEARCH APPLICATIONS")
    print(f"   - Aging and longevity research in Drosophila")
    print(f"   - Tissue-specific cell type characterization")
    print(f"   - Developmental biology and cellular transitions")
    print(f"   - Comparative analysis with other model organisms")
    
    # Return summary information
    summary = {
        'dataset_name': dataset_name,
        'shape': adata.shape,
        'file_size_gb': file_size_gb,
        'obs_columns': list(adata.obs.columns),
        'var_columns': list(adata.var.columns),
        'obsm_keys': list(adata.obsm.keys()),
        'layers_keys': list(adata.layers.keys()) if adata.layers else [],
        'cell_type_columns': cell_type_columns,
        'age_columns': age_columns,
        'tissue_columns': tissue_columns,
        'genetic_columns': genetic_columns,
        'tech_columns': tech_columns,
        'qc_columns': qc_columns,
        'quality_passed': all_good
    }
    
    return adata, summary

def main():
    """Explore both body and head fly datasets."""
    
    print("🪰 AGING FLY CELL ATLAS - COMPREHENSIVE DATA EXPLORATION")
    print("=" * 80)
    
    # Define dataset paths
    datasets = {
        'body': Path('data/afca_body.h5ad'),
        'head': Path('data/afca_head.h5ad')
    }
    
    all_summaries = {}
    
    for dataset_name, file_path in datasets.items():
        adata, summary = explore_fly_dataset(dataset_name, file_path)
        
        if summary:
            all_summaries[dataset_name] = summary
        
        print("\n" + "="*80 + "\n")
    
    # Cross-dataset comparison
    if len(all_summaries) > 1:
        print("🔍 CROSS-DATASET COMPARISON")
        print("=" * 60)
        
        for metric in ['shape', 'file_size_gb']:
            print(f"\n📊 {metric.upper()}:")
            for dataset, summary in all_summaries.items():
                print(f"   {dataset}: {summary[metric]}")
        
        # Compare metadata availability
        print(f"\n🏷️  METADATA COMPARISON:")
        all_obs_columns = set()
        for summary in all_summaries.values():
            all_obs_columns.update(summary['obs_columns'])
        
        for col in sorted(all_obs_columns):
            availability = []
            for dataset, summary in all_summaries.items():
                if col in summary['obs_columns']:
                    availability.append(f"✅ {dataset}")
                else:
                    availability.append(f"❌ {dataset}")
            print(f"   {col}: {' | '.join(availability)}")
    
    # Save comprehensive results
    if all_summaries:
        # Ensure processed directory exists
        Path('processed').mkdir(exist_ok=True)
        
        with open('processed/fly_exploration_results.json', 'w') as f:
            json.dump(all_summaries, f, indent=2, default=str)
        
        print(f"\n💾 Exploration results saved to processed/fly_exploration_results.json")
        
        # Summary statistics
        print(f"\n📊 FINAL SUMMARY")
        total_cells = sum(summary['shape'][0] for summary in all_summaries.values())
        total_size = sum(summary['file_size_gb'] for summary in all_summaries.values())
        
        print(f"   Total cells across datasets: {total_cells:,}")
        print(f"   Total data size: {total_size:.1f} GB")
        
        print(f"\n🎉 Fly cell atlas exploration complete!")
        print(f"   Ready for downstream analysis and integration")

if __name__ == "__main__":
    main() 