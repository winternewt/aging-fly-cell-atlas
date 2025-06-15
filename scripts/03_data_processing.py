#!/usr/bin/env python3
"""
Phase 3: Data Processing for Aging Fly Cell Atlas (AFCA)
========================================================

Processes the H5AD files into HuggingFace-compatible parquet files:
- Expression matrix (sparse -> dense conversion with chunking)
- Sample metadata (cell-level information)
- Feature metadata (gene information)
- Dimensionality reduction projections (PCA, UMAP, t-SNE)
- Unstructured metadata (all additional data)

Processing Strategy:
- Process head and body datasets separately to avoid OOM
- Use chunking for large expression matrices
- Optimize data types for efficiency
- Apply pandas index bug fixes
- Save intermediate results to avoid data loss
- CLI interface for selective processing

Requirements:
- Memory-efficient processing for 566K × 16K matrices
- Sparse matrix handling for efficiency
- Proper data type optimization
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import shutil
import gc
import os
import psutil

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import pyarrow.parquet as pq
import typer
from typing_extensions import Annotated
import warnings

# Configure scanpy
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help="Process Aging Fly Cell Atlas data into HuggingFace format")

def get_memory_usage() -> float:
    """Get current memory usage in GB"""
    return psutil.virtual_memory().used / (1024**3)

def log_memory_status(stage: str) -> None:
    """Log current memory status"""
    memory_gb = get_memory_usage()
    available_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"{stage} - Memory: {memory_gb:.1f}GB used, {available_gb:.1f}GB available")

def make_json_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable objects for JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def log_memory_usage(stage: str, adata: sc.AnnData) -> None:
    """Log memory usage and dataset info"""
    memory_mb = adata.X.data.nbytes / 1024**2 if sparse.issparse(adata.X) else adata.X.nbytes / 1024**2
    logger.info(f"{stage}: Shape {adata.shape}, Memory: {memory_mb:.1f}MB")

def save_stage_result(output_dir: Path, tissue: str, stage: str, result: Dict[str, Any]) -> None:
    """Save intermediate results for each stage"""
    result_file = output_dir / f"{tissue}_{stage}_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"💾 Saved {stage} result for {tissue}")

def load_stage_result(output_dir: Path, tissue: str, stage: str) -> Optional[Dict[str, Any]]:
    """Load existing stage result if available"""
    result_file = output_dir / f"{tissue}_{stage}_result.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            result = json.load(f)
        logger.info(f"📖 Loaded existing {stage} result for {tissue}")
        return result
    return None

def get_completed_stages(output_dir: Path, tissue: str) -> Set[str]:
    """Get list of completed stages for a tissue"""
    stages = {'expression', 'sample_metadata', 'feature_metadata', 'projections', 'unstructured'}
    completed = set()
    
    for stage in stages:
        if load_stage_result(output_dir, tissue, stage) is not None:
            completed.add(stage)
    
    if completed:
        logger.info(f"🔄 Found completed stages for {tissue}: {', '.join(sorted(completed))}")
    
    return completed

def fix_pandas_index_column_bug(parquet_file: Path) -> bool:
    """
    Fix the pandas __index_level_0__ bug in parquet files
    
    This is a known bug in pandas/PyArrow where pandas saves the index as an extra 
    '__index_level_0__' column when writing to parquet format. 
    This is a known upstream issue with no planned fix
    
    References:
    - https://github.com/pandas-dev/pandas/issues/51664
    - https://github.com/pola-rs/polars/issues/7291

    Args:
        parquet_file: Path to the parquet file to fix
        
    Returns:
        bool: True if fix was applied successfully, False otherwise
    """
    logger.info(f"🔧 Checking for pandas __index_level_0__ bug in {parquet_file.name}")
    
    try:
        # Check if the bug exists
        pf = pq.ParquetFile(parquet_file)
        schema_names = pf.schema_arrow.names
        
        if '__index_level_0__' not in schema_names:
            logger.info("✅ No __index_level_0__ column found - file is clean")
            return True
            
        logger.warning(f"🐛 Found pandas __index_level_0__ bug - fixing...")
        logger.info(f"   Current columns: {len(schema_names)} (expected: {len(schema_names)-1})")
        
        # Create backup
        backup_file = parquet_file.with_suffix('.backup.parquet')
        if not backup_file.exists():
            shutil.copy2(parquet_file, backup_file)
            logger.info(f"📦 Backup created: {backup_file.name}")
        
        # Apply fix using PyArrow
        table = pq.read_table(parquet_file)
        
        # Filter out the problematic column
        columns_to_keep = [name for name in table.column_names if name != '__index_level_0__']
        clean_table = table.select(columns_to_keep)
        
        # Write clean table to temporary file first
        temp_file = parquet_file.with_suffix('.temp.parquet')
        pq.write_table(clean_table, temp_file, compression='snappy')
        
        # Verify the fix
        temp_pf = pq.ParquetFile(temp_file)
        temp_schema_names = temp_pf.schema_arrow.names
        
        if '__index_level_0__' not in temp_schema_names:
            # Replace original with fixed version
            shutil.move(temp_file, parquet_file)
            logger.info(f"✅ Fixed pandas __index_level_0__ bug")
            logger.info(f"   Column count: {len(schema_names)} → {len(temp_schema_names)}")
            return True
        else:
            # Fix failed, clean up
            temp_file.unlink()
            logger.error("❌ Fix verification failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error fixing pandas index bug: {e}")
        return False

def process_expression_matrix(adata: sc.AnnData, tissue: str, output_dir: Path, 
                            aggressive_chunking: bool = False) -> Dict[str, Any]:
    """
    Process and save expression matrix with chunking to avoid OOM
    
    Strategy:
    - Check sparsity and memory requirements
    - Use aggressive chunking for body dataset
    - Convert to float32 for efficiency
    - More frequent garbage collection
    """
    logger.info(f"Starting expression matrix processing for {tissue}...")
    log_memory_usage(f"Expression matrix ({tissue})", adata)
    log_memory_status("Before expression processing")
    
    # Calculate memory requirements for dense conversion
    dense_memory_gb = (adata.n_obs * adata.n_vars * 4) / (1024**3)  # float32 = 4 bytes
    sparsity = 1.0 - (adata.X.nnz / (adata.n_obs * adata.n_vars))
    
    logger.info(f"Dense conversion would require: {dense_memory_gb:.2f}GB")
    logger.info(f"Current sparsity: {sparsity:.2%}")
    
    output_file = output_dir / f"aging_fly_{tissue}_expression.parquet"
    
    # Determine chunk size based on tissue and available memory  
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    if tissue == 'body' or aggressive_chunking:
        # More aggressive chunking for body dataset
        chunk_size = min(2000, max(500, int(available_memory_gb * 100)))  # Scale with available memory
        logger.warning(f"🚨 Using aggressive chunking for {tissue} (chunk_size={chunk_size})")
    else:
        chunk_size = 5000
    
    logger.info(f"Processing expression matrix in chunks (size: {chunk_size})...")
    chunks = []
    
    for i in range(0, adata.n_obs, chunk_size):
        end_idx = min(i + chunk_size, adata.n_obs)
        chunk = adata[i:end_idx, :].copy()
        
        if sparse.issparse(chunk.X):
            chunk_dense = chunk.X.toarray().astype(np.float32)
        else:
            chunk_dense = chunk.X.astype(np.float32)
        
        chunk_df = pd.DataFrame(
            chunk_dense,
            index=chunk.obs_names,
            columns=chunk.var_names
        )
        chunks.append(chunk_df)
        
        chunk_num = i//chunk_size + 1
        total_chunks = (adata.n_obs-1)//chunk_size + 1
        logger.info(f"Processed chunk {chunk_num}/{total_chunks}")
        
        # More aggressive cleanup for body dataset
        del chunk, chunk_dense
        if tissue == 'body' or aggressive_chunking:
            gc.collect()  # Force GC every chunk
        
        # Memory check for body dataset
        if tissue == 'body':
            current_memory_gb = get_memory_usage()
            if current_memory_gb > 24:  # Warning at 24GB
                logger.warning(f"⚠️  High memory usage: {current_memory_gb:.1f}GB")
                # Force garbage collection
                gc.collect()
    
    # Combine chunks
    logger.info("Combining chunks...")
    log_memory_status("Before combining chunks")
    
    expression_df = pd.concat(chunks, axis=0)
    del chunks  # Free memory immediately
    gc.collect()
    
    log_memory_status("After combining chunks")
    
    # Save with compression
    logger.info(f"Saving expression matrix: {expression_df.shape}")
    expression_df.to_parquet(output_file, compression='snappy')
    
    # Apply pandas __index_level_0__ bug fix
    fix_success = fix_pandas_index_column_bug(output_file)
    
    stats = {
        'file': str(output_file),
        'shape': list(expression_df.shape),
        'memory_gb': dense_memory_gb,
        'sparsity_percent': sparsity * 100,
        'dtype': str(expression_df.dtypes.iloc[0]),
        'pandas_index_bug_fixed': fix_success,
        'chunk_size_used': chunk_size,
        'aggressive_chunking': aggressive_chunking
    }
    
    logger.info(f"✅ Expression matrix saved: {expression_df.shape}")
    del expression_df
    gc.collect()
    log_memory_status("After expression processing")
    return stats

def process_sample_metadata(adata: sc.AnnData, tissue: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save sample (cell) metadata"""
    logger.info(f"Processing sample metadata for {tissue}...")
    
    sample_metadata = adata.obs.copy()
    
    # Verify critical columns exist
    critical_cols = ['age', 'sex', 'afca_annotation', 'afca_annotation_broad']
    missing_cols = [col for col in critical_cols if col not in sample_metadata.columns]
    
    if missing_cols:
        logger.warning(f"Missing critical columns: {missing_cols}")
    else:
        logger.info("✅ All critical metadata columns present")
    
    # Add tissue column
    sample_metadata['tissue'] = tissue
    
    # Add standardized age column if needed
    if 'age_numeric' not in sample_metadata.columns and 'age' in sample_metadata.columns:
        # Convert age to numeric
        sample_metadata['age_numeric'] = pd.to_numeric(sample_metadata['age'], errors='coerce')
        logger.info("Added numeric age column")
    
    # Optimize data types
    for col in sample_metadata.columns:
        if sample_metadata[col].dtype == 'object':
            # Convert categorical strings to category type for efficiency
            if sample_metadata[col].nunique() < len(sample_metadata) * 0.5:
                sample_metadata[col] = sample_metadata[col].astype('category')
    
    output_file = output_dir / f"aging_fly_{tissue}_sample_metadata.parquet"
    sample_metadata.to_parquet(output_file, compression='snappy')
    
    stats = {
        'file': str(output_file),
        'shape': list(sample_metadata.shape),
        'columns': list(sample_metadata.columns),
        'missing_columns': missing_cols,
        'age_groups': sample_metadata['age'].value_counts().to_dict() if 'age' in sample_metadata.columns else {},
        'cell_types': sample_metadata['afca_annotation'].value_counts().head(10).to_dict() if 'afca_annotation' in sample_metadata.columns else {},
        'sex_distribution': sample_metadata['sex'].value_counts().to_dict() if 'sex' in sample_metadata.columns else {}
    }
    
    logger.info(f"✅ Sample metadata saved: {sample_metadata.shape}")
    return stats

def process_feature_metadata(adata: sc.AnnData, tissue: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save feature (gene) metadata"""
    logger.info(f"Processing feature metadata for {tissue}...")
    
    feature_metadata = adata.var.copy()
    
    # Ensure gene IDs are present
    if 'gene_ids' not in feature_metadata.columns:
        feature_metadata['gene_ids'] = feature_metadata.index
        logger.info("Added gene_ids column from index")
    
    # Check for gene symbols and other annotations
    symbol_cols = [col for col in feature_metadata.columns if 'symbol' in col.lower()]
    if symbol_cols:
        logger.info(f"Gene symbol columns found: {symbol_cols}")
    
    output_file = output_dir / f"aging_fly_{tissue}_feature_metadata.parquet"
    feature_metadata.to_parquet(output_file, compression='snappy')
    
    stats = {
        'file': str(output_file),
        'shape': list(feature_metadata.shape),
        'columns': list(feature_metadata.columns),
        'has_symbols': len(symbol_cols) > 0,
        'symbol_columns': symbol_cols
    }
    
    logger.info(f"✅ Feature metadata saved: {feature_metadata.shape}")
    return stats

def process_projections(adata: sc.AnnData, tissue: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save all dimensionality reduction projections"""
    logger.info(f"Processing dimensionality reduction projections for {tissue}...")
    
    projection_stats = {}
    available_projections = list(adata.obsm.keys())
    logger.info(f"Available projections: {available_projections}")
    
    for proj_name in available_projections:
        if proj_name.startswith('X_'):
            proj_data = adata.obsm[proj_name]
            
            # Convert to DataFrame
            proj_df = pd.DataFrame(
                proj_data,
                index=adata.obs_names,
                columns=[f"{proj_name.split('_')[1].upper()}{i+1}" for i in range(proj_data.shape[1])]
            )
            
            # Save projection
            output_file = output_dir / f"aging_fly_{tissue}_projection_{proj_name}.parquet"
            proj_df.to_parquet(output_file, compression='snappy')
            
            projection_stats[proj_name] = {
                'file': str(output_file),
                'shape': list(proj_df.shape),
                'dimensions': proj_data.shape[1]
            }
            
            logger.info(f"✅ Saved {proj_name}: {proj_df.shape}")
        else:
            logger.info(f"Skipping non-projection: {proj_name}")
    
    return projection_stats

def process_unstructured_metadata(adata: sc.AnnData, tissue: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save unstructured metadata (uns)"""
    logger.info(f"Processing unstructured metadata for {tissue}...")
    
    try:
        # Make data JSON serializable
        unstructured_data = make_json_serializable(adata.uns)
        
        output_file = output_dir / f"aging_fly_{tissue}_unstructured_metadata.json"
        
        with open(output_file, 'w') as f:
            json.dump(unstructured_data, f, indent=2)
        
        # Count keys and estimate size
        key_count = len(unstructured_data) if isinstance(unstructured_data, dict) else 0
        file_size_mb = output_file.stat().st_size / (1024**2)
        
        stats = {
            'file': str(output_file),
            'key_count': key_count,
            'file_size_mb': round(file_size_mb, 2),
            'top_keys': list(unstructured_data.keys())[:10] if isinstance(unstructured_data, dict) else []
        }
        
        logger.info(f"✅ Unstructured metadata saved: {key_count} keys, {file_size_mb:.1f}MB")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to process unstructured metadata: {e}")
        return {'error': str(e)}

def process_single_dataset(data_file: Path, tissue: str, output_dir: Path,
                          skip_stages: Set[str] = None, aggressive_chunking: bool = False) -> Dict[str, Any]:
    """Process a single H5AD dataset (head or body) with stage resumption"""
    logger.info(f"\n🧬 Processing {tissue.upper()} dataset: {data_file}")
    
    if skip_stages is None:
        skip_stages = set()
    
    # Check for existing results
    completed_stages = get_completed_stages(output_dir, tissue)
    stages_to_skip = skip_stages.union(completed_stages)
    
    if stages_to_skip:
        logger.info(f"⏭️  Skipping stages: {', '.join(sorted(stages_to_skip))}")
    
    # Processing results tracking
    processing_results = {
        'dataset_info': {
            'tissue': tissue,
            'file': str(data_file),
            'processing_time': None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'aggressive_chunking': aggressive_chunking
        }
    }
    
    # Load existing results
    for stage in ['expression', 'sample_metadata', 'feature_metadata', 'projections', 'unstructured']:
        if stage in completed_stages:
            existing_result = load_stage_result(output_dir, tissue, stage)
            if existing_result:
                processing_results[stage] = existing_result
    
    # Load data only if we need to process something
    stages_needed = {'expression', 'sample_metadata', 'feature_metadata', 'projections', 'unstructured'} - stages_to_skip
    
    if not stages_needed:
        logger.info(f"✅ All stages already completed for {tissue}")
        return processing_results
    
    logger.info(f"Loading {tissue} data from {data_file}...")
    log_memory_status("Before loading data")
    
    try:
        adata = sc.read_h5ad(data_file)
        logger.info(f"✅ {tissue.capitalize()} data loaded: {adata.shape}")
        processing_results['dataset_info']['shape'] = list(adata.shape)
        log_memory_usage(f"Initial ({tissue})", adata)
        log_memory_status("After loading data")
    except Exception as e:
        logger.error(f"Failed to load {tissue} data: {e}")
        return {'error': str(e)}
    
    start_time = time.time()
    
    try:
        # Task 3.1: Expression Matrix
        if 'expression' not in stages_to_skip:
            logger.info(f"\n🧬 Task 3.1: Processing {tissue} Expression Matrix")
            result = process_expression_matrix(adata, tissue, output_dir, aggressive_chunking)
            processing_results['expression'] = result
            save_stage_result(output_dir, tissue, 'expression', result)
        
        # Task 3.2: Sample Metadata
        if 'sample_metadata' not in stages_to_skip:
            logger.info(f"\n📊 Task 3.2: Processing {tissue} Sample Metadata")
            result = process_sample_metadata(adata, tissue, output_dir)
            processing_results['sample_metadata'] = result
            save_stage_result(output_dir, tissue, 'sample_metadata', result)
        
        # Task 3.3: Feature Metadata
        if 'feature_metadata' not in stages_to_skip:
            logger.info(f"\n🧪 Task 3.3: Processing {tissue} Feature Metadata")
            result = process_feature_metadata(adata, tissue, output_dir)
            processing_results['feature_metadata'] = result
            save_stage_result(output_dir, tissue, 'feature_metadata', result)
        
        # Task 3.4: Dimensionality Reductions
        if 'projections' not in stages_to_skip:
            logger.info(f"\n📈 Task 3.4: Processing {tissue} Projections")
            result = process_projections(adata, tissue, output_dir)
            processing_results['projections'] = result
            save_stage_result(output_dir, tissue, 'projections', result)
        
        # Task 3.5: Unstructured Metadata
        if 'unstructured' not in stages_to_skip:
            logger.info(f"\n📋 Task 3.5: Processing {tissue} Unstructured Metadata")
            result = process_unstructured_metadata(adata, tissue, output_dir)
            processing_results['unstructured'] = result
            save_stage_result(output_dir, tissue, 'unstructured', result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        processing_results['dataset_info']['processing_time'] = f"{processing_time:.1f}s"
        
        logger.info(f"\n✅ {tissue.capitalize()} Processing Complete!")
        logger.info(f"⏱️  Processing time: {processing_time:.1f}s")
        
        # Save overall result
        overall_result_file = output_dir / f"{tissue}_overall_result.json"
        with open(overall_result_file, 'w') as f:
            json.dump(processing_results, f, indent=2)
        logger.info(f"💾 Saved overall result for {tissue}")
        
        # Clean up memory
        del adata
        gc.collect()
        log_memory_status("After cleanup")
        
        return processing_results
        
    except Exception as e:
        logger.error(f"{tissue.capitalize()} processing failed: {e}")
        processing_results['error'] = str(e)
        
        # Save partial results even on error
        error_result_file = output_dir / f"{tissue}_error_result.json"
        with open(error_result_file, 'w') as f:
            json.dump(processing_results, f, indent=2)
        logger.info(f"💾 Saved partial results despite error")
        
        # Clean up memory even on error
        del adata
        gc.collect()
        
        return processing_results

def combine_metadata_files(output_dir: Path, tissues: List[str]) -> None:
    """Combine metadata files from different tissues"""
    logger.info("\n🔗 Combining metadata files across tissues...")
    
    # Combine sample metadata
    sample_dfs = []
    for tissue in tissues:
        sample_file = output_dir / f"aging_fly_{tissue}_sample_metadata.parquet"
        if sample_file.exists():
            df = pd.read_parquet(sample_file)
            sample_dfs.append(df)
            logger.info(f"Loaded {tissue} sample metadata: {df.shape}")
    
    if sample_dfs:
        combined_sample_df = pd.concat(sample_dfs, axis=0, ignore_index=False)
        combined_file = output_dir / "aging_fly_combined_sample_metadata.parquet"
        combined_sample_df.to_parquet(combined_file, compression='snappy')
        logger.info(f"✅ Combined sample metadata saved: {combined_sample_df.shape}")
    
    # Feature metadata should be identical, so just copy one
    for tissue in tissues:
        feature_file = output_dir / f"aging_fly_{tissue}_feature_metadata.parquet"
        if feature_file.exists():
            combined_feature_file = output_dir / "aging_fly_combined_feature_metadata.parquet"
            shutil.copy2(feature_file, combined_feature_file)
            logger.info(f"✅ Combined feature metadata copied from {tissue}")
            break

@app.command()
def process(
    tissue: Annotated[str, typer.Argument(help="Which tissue to process: 'head', 'body', or 'both'")] = "both",
    skip_expression: Annotated[bool, typer.Option(help="Skip expression matrix processing")] = False,
    skip_metadata: Annotated[bool, typer.Option(help="Skip metadata processing")] = False,
    skip_projections: Annotated[bool, typer.Option(help="Skip projection processing")] = False,
    aggressive_chunking: Annotated[bool, typer.Option(help="Use aggressive chunking (for low memory)")] = False,
    data_dir: Annotated[str, typer.Option(help="Data directory path")] = "data",
    output_dir: Annotated[str, typer.Option(help="Output directory path")] = "processed"
) -> None:
    """Process Aging Fly Cell Atlas data into HuggingFace format"""
    
    start_time = time.time()
    logger.info("=== Phase 3: Aging Fly Cell Atlas Data Processing Started ===")
    
    # Validate tissue parameter
    valid_tissues = {'head', 'body', 'both'}
    if tissue not in valid_tissues:
        logger.error(f"Invalid tissue '{tissue}'. Must be one of: {', '.join(valid_tissues)}")
        raise typer.Exit(1)
    
    # Setup paths
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    head_file = data_path / "afca_head.h5ad"
    body_file = data_path / "afca_body.h5ad"
    
    # Determine which datasets to process
    datasets_to_process = []
    if tissue in ['head', 'both']:
        if head_file.exists():
            datasets_to_process.append(('head', head_file))
        else:
            logger.warning(f"Head file not found: {head_file}")
    
    if tissue in ['body', 'both']:
        if body_file.exists():
            datasets_to_process.append(('body', body_file))
        else:
            logger.warning(f"Body file not found: {body_file}")
    
    if not datasets_to_process:
        logger.error("No valid datasets found to process")
        raise typer.Exit(1)
    
    # Setup skip stages
    skip_stages = set()
    if skip_expression:
        skip_stages.add('expression')
    if skip_metadata:
        skip_stages.update(['sample_metadata', 'feature_metadata', 'unstructured'])
    if skip_projections:
        skip_stages.add('projections')
    
    # Process datasets
    all_results = {}
    
    for tissue_name, data_file in datasets_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {tissue_name.upper()} dataset")
        logger.info(f"{'='*60}")
        
        # Use aggressive chunking for body by default, or if explicitly requested
        use_aggressive = aggressive_chunking or (tissue_name == 'body')
        
        results = process_single_dataset(data_file, tissue_name, output_path, 
                                       skip_stages, use_aggressive)
        all_results[tissue_name] = results
        
        # Force garbage collection between datasets
        gc.collect()
        log_memory_status(f"After processing {tissue_name}")
    
    # Generate summary
    generate_summary(output_path, all_results, start_time)

@app.command()
def summary(
    output_dir: Annotated[str, typer.Option(help="Output directory path")] = "processed"
) -> None:
    """Generate summary from existing results without reprocessing"""
    
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.error(f"Output directory not found: {output_path}")
        raise typer.Exit(1)
    
    logger.info("📊 Generating summary from existing results...")
    
    # Load existing results
    all_results = {}
    for tissue in ['head', 'body']:
        overall_result_file = output_path / f"{tissue}_overall_result.json"
        if overall_result_file.exists():
            with open(overall_result_file, 'r') as f:
                all_results[tissue] = json.load(f)
            logger.info(f"✅ Loaded {tissue} results")
        else:
            logger.warning(f"⚠️  No results found for {tissue}")
    
    if not all_results:
        logger.error("No existing results found")
        raise typer.Exit(1)
    
    generate_summary(output_path, all_results, time.time())

def generate_summary(output_path: Path, all_results: Dict[str, Any], start_time: float) -> None:
    """Generate processing summary"""
    
    # Combine metadata files if both tissues processed
    tissues = list(all_results.keys())
    if len(tissues) > 1:
        combine_metadata_files(output_path, tissues)
    
    # Calculate total processing time
    total_processing_time = time.time() - start_time
    
    summary = {
        'processing_info': {
            'total_time': f"{total_processing_time:.1f}s",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'datasets_processed': len(tissues)
        },
        'results': all_results
    }
    
    summary_file = output_path / "phase3_processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✅ Processing Summary Generated!")
    logger.info(f"⏱️  Total time: {total_processing_time:.1f}s")
    logger.info(f"📄 Summary saved: {summary_file}")
    
    # List all created files
    logger.info("\n📁 Created Files:")
    for file_path in sorted(output_path.glob("aging_fly_*.parquet")):
        size_mb = file_path.stat().st_size / (1024**2)
        logger.info(f"  {file_path.name} ({size_mb:.1f}MB)")
    
    for file_path in sorted(output_path.glob("aging_fly_*.json")):
        size_mb = file_path.stat().st_size / (1024**2)
        logger.info(f"  {file_path.name} ({size_mb:.1f}MB)")
    
    # Calculate total cells if available
    total_cells = 0
    for tissue_result in all_results.values():
        if 'dataset_info' in tissue_result and 'shape' in tissue_result['dataset_info']:
            total_cells += tissue_result['dataset_info']['shape'][0]
    
    if total_cells > 0:
        logger.info(f"\n🎉 Total cells processed: {total_cells:,}")

if __name__ == "__main__":
    app() 