#!/usr/bin/env python3
"""
Phase 1: Data Retrieval & Setup
Aging Fly Cell Atlas (AFCA) - GSE218661

This script programmatically retrieves h5ad files and metadata from GSE218661
for the Aging Fly Cell Atlas study. Downloads both head and body data files.

Key features:
- Downloads h5ad files from GEO supplementary files 
- Extracts comprehensive metadata from all available sources
- Organizes data in proper directory structure
- Validates downloaded files
"""

import os
import sys
import requests
import GEOparse
import pandas as pd
import json
import warnings
import gzip
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from urllib.parse import urlparse
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_directories() -> Dict[str, Path]:
    """Create necessary directory structure for AFCA data."""
    
    print("🗂️  SETTING UP DIRECTORY STRUCTURE")
    print("=" * 50)
    
    # Define directory structure
    dirs = {
        'data': Path('data'),
        'raw': Path('data/raw'),
        'processed': Path('processed'), 
        'metadata': Path('data/metadata'),
        'logs': Path('data/logs'),
        'supplementary': Path('data/raw/supplementary')
    }
    
    # Create directories
    for name, path in dirs.items():
        path.mkdir(exist_ok=True, parents=True)
        print(f"   ✅ Created: {path}")
    
    return dirs

def extract_geo_metadata(accession: str = "GSE218661") -> Dict:
    """Extract comprehensive metadata from GEO using GEOparse."""
    
    print(f"\n📊 EXTRACTING GEO METADATA FOR {accession}")
    print("=" * 50)
    
    try:
        # Download GEO metadata
        print(f"   📡 Connecting to GEO database...")
        gse = GEOparse.get_GEO(geo=accession, destdir="data/metadata/")
        
        metadata = {
            'accession': accession,
            'title': gse.metadata.get('title', [''])[0],
            'summary': gse.metadata.get('summary', [''])[0],
            'overall_design': gse.metadata.get('overall_design', [''])[0],
            'submission_date': gse.metadata.get('submission_date', [''])[0],
            'last_update_date': gse.metadata.get('last_update_date', [''])[0],
            'organism': gse.metadata.get('organism', []),
            'platform_organism': gse.metadata.get('platform_organism', []),
            'contact_email': gse.metadata.get('contact_email', [''])[0],
            'contact_name': gse.metadata.get('contact_name', [''])[0],
            'contact_institute': gse.metadata.get('contact_institute', [''])[0],
            'supplementary_file': gse.metadata.get('supplementary_file', []),
            'relation': gse.metadata.get('relation', []),
            'sample_count': len(gse.gsms),
            'platform_count': len(gse.gpls),
            'samples': {},
            'platforms': {}
        }
        
        # Extract sample metadata
        print(f"   🧪 Extracting metadata for {len(gse.gsms)} samples...")
        for gsm_name, gsm in gse.gsms.items():
            metadata['samples'][gsm_name] = {
                'title': gsm.metadata.get('title', [''])[0],
                'source_name_ch1': gsm.metadata.get('source_name_ch1', [''])[0],
                'organism_ch1': gsm.metadata.get('organism_ch1', [''])[0],
                'characteristics_ch1': gsm.metadata.get('characteristics_ch1', []),
                'treatment_protocol_ch1': gsm.metadata.get('treatment_protocol_ch1', [''])[0],
                'extract_protocol_ch1': gsm.metadata.get('extract_protocol_ch1', [''])[0],
                'description': gsm.metadata.get('description', [''])[0],
                'data_processing': gsm.metadata.get('data_processing', []),
                'platform_id': gsm.metadata.get('platform_id', [''])[0],
                'contact_name': gsm.metadata.get('contact_name', [''])[0],
                'supplementary_file': gsm.metadata.get('supplementary_file', []),
                'submission_date': gsm.metadata.get('submission_date', [''])[0],
                'last_update_date': gsm.metadata.get('last_update_date', [''])[0]
            }
        
        # Extract platform metadata
        print(f"   🔬 Extracting metadata for {len(gse.gpls)} platforms...")
        for gpl_name, gpl in gse.gpls.items():
            metadata['platforms'][gpl_name] = {
                'title': gpl.metadata.get('title', [''])[0],
                'organism': gpl.metadata.get('organism', [''])[0],
                'technology': gpl.metadata.get('technology', [''])[0],
                'distribution': gpl.metadata.get('distribution', [''])[0],
                'description': gpl.metadata.get('description', [''])[0],
                'submission_date': gpl.metadata.get('submission_date', [''])[0],
                'last_update_date': gpl.metadata.get('last_update_date', [''])[0]
            }
        
        print(f"   ✅ Successfully extracted metadata for {accession}")
        return metadata, gse
        
    except Exception as e:
        print(f"   ❌ Error extracting GEO metadata: {e}")
        return {}, None

def download_geo_supplementary_files(gse, dirs: Dict[str, Path]) -> Dict[str, bool]:
    """Download supplementary files from GEO which should contain h5ad files."""
    
    print("\n📦 DOWNLOADING GEO SUPPLEMENTARY FILES")
    print("=" * 50)
    
    supp_dir = dirs['supplementary']
    download_results = {'supplementary_files': False}
    
    try:
        print(f"   📂 Downloading supplementary files to: {supp_dir}")
        
        # Check if files already exist
        existing_files = list(supp_dir.glob('*'))
        if existing_files:
            print(f"   ✅ Found {len(existing_files)} existing files in {supp_dir}")
            download_results['supplementary_files'] = True
        else:
            # Download supplementary files
            gse.download_supplementary_files(directory=str(supp_dir))
            
            # Check if download was successful
            downloaded_files = list(supp_dir.glob('*'))
            if downloaded_files:
                print(f"   ✅ Successfully downloaded {len(downloaded_files)} supplementary files")
                download_results['supplementary_files'] = True
            else:
                print(f"   ❌ No supplementary files downloaded")
                
    except Exception as e:
        print(f"   ❌ Error downloading supplementary files: {e}")
        print(f"   🔍 You may need to manually download files from:")
        print(f"       https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218661")
    
    return download_results

def download_h5ad_files_manually(gse, dirs: Dict[str, Path]) -> Dict[str, bool]:
    """Manually download h5ad files using URLs extracted from GEO metadata."""
    
    print("\n📥 EXTRACTING H5AD URLS FROM GEO AND DOWNLOADING")
    print("=" * 50)
    
    supp_dir = dirs['supplementary']
    download_results = {'h5ad_head': False, 'h5ad_body': False}
    
    # Extract supplementary file URLs from GEO metadata
    h5ad_files = {}
    
    # Check GSE-level supplementary files
    if hasattr(gse, 'metadata') and 'supplementary_file' in gse.metadata:
        supp_files = gse.metadata['supplementary_file']
        print(f"   🔍 Found {len(supp_files)} GSE-level supplementary files")
        
        for supp_file in supp_files:
            if '.h5ad' in supp_file.lower():
                print(f"      📄 H5AD file found: {supp_file}")
                
                # Determine tissue type from filename
                if 'head' in supp_file.lower():
                    tissue = 'head'
                elif 'body' in supp_file.lower():
                    tissue = 'body'
                else:
                    tissue = 'unknown'
                
                # Extract filename from URL
                filename = supp_file.split('/')[-1]
                
                # Convert FTP URLs to HTTP URLs for requests compatibility
                url = supp_file
                if url.startswith('ftp://ftp.ncbi.nlm.nih.gov'):
                    url = url.replace('ftp://ftp.ncbi.nlm.nih.gov', 'https://ftp.ncbi.nlm.nih.gov')
                
                h5ad_files[tissue] = {
                    'url': url,
                    'filename': filename
                }
    
    # Also check individual GSM samples for supplementary files
    for gsm_name, gsm in gse.gsms.items():
        if hasattr(gsm, 'metadata') and 'supplementary_file' in gsm.metadata:
            supp_files = gsm.metadata['supplementary_file']
            for supp_file in supp_files:
                if '.h5ad' in supp_file.lower():
                    print(f"      📄 GSM H5AD file found in {gsm_name}: {supp_file}")
                    
                    # Determine tissue type from filename or GSM metadata
                    tissue = 'unknown'
                    if 'head' in supp_file.lower():
                        tissue = 'head'
                    elif 'body' in supp_file.lower():
                        tissue = 'body'
                    elif hasattr(gsm, 'metadata') and 'source_name_ch1' in gsm.metadata:
                        source = gsm.metadata['source_name_ch1'][0].lower()
                        if 'head' in source:
                            tissue = 'head'
                        elif 'body' in source:
                            tissue = 'body'
                    
                    filename = supp_file.split('/')[-1]
                    
                    # Only add if we don't already have this tissue or if this looks more comprehensive
                    if tissue not in h5ad_files or 'combined' in filename.lower():
                        # Convert FTP URLs to HTTP URLs for requests compatibility
                        url = supp_file
                        if url.startswith('ftp://ftp.ncbi.nlm.nih.gov'):
                            url = url.replace('ftp://ftp.ncbi.nlm.nih.gov', 'https://ftp.ncbi.nlm.nih.gov')
                        
                        h5ad_files[tissue] = {
                            'url': url,
                            'filename': filename
                        }
    
    # If no h5ad files found in metadata, construct URLs based on GEO conventions
    if not h5ad_files:
        print("   ⚠️  No h5ad files found in GEO metadata, constructing standard URLs...")
        accession = gse.get_accession()
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}/suppl/"
        
        h5ad_files = {
            'head': {
                'url': f"{base_url}{accession}_adata_head_S_v1.0.h5ad.gz",
                'filename': f"{accession}_adata_head_S_v1.0.h5ad.gz"
            },
            'body': {
                'url': f"{base_url}{accession}_adata_body_S_v1.0.h5ad.gz", 
                'filename': f"{accession}_adata_body_S_v1.0.h5ad.gz"
            }
        }
        print(f"   🔧 Constructed URLs for {accession}")
    
    print(f"\n   📋 H5AD files to download:")
    for tissue, file_info in h5ad_files.items():
        print(f"      🧬 {tissue.title()}: {file_info['filename']}")
        print(f"         URL: {file_info['url']}")
    
    # Download each h5ad file
    for tissue, file_info in h5ad_files.items():
        file_path = supp_dir / file_info['filename']
        
        if file_path.exists():
            print(f"\n   ✅ {tissue.title()} h5ad file already exists: {file_path}")
            download_results[f'h5ad_{tissue}'] = True
            continue
            
        try:
            print(f"\n   📡 Downloading {tissue} h5ad file...")
            print(f"      Source: {file_info['url']}")
            print(f"      Destination: {file_path}")
            
            response = requests.get(file_info['url'], stream=True)
            response.raise_for_status()
            
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r      Progress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='', flush=True)
            
            print(f"\n   ✅ Successfully downloaded {tissue} h5ad file: {file_path}")
            download_results[f'h5ad_{tissue}'] = True
            
        except Exception as e:
            print(f"\n   ❌ Error downloading {tissue} h5ad file: {e}")
            continue
    
    return download_results

def process_h5ad_files(dirs: Dict[str, Path]) -> Dict[str, any]:
    """Process h5ad files and extract information."""
    
    print("\n🔬 PROCESSING H5AD FILES")
    print("=" * 50)
    
    supp_dir = dirs['supplementary']
    data_dir = dirs['data']
    h5ad_info = {
        'files_found': [],
        'files_processed': {},
        'total_cells': 0,
        'total_genes': 0,
        'datasets': {}
    }
    
    # Find h5ad files
    h5ad_files = list(supp_dir.glob('*.h5ad'))
    if not h5ad_files:
        # Also check for compressed files
        h5ad_files.extend(list(supp_dir.glob('*.h5ad.gz')))
    
    print(f"   🔍 Found {len(h5ad_files)} h5ad files")
    
    for h5ad_file in h5ad_files:
        try:
            print(f"   📖 Processing: {h5ad_file.name}")
            
            # Read h5ad file
            if h5ad_file.suffix == '.gz':
                # Handle compressed files
                print(f"      📂 Decompressing {h5ad_file.name}")
                with gzip.open(h5ad_file, 'rb') as f_in:
                    decompressed_file = h5ad_file.with_suffix('')
                    with open(decompressed_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                adata = sc.read_h5ad(decompressed_file)
                
                # Move to data root with clean name and remove compressed file
                tissue_type = 'unknown'
                filename_lower = h5ad_file.name.lower()
                if 'head' in filename_lower:
                    tissue_type = 'head'
                elif 'body' in filename_lower:
                    tissue_type = 'body'
                
                final_filename = f"afca_{tissue_type}.h5ad"
                final_path = data_dir / final_filename
                
                print(f"      📁 Moving to data root: {final_path}")
                decompressed_file.rename(final_path)
                
                print(f"      🗑️  Removing compressed file: {h5ad_file}")
                h5ad_file.unlink()
                
                # Update file reference for processing
                h5ad_file = final_path
                
            else:
                adata = sc.read_h5ad(h5ad_file)
            
            # Extract basic information
            file_info = {
                'filename': h5ad_file.name,
                'filepath': str(h5ad_file),
                'n_obs': adata.n_obs,
                'n_vars': adata.n_vars,
                'obs_columns': list(adata.obs.columns),
                'var_columns': list(adata.var.columns),
                'uns_keys': list(adata.uns.keys()) if adata.uns else [],
                'obsm_keys': list(adata.obsm.keys()) if adata.obsm else [],
                'varm_keys': list(adata.varm.keys()) if adata.varm else [],
            }
            
            # Identify tissue type from filename or metadata
            tissue_type = 'unknown'
            filename_lower = h5ad_file.name.lower()
            if 'head' in filename_lower:
                tissue_type = 'head'
            elif 'body' in filename_lower:
                tissue_type = 'body'
            elif 'combined' in filename_lower or 'full' in filename_lower:
                tissue_type = 'combined'
            
            file_info['tissue_type'] = tissue_type
            
            # Extract age information if available
            if 'age' in adata.obs.columns:
                ages = adata.obs['age'].unique()
                file_info['ages'] = list(ages)
                print(f"      📅 Ages found: {ages}")
            
            # Extract cell type information if available
            cell_type_cols = [col for col in adata.obs.columns 
                            if any(term in col.lower() for term in ['cell_type', 'celltype', 'annotation', 'cluster'])]
            if cell_type_cols:
                file_info['cell_type_columns'] = cell_type_cols
                for col in cell_type_cols[:2]:  # Limit to first 2 to avoid too much output
                    cell_types = adata.obs[col].unique()
                    file_info[f'{col}_unique_values'] = len(cell_types)
                    print(f"      🧬 {col}: {len(cell_types)} unique values")
            
            h5ad_info['files_processed'][h5ad_file.name] = file_info
            h5ad_info['total_cells'] += adata.n_obs
            h5ad_info['total_genes'] = max(h5ad_info['total_genes'], adata.n_vars)
            
            print(f"      ✅ Processed: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
            
        except Exception as e:
            print(f"      ❌ Error processing {h5ad_file.name}: {e}")
            continue
    
    h5ad_info['files_found'] = [f.name for f in h5ad_files]
    
    if h5ad_info['files_processed']:
        print(f"\n   📊 Summary:")
        print(f"      📁 Files processed: {len(h5ad_info['files_processed'])}")
        print(f"      🧬 Total cells: {h5ad_info['total_cells']:,}")
        print(f"      🧮 Max genes: {h5ad_info['total_genes']:,}")
        print(f"      📂 Final h5ad files location: data/")
    
    return h5ad_info

def create_afca_data_info() -> Dict:
    """Create comprehensive information about AFCA dataset."""
    
    print("\n📋 CREATING AFCA DATA INFORMATION")
    print("=" * 50)
    
    afca_info = {
        'dataset_name': 'Aging Fly Cell Atlas (AFCA)',
        'accession': 'GSE218661',
        'publication': {
            'title': 'Aging Fly Cell Atlas identifies exhaustive aging features at cellular resolution',
            'authors': 'Lu, T.-C., Brbić, M., Park, Y.-J., et al.',
            'journal': 'Science',
            'year': 2023,
            'volume': 380,
            'issue': 6650,
            'doi': '10.1126/science.adg0934'
        },
        'data_description': {
            'organism': 'Drosophila melanogaster',
            'technology': '10x Chromium single-nucleus RNA-seq (snRNA-seq)',
            'total_nuclei': '868,000+',
            'cell_types': 163,
            'ages': ['5d', '30d', '50d', '70d'],
            'sexes': ['Male', 'Female'],
            'tissues': ['Head', 'Body']
        },
        'data_access': {
            'web_portal': 'https://hongjielilab.org/afca/',
            'geo_repository': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218661',
            'zenodo': 'https://doi.org/10.5281/zenodo.7853649',
            'cellxgene_head': 'https://cellxgene.cziscience.com/',
            'cellxgene_body': 'https://cellxgene.cziscience.com/',  
            'cellxgene_combined': 'https://cellxgene.cziscience.com/'
        }
    }
    
    print("   ✅ Created comprehensive AFCA dataset information")
    return afca_info

def save_metadata_files(metadata: Dict, afca_info: Dict, h5ad_info: Dict, dirs: Dict[str, Path]) -> None:
    """Save all collected metadata to organized files."""
    
    print("\n💾 SAVING METADATA FILES")
    print("=" * 50)
    
    try:
        # Save GEO metadata
        geo_file = dirs['metadata'] / 'geo_metadata.json'
        with open(geo_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved GEO metadata: {geo_file}")
        
        # Save AFCA dataset information
        afca_file = dirs['metadata'] / 'afca_dataset_info.json'
        with open(afca_file, 'w', encoding='utf-8') as f:
            json.dump(afca_info, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved AFCA info: {afca_file}")
        
        # Save h5ad processing results
        h5ad_file = dirs['metadata'] / 'h5ad_processing_info.json'
        with open(h5ad_file, 'w', encoding='utf-8') as f:
            json.dump(h5ad_info, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved h5ad info: {h5ad_file}")
        
        # Create summary metadata
        summary = {
            'retrieval_date': pd.Timestamp.now().isoformat(),
            'accession': metadata.get('accession', 'GSE218661'),
            'title': metadata.get('title', afca_info['dataset_name']),
            'organism': afca_info['data_description']['organism'],
            'total_samples': metadata.get('sample_count', 'Unknown'),
            'technology': afca_info['data_description']['technology'],
            'h5ad_files_found': len(h5ad_info.get('files_found', [])),
            'h5ad_files_processed': len(h5ad_info.get('files_processed', {})),
            'total_cells_in_h5ad': h5ad_info.get('total_cells', 0),
            'max_genes_in_h5ad': h5ad_info.get('total_genes', 0)
        }
        
        summary_file = dirs['metadata'] / 'retrieval_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved retrieval summary: {summary_file}")
        
        # Save sample information as CSV if available
        if metadata.get('samples'):
            samples_data = []
            for gsm_id, sample_info in metadata['samples'].items():
                row = {'sample_id': gsm_id}
                row.update(sample_info)
                # Flatten characteristics list
                if isinstance(sample_info.get('characteristics_ch1'), list):
                    for i, char in enumerate(sample_info['characteristics_ch1']):
                        row[f'characteristic_{i+1}'] = char
                samples_data.append(row)
            
            samples_df = pd.DataFrame(samples_data)
            samples_file = dirs['metadata'] / 'samples_metadata.csv'
            samples_df.to_csv(samples_file, index=False)
            print(f"   ✅ Saved samples metadata: {samples_file}")
        
    except Exception as e:
        print(f"   ❌ Error saving metadata: {e}")

def generate_download_instructions() -> str:
    """Generate instructions for manual data download."""
    
    instructions = """
🔽 MANUAL DOWNLOAD INSTRUCTIONS FOR AFCA DATA
==============================================

1. GEO Repository (GSE218661) - Primary Source:
   • Visit: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218661
   • Download supplementary files (look for .h5ad files)
   • Place in: data/raw/supplementary/

2. AFCA Web Portal (Interactive):
   • Visit: https://hongjielilab.org/afca/
   • Access interactive data portal for exploration

3. CellxGene Portal:
   • Search for "Aging Fly Cell Atlas"
   • URL: https://cellxgene.cziscience.com/

4. Zenodo Repository (Analysis Code & Data):
   • Visit: https://doi.org/10.5281/zenodo.7853649

Expected h5ad files:
- Head data: Contains head tissue single-nucleus data
- Body data: Contains body tissue single-nucleus data  
- Combined data: May contain integrated head+body data

After manual download, place files in: data/raw/supplementary/
Then re-run this script to process the downloaded files.
"""
    
    return instructions

def main():
    """Main data retrieval workflow for AFCA GSE218661."""
    
    print("🧬 AGING FLY CELL ATLAS (AFCA) - DATA RETRIEVAL")
    print("=" * 60)
    print("🎯 Target: GSE218661 (Aging Fly Cell Atlas)")
    print("📋 Goal: Download h5ad files and extract comprehensive metadata")
    print()
    
    # Setup directories
    dirs = setup_directories()
    
    # Extract GEO metadata and get GSE object
    geo_metadata, gse = extract_geo_metadata("GSE218661")
    
    if not gse:
        print("❌ Failed to retrieve GEO metadata. Cannot proceed.")
        sys.exit(1)
    
    # Download supplementary files from GEO (this gets the sample directories but not h5ad files)
    download_results = download_geo_supplementary_files(gse, dirs)
    
    # Manually download the h5ad files
    h5ad_download_results = download_h5ad_files_manually(gse, dirs)
    download_results.update(h5ad_download_results)
    
    # Process h5ad files
    h5ad_info = process_h5ad_files(dirs)
    
    # Create comprehensive AFCA information
    afca_info = create_afca_data_info()
    
    # Save all metadata
    save_metadata_files(geo_metadata, afca_info, h5ad_info, dirs)
    
    # Generate download instructions
    instructions = generate_download_instructions()
    instructions_file = dirs['data'] / 'DOWNLOAD_INSTRUCTIONS.txt'
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    # Final summary
    print("\n🎉 DATA RETRIEVAL SUMMARY")
    print("=" * 50)
    
    print("📂 Directory Structure Created:")
    for name, path in dirs.items():
        print(f"   ✅ {name}: {path}")
    
    print(f"\n📊 Download Results:")
    for category, success in download_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {category}")
    
    print(f"\n🔬 H5AD Processing Results:")
    print(f"   📁 Files found: {len(h5ad_info.get('files_found', []))}")
    print(f"   ✅ Files processed: {len(h5ad_info.get('files_processed', {}))}")
    if h5ad_info.get('total_cells', 0) > 0:
        print(f"   🧬 Total cells: {h5ad_info['total_cells']:,}")
        print(f"   🧮 Max genes: {h5ad_info['total_genes']:,}")
    
    print(f"\n📋 Metadata Files Created:")
    metadata_files = [
        'geo_metadata.json',
        'afca_dataset_info.json', 
        'h5ad_processing_info.json',
        'retrieval_summary.json',
        'samples_metadata.csv',
        'DOWNLOAD_INSTRUCTIONS.txt'
    ]
    
    for filename in metadata_files:
        if filename == 'DOWNLOAD_INSTRUCTIONS.txt':
            file_path = dirs['data'] / filename
        else:
            file_path = dirs['metadata'] / filename
        if file_path.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ⚠️  {filename} (may not be created)")
    
    if not any([download_results.get('h5ad_head', False), download_results.get('h5ad_body', False)]):
        print(f"\n⚠️  H5AD FILES NOT DOWNLOADED")
        print("📖 Please check the manual download function or download directly from:")
        print("🌐 Head: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE218nnn/GSE218661/suppl/GSE218661_adata_head_S_v1.0.h5ad.gz")
        print("🌐 Body: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE218nnn/GSE218661/suppl/GSE218661_adata_body_S_v1.0.h5ad.gz")
    else:
        print(f"✅ Successfully downloaded h5ad files")
    
    if h5ad_info.get('files_processed'):
        print(f"✅ Successfully processed {len(h5ad_info['files_processed'])} h5ad files")
        
        # Show tissue breakdown
        tissues = {}
        for filename, info in h5ad_info['files_processed'].items():
            tissue = info.get('tissue_type', 'unknown')
            if tissue not in tissues:
                tissues[tissue] = {'files': 0, 'cells': 0}
            tissues[tissue]['files'] += 1
            tissues[tissue]['cells'] += info.get('n_obs', 0)
        
        print(f"\n📊 Tissue Breakdown:")
        for tissue, stats in tissues.items():
            print(f"   🧬 {tissue.title()}: {stats['files']} files, {stats['cells']:,} cells")
    else:
        print(f"⚠️  No h5ad files found or processed")
        print("   Files may need to be downloaded manually or decompressed")
    
    print(f"\n🎯 NEXT STEPS:")
    print("   1. Verify h5ad files were downloaded successfully")
    print("   2. Check data/raw/supplementary/ for .h5ad.gz files")
    print("   3. Run this script again to process downloaded files")
    print("   4. Run 02_data_exploration.py to analyze the data")
    print("   5. Visit AFCA web portal for interactive exploration")
    
    print(f"\n💾 All metadata saved to: {dirs['metadata']}")
    print("🚀 Ready for data exploration phase!")

if __name__ == "__main__":
    main() 