#!/usr/bin/env python3
"""
Selective HuggingFace Dataset Upload Script for Aging Fly Cell Atlas

This script uploads only the essential files for the aging fly dataset:
- Processed parquet files (head dataset only, body WIP)
- JSON metadata files (including unstructured metadata)
- README.md
- LICENSE
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from typing import List, Optional
import typer

app = typer.Typer()

def get_files_to_upload(base_path: Path) -> List[tuple[Path, str]]:
    """
    Get list of (local_path, repo_path) tuples for files to upload
    """
    files_to_upload = []
    
    # Processed data files
    processed_dir = base_path / "processed"
    
    # Core dataset files (complete head and body datasets)
    core_patterns = [
        # HEAD TISSUE
        "aging_fly_head_expression.parquet",
        "aging_fly_head_sample_metadata.parquet", 
        "aging_fly_head_feature_metadata.parquet",
        "aging_fly_head_projection_X_pca.parquet",
        "aging_fly_head_projection_X_tsne.parquet",
        "aging_fly_head_projection_X_umap.parquet",
        "aging_fly_head_unstructured_metadata.json",
        # BODY TISSUE
        "aging_fly_body_expression.parquet",
        "aging_fly_body_sample_metadata.parquet", 
        "aging_fly_body_feature_metadata.parquet",
        "aging_fly_body_projection_X_pca.parquet",
        "aging_fly_body_projection_X_tsne.parquet",
        "aging_fly_body_projection_X_umap.parquet",
        "aging_fly_body_unstructured_metadata.json"
    ]
    
    # Additional metadata files for completeness
    additional_patterns = [
        "head_legacy_summary.json",
        "fly_exploration_results.json"
    ]
    
    # Data retrieval metadata (from data/ directory)
    data_dir = base_path / "data" / "metadata"
    if data_dir.exists():
        retrieval_metadata = [
            "retrieval_summary.json",
            "afca_dataset_info.json",
            "h5ad_processing_info.json"
        ]
        for metadata_file in retrieval_metadata:
            file_path = data_dir / metadata_file
            if file_path.exists():
                # Put in metadata/ subdirectory in repo
                files_to_upload.append((file_path, f"metadata/{metadata_file}"))
            else:
                print(f"ℹ️  Optional retrieval metadata not found: {metadata_file}")
    
    # Add core required files
    for pattern in core_patterns:
        file_path = processed_dir / pattern
        if file_path.exists():
            files_to_upload.append((file_path, file_path.name))
        else:
            print(f"⚠️  Missing required file: {pattern}")
    
    # Add additional metadata files
    for pattern in additional_patterns:
        file_path = processed_dir / pattern
        if file_path.exists():
            files_to_upload.append((file_path, file_path.name))
    
    # Documentation files
    doc_files = ["README.md", "LICENSE"]
    for doc_file in doc_files:
        file_path = base_path / doc_file
        if file_path.exists():
            files_to_upload.append((file_path, doc_file))
        else:
            print(f"⚠️  Missing documentation file: {doc_file}")
    
    # Processing scripts (put in root and strip numbering prefix)
    scripts_dir = base_path / "scripts"
    script_mappings = {
        "01_data_retrieval.py": "data_retrieval.py",   # Strip 01_ prefix 
        "03_data_processing.py": "data_processing.py"  # Strip 03_ prefix
    }
    for script_file, repo_name in script_mappings.items():
        file_path = scripts_dir / script_file
        if file_path.exists():
            files_to_upload.append((file_path, repo_name))
        else:
            print(f"⚠️  Missing processing script: {script_file}")
    
    # TODO: HuggingFace dataset loading script (if created)
    hf_loading_script = base_path / "aging_fly_atlas.py"
    if hf_loading_script.exists():
        files_to_upload.append((hf_loading_script, "aging_fly_atlas.py"))
        print("✅ HuggingFace dataset loading script found")
    else:
        print("ℹ️  Optional: HuggingFace dataset loading script (aging_fly_atlas.py) not found")
        print("    Dataset will still load properly via standard HuggingFace methods")
    
    return files_to_upload

@app.command()
def upload_dataset(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g., 'longevity-gpt/aging-fly-cell-atlas')"),
    token: Optional[str] = typer.Option(None, help="HuggingFace token (or set HF_TOKEN env var)"),
    base_path: str = typer.Option(".", help="Base path of the project"),
    dry_run: bool = typer.Option(False, help="Show what would be uploaded without doing it"),
    create_if_not_exists: bool = typer.Option(True, help="Create repo if it doesn't exist")
):
    """
    Upload aging fly cell atlas dataset to HuggingFace Hub
    """
    base_path = Path(base_path).resolve()
    
    # Get files to upload
    files_to_upload = get_files_to_upload(base_path)
    
    print(f"📁 Base path: {base_path}")
    print(f"🎯 Repository: {repo_id}")
    print(f"📊 Files to upload: {len(files_to_upload)}")
    print("🧬 Dataset: Aging Fly Cell Atlas (Complete head and body datasets)")
    print()
    
    # Show files categorized by type
    core_files = []
    metadata_files = []
    doc_files = []
    script_files = []
    
    for local_path, repo_path in files_to_upload:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        
        if repo_path.endswith('.parquet'):
            core_files.append((repo_path, size_mb))
        elif repo_path.endswith('.json'):
            metadata_files.append((repo_path, size_mb))
        elif repo_path.endswith('.py'):
            script_files.append((repo_path, size_mb))
        else:
            doc_files.append((repo_path, size_mb))
    
    # Display file inventory
    print("🗂️  **CORE DATASET FILES** (Head and Body tissues):")
    total_core_size = 0
    for filename, size_mb in core_files:
        total_core_size += size_mb
        print(f"  📄 {filename:<55} ({size_mb:>8.1f} MB)")
    
    print(f"\n📊 **METADATA FILES**:")
    total_meta_size = 0
    for filename, size_mb in metadata_files:
        total_meta_size += size_mb
        print(f"  📄 {filename:<55} ({size_mb:>8.1f} MB)")
    
    print(f"\n📚 **DOCUMENTATION FILES**:")
    total_doc_size = 0
    for filename, size_mb in doc_files:
        total_doc_size += size_mb
        print(f"  📄 {filename:<55} ({size_mb:>8.1f} MB)")
    
    print(f"\n💻 **PROCESSING SCRIPTS**:")
    total_script_size = 0
    for filename, size_mb in script_files:
        total_script_size += size_mb
        print(f"  📄 {filename:<55} ({size_mb:>8.1f} MB)")
    
    total_size = total_core_size + total_meta_size + total_doc_size + total_script_size
    print(f"\n💾 **TOTAL SIZE**: {total_size:.1f} MB")
    print(f"   ├── Core data: {total_core_size:.1f} MB")
    print(f"   ├── Metadata: {total_meta_size:.1f} MB")
    print(f"   ├── Documentation: {total_doc_size:.1f} MB")
    print(f"   └── Scripts: {total_script_size:.1f} MB")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files uploaded")
        return
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repo if needed
    if create_if_not_exists:
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
                exist_ok=True
            )
            print(f"✅ Repository {repo_id} ready")
        except Exception as e:
            print(f"⚠️  Repo creation warning: {e}")
    
    # Upload files
    print(f"\n🚀 Uploading files...")
    
    for local_path, repo_path in files_to_upload:
        try:
            print(f"  📤 Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"  ✅ {repo_path} uploaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to upload {repo_path}: {e}")
    
    print(f"\n🎉 Upload complete! View at: https://huggingface.co/datasets/{repo_id}")

@app.command()
def list_files(
    base_path: str = typer.Option(".", help="Base path of the project")
):
    """
    List files that would be uploaded (dry run)
    """
    base_path = Path(base_path).resolve()
    files_to_upload = get_files_to_upload(base_path)
    
    print(f"Files to upload from {base_path}:")
    for local_path, repo_path in files_to_upload:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {repo_path:<50} ({size_mb:>8.1f} MB)")

@app.command()
def check_completeness():
    """
    Check dataset completeness for upload readiness
    """
    print("🏆 **AGING FLY CELL ATLAS - DATASET COMPLETENESS CHECK**\n")
    
    base_path = Path(".").resolve()
    processed_dir = base_path / "processed"
    
    # Required files for complete dataset (head and body)
    required_files = {
        # HEAD TISSUE
        "aging_fly_head_expression.parquet": "Head expression matrix (290K cells)",
        "aging_fly_head_sample_metadata.parquet": "Head cell metadata",
        "aging_fly_head_feature_metadata.parquet": "Head gene annotations", 
        "aging_fly_head_projection_X_pca.parquet": "Head PCA embeddings",
        "aging_fly_head_projection_X_tsne.parquet": "Head t-SNE visualization",
        "aging_fly_head_projection_X_umap.parquet": "Head UMAP visualization",
        "aging_fly_head_unstructured_metadata.json": "Head processing metadata",
        # BODY TISSUE
        "aging_fly_body_expression.parquet": "Body expression matrix (276K cells)",
        "aging_fly_body_sample_metadata.parquet": "Body cell metadata",
        "aging_fly_body_feature_metadata.parquet": "Body gene annotations", 
        "aging_fly_body_projection_X_pca.parquet": "Body PCA embeddings",
        "aging_fly_body_projection_X_tsne.parquet": "Body t-SNE visualization",
        "aging_fly_body_projection_X_umap.parquet": "Body UMAP visualization",
        "aging_fly_body_unstructured_metadata.json": "Body processing metadata"
    }
    
    documentation_files = {
        "README.md": "Dataset card with YAML config"
    }
    
    print("📊 **REQUIRED DATA FILES** (Head and Body datasets):")
    missing_core = []
    for filename, description in required_files.items():
        file_path = processed_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename:<55} ({size_mb:>8.1f} MB) - {description}")
        else:
            print(f"  ❌ {filename:<55} MISSING - {description}")
            missing_core.append(filename)
    
    print(f"\n📚 **DOCUMENTATION FILES**:")
    missing_docs = []
    for filename, description in documentation_files.items():
        file_path = base_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename:<55} ({size_mb:>8.1f} MB) - {description}")
        else:
            print(f"  ❌ {filename:<55} MISSING - {description}")
            missing_docs.append(filename)
    
    # Body dataset status
    print(f"\n🚧 **BODY DATASET STATUS**:")
    body_files = [
        "aging_fly_body_expression.parquet",
        "aging_fly_body_sample_metadata.parquet",
        "aging_fly_body_feature_metadata.parquet"
    ]
    
    body_present = 0
    for filename in body_files:
        file_path = processed_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename:<55} ({size_mb:>8.1f} MB) - Available")
            body_present += 1
        else:
            print(f"  🚧 {filename:<55} WIP - In development")
    
    if body_present > 0:
        print(f"     {body_present}/{len(body_files)} body files ready")
    else:
        print(f"     Body dataset processing in progress")
    
    # Summary
    total_required = len(required_files) + len(documentation_files)
    total_missing = len(missing_core) + len(missing_docs)
    completion_rate = ((total_required - total_missing) / total_required) * 100
    
    print(f"\n🎯 **COMPLETION SUMMARY** (Complete dataset):")
    print(f"   📊 Required files: {total_required}")
    print(f"   ✅ Present: {total_required - total_missing}")
    print(f"   ❌ Missing: {total_missing}")
    print(f"   📈 Completion rate: {completion_rate:.1f}%")
    
    if total_missing == 0:
        print(f"\n🏆 **UPLOAD READY!** All required files present for complete dataset.")
        print(f"    📝 Note: Both head and body datasets fully processed and ready!")
    else:
        print(f"\n⚠️  **MISSING FILES** - Upload readiness: {completion_rate:.1f}%")
        if missing_core:
            print(f"     Missing core files: {', '.join(missing_core)}")
        if missing_docs:
            print(f"     Missing documentation: {', '.join(missing_docs)}")

if __name__ == "__main__":
    app() 