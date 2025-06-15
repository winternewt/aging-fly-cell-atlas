#!/usr/bin/env python3
"""
Quick launcher script for Phase 3 processing with different options
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str]) -> None:
    """Run command and handle errors"""
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
    else:
        print(f"✅ Command completed successfully")
    print("-" * 50)

def main():
    script_path = Path(__file__).parent / "03_data_processing.py"
    
    print("=== Phase 3 Data Processing Options ===\n")
    
    # Example commands
    examples = [
        # Process head only (safe)
        ["python3", str(script_path), "process", "head"],
        
        # Process body with aggressive chunking (for OOM issues)
        ["python3", str(script_path), "process", "body", "--aggressive-chunking"],
        
        # Process body, skip expression matrix if it's causing OOM
        ["python3", str(script_path), "process", "body", "--skip-expression"],
        
        # Process only metadata and projections for body
        ["python3", str(script_path), "process", "body", "--skip-expression"],
        
        # Generate summary from existing results (no reprocessing)
        ["python3", str(script_path), "summary"],
    ]
    
    print("Available commands:")
    for i, cmd in enumerate(examples, 1):
        print(f"{i}. {' '.join(cmd[2:])}")  # Skip python3 script_path
    
    print("\nDirect usage examples:")
    print("# Process head dataset only:")
    print(f"python3 {script_path} process head")
    
    print("\n# Process body with memory optimizations:")
    print(f"python3 {script_path} process body --aggressive-chunking")
    
    print("\n# Process body, skip expression matrix:")
    print(f"python3 {script_path} process body --skip-expression")
    
    print("\n# Generate summary without reprocessing:")
    print(f"python3 {script_path} summary")
    
    print("\n# Get help:")
    print(f"python3 {script_path} --help")
    
    # Ask user what to run
    print("\n" + "="*50)
    choice = input("Enter command number to run (1-5), or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        return
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(examples):
            run_command(examples[choice_num - 1])
        else:
            print("❌ Invalid choice")
    except ValueError:
        print("❌ Please enter a number or 'q'")

if __name__ == "__main__":
    main() 