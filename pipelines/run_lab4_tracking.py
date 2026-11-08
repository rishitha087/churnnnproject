import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def execute_pipeline():
    print("=" * 60)
    print("STARTING LAB 4: MLFLOW EXPERIMENT TRACKING & REPRODUCIBILITY")
    print("=" * 60)
    
    scripts = [
        "src/preprocess.py",
        "src/train_mlflow.py",
        "src/validate_reproducibility.py"
    ]
    
    for rel_path in scripts:
        script_full = BASE_DIR / rel_path
        print(f"\n[INFO] ---> Executing {rel_path}...")
        result = subprocess.run([sys.executable, str(script_full)], cwd=str(BASE_DIR))
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {rel_path} failed with exit code {result.returncode}.")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("[SUCCESS] Lab 4 Pipeline fully executed!")
    print("=" * 60)

if __name__ == "__main__":
    execute_pipeline()
