import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_all_stages():
    print("#" * 70)
    print("TELCO CUSTOMER CHURN - COMPLETE END-TO-END MLOPS PIPELINE")
    print("#" * 70)
    
    stages = [
        ("Lab 3: Baseline ML Pipeline", BASE_DIR / "pipelines" / "run_lab3_baseline.py"),
        ("Lab 4: Experiment Tracking & Reproducibility", BASE_DIR / "pipelines" / "run_lab4_tracking.py"),
        ("Lab 5: Production Data Pipeline & Schema Validation", BASE_DIR / "pipelines" / "run_lab5_pipeline.py"),
        ("Lab 6: MLflow Model Registry & Champion Lifecycle", BASE_DIR / "pipelines" / "run_lab6_registry.py"),
        ("Phase 15: Production Model Inference Demo", BASE_DIR / "src" / "predict.py")
    ]
    
    import subprocess
    for stage_name, script_path in stages:
        print("\n" + "=" * 70)
        print(f"STAGE: {stage_name}")
        print("=" * 70)
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
        if res.returncode != 0:
            print(f"\n[FATAL] Stage '{stage_name}' failed with return code {res.returncode}. Aborting.")
            sys.exit(res.returncode)
            
    print("\n" + "#" * 70)
    print("[ALL STAGES COMPLETED SUCCESSFULLY]")
    print("#" * 70)

if __name__ == "__main__":
    run_all_stages()
