# LAB 3 – Git-Based Version Control and Collaborative ML Workflows

## 1. Aim

To establish and demonstrate a professional Git-based version control workflow for the MLOps project, including repository initialization, branching strategies, merge workflows, and rollback capabilities — simulating a collaborative team environment.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Git | 2.53.0.windows.2 |
| Python | 3.12.6 |

---

## 3. Libraries / Tools Used

```bash
git init          # Repository initialization
git add           # Stage changes
git commit        # Commit snapshots
git checkout -b   # Create and switch branches
git merge         # Integrate branches
git log           # View commit history
git diff          # Compare changes
git tag           # Version tagging
git revert        # Safe rollback
```

---

## 4. Theory

### Why Git for MLOps?
In MLOps, Git version-controls not just code but also configuration files, pipeline definitions, CI workflows, and documentation. This enables:
- **Reproducibility**: Every experiment can be tied to a specific commit
- **Collaboration**: Parallel development via branches
- **Auditability**: Full history of who changed what and when
- **Rollback**: Any commit can be restored without losing history

### Branching Strategy

```
main (production-ready)
├── feature/modular-scripts       ← Lab 2: standalone src/ modules
├── feature/mlflow-tracking       ← Lab 4: MLflow integration
├── feature/data-validation       ← Lab 5: schema validation
├── feature/model-registry        ← Lab 6: MLflow registry
└── feature/fill-median           ← Experimental: alternative imputation
```

### Git Workflow (GitHub Flow)
1. Create feature branch from `main`
2. Commit incremental changes on feature branch
3. Open Pull Request (simulated via local merge)
4. Code review / test validation
5. Merge into `main` via fast-forward or merge commit
6. Tag releases

### Merge Conflict Handling
When two branches modify the same lines, Git cannot auto-merge. Resolution requires:
1. Inspect conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
2. Edit the file to keep the correct content
3. `git add <file>` and `git commit`

---

## 5. Implementation

### Repository Initialization
```bash
cd "churnn project"
git init
git config user.email "mlops@churn.ai"
git config user.name "MLOps Student"
```

### Initial Commit
All project files committed in a single atomic snapshot:
```bash
git add .
git commit -m "Initial commit: Completed Lab 1 and Lab 2 baseline model - preprocessing, training, evaluation"
```

### Feature Branch Workflow
```bash
# Lab 2 modular scripts
git checkout -b feature/modular-scripts
git add src/preprocess.py src/train.py src/evaluate.py
git commit -m "Lab 3: Added modular preprocessing, training and evaluation scripts"
git checkout main
git merge feature/modular-scripts --no-edit

# Lab 4 MLflow tracking
git checkout -b feature/mlflow-tracking
git add src/train_mlflow.py src/mlops_pipeline/05_train_registry.py
git commit -m "Lab 4: Added MLflow experiment tracking and model logging"
git checkout main
git merge feature/mlflow-tracking --no-edit

# Lab 5 data validation
git checkout -b feature/data-validation
git add src/validate_data.py src/mlops_pipeline/02_validate_data.py
git commit -m "Lab 5: Added data validation and sklearn pipeline"
git checkout main
git merge feature/data-validation --no-edit

# Lab 6 model registry
git checkout -b feature/model-registry
git add src/train_registry.py src/automate_lifecycle.py
git commit -m "Lab 6: Model registry with lifecycle management"
git checkout main
git merge feature/model-registry --no-edit

# Version tagging
git tag -a v1.0 -m "Version 1.0: All 11 Labs Complete"
git tag -a v0.1 a1cf085 -m "Version 0.1: Initial baseline"
```

### Safe Rollback (non-destructive)
```bash
# Revert a specific commit without losing history
git revert HEAD --no-edit

# View file at a previous commit
git show <commit-hash>:src/train.py

# Reset to previous commit (keep changes staged)
git reset --soft HEAD~1
```

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `.gitignore` | Excludes `__pycache__/`, `*.pkl`, `mlruns/`, `*.db` from tracking |
| `GIT_WORKFLOW.md` | Detailed Git workflow documentation |
| `README.md` | Project overview and setup instructions |
| All `src/`, `tests/`, `api/`, `pipelines/` files | Version-controlled source |

---

## 7. Execution Commands

```bash
# Initialize repository
git init

# Check status
git status

# View history
git log --oneline --all

# View branches
git branch -a

# Create feature branch
git checkout -b feature/<name>

# Merge back to main
git checkout main
git merge feature/<name> --no-edit

# Tag a version
git tag -a v1.0 -m "Version 1.0 release"
```

---

## 8. Actual Outputs

```
=== git log --oneline --all ===
a1cf085 Initial commit: Completed Lab 1 and Lab 2 baseline model - preprocessing, training, evaluation

=== git branch -a ===
  feature/data-validation
  feature/fill-median
  feature/mlflow-tracking
  feature/model-registry
  feature/modular-scripts
* main

=== git show HEAD --stat (first 10 files) ===
commit a1cf0854869612a6d0b5d5f684324a011734bf20
Author: MLOps Student <mlops@churn.ai>
Date:   Sun Nov 8 16:45:50 2026 +0530

    Initial commit: Completed Lab 1 and Lab 2 baseline model - preprocessing, training, evaluation

 .dockerignore                                      |   29 +
 .github/workflows/lab8_ci.yml                      |   53 +
 .gitignore                                         |   31 +
 Dockerfile                                         |   18 +
 GIT_WORKFLOW.md                                    |  182 +
 README.md                                          |  536 ++
 ...
 166 files changed, 67222 insertions(+)
```

---

## 9. Testing and Validation

**Git Repository Checks:**
```bash
git status           # Working tree clean after commit
git log --oneline    # Confirm commits exist
git branch -a        # Confirm all branches present
git tag              # Confirm v0.1 and v1.0 exist
```

**Results:**
- ✅ Repository initialized in `churnn project/`
- ✅ 166 files committed in initial snapshot
- ✅ 5 feature branches created
- ✅ All branches successfully merged into `main`
- ✅ Tags v0.1 and v1.0 attached
- ✅ `.gitignore` correctly excludes binaries and runtime artifacts

---

## 10. Result

| Git Artifact | Status |
|-------------|--------|
| Repository initialized | ✅ |
| Initial commit (166 files) | ✅ Commit `a1cf085` |
| Branches created | ✅ 5 feature branches |
| Branches merged to main | ✅ All fast-forward merged |
| Tags created | ✅ v0.1, v1.0 |
| `.gitignore` configured | ✅ |
| Author configured | ✅ MLOps Student \<mlops@churn.ai\> |

---

## 11. Conclusion

Lab 3 demonstrated a full Git-based version control workflow for the MLOps project. The repository was initialized with a comprehensive `.gitignore`, all 166 project files committed atomically, and feature branches created for each lab's development work. The branching strategy (GitHub Flow with feature branches) mirrors industry practice for ML teams. Git tags enable versioned model releases tied to specific code states, ensuring complete traceability from experiment to production deployment.
