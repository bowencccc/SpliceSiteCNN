# Splice Site Prediction

This project includes two main Python scripts:
- `model_train.py`: Trains a CNN model for splice site prediction using labeled sequence data.
- `score.py`: Applies the trained model to score splice junctions.

---

## 🔧 Setup Instructions

### 1. Clone or Download the Project

If you received the files directly, place them in a project directory. Otherwise:

```bash
git clone <repo_url>
cd <project_directory>

conda create -n splice_pred python=3.10 -y
conda activate splice_pred