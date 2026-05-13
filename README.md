# UCI HAR Dataset Project

This project uses the UCI Human Activity Recognition (HAR) dataset to train machine learning models that classify human activities such as walking, sitting, standing, and walking upstairs or downstairs.

The project demonstrates:

- Data preprocessing and feature scaling
- Training Random Forest and XGBoost classifiers
- Using a weighted soft-voting ensemble to improve predictions
- Generating plots like confusion matrices, learning curves, and activity distribution

---

## Folder Structure

- `src/` - Python scripts (`preprocessing.py`, `modeling.py`, `eda_har.ipynb`)
- `data/` - Original HAR dataset files
- `plot/` - Generated plots
- `.gitignore` - Ignores temporary files, outputs, and the virtual environment

---

## How to Run

1. Activate your virtual environment.
2. Preprocess the data:

python src/preprocessing.py

Train models and generate plots:
python src/modeling.py

Check the plot/ folder for visualizations.

Notes
-Random Forest gives strong overall accuracy (~93%).
-Weighted soft voting combines predictions from RF and XGBoost to improve balance.
-This code is modular, so you can adapt it later for your own sensor data.
