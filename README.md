# Jenan & Taleen #
# Comparative Study of Image Classification

This project implements and compares three machine learning models on an image classification task:
- **Naive Bayes**
- **Decision Tree**
- **Feedforward Neural Network (MLP)**

We used a custom dataset of German Traffic Sign images and evaluated each model based on accuracy, precision, recall, and F1 score.

## Folder Structure

all_data/ # raw dataset (all images grouped by class)
train/ # auto-created training set (80%)
test/ # auto-created test set (20%)
helpers/ # data loading, evaluation tools, CV utils
models/ # separate scripts for NB, DT, and MLP
outputs/ # auto-saved reports and plots
main.py # main entry point (runs sorting + models)
requirements.txt # dependencies for pip

## Quick Start
```bash
pip install -r requirements.txt # some libraries we've used
python main.py               # run all models

''' when u run python main.py --model all:
    Automatically split all_data/ into train/ and test/ folders (80/20 split)
    Resize, flatten, and preprocess the images
    Train and evaluate all three models}'''

python main.py --cv           # add 5‑fold cross‑validation
python main.py --model dt     # run only Decision Tree for exp
```

Outputs for each model:
* CSV classification reports → `outputs/<model>_classification_report.csv`
* PNG confusion matrices     → `outputs/<model>_confusion_matrix.png`
* Decision tree visualisation→ `outputs/decision_tree_structure.png`
