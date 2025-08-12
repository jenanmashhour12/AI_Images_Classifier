# Jenan, Taleen | Section:3 #
import argparse, os
from helpers.data_loader import load_images
import contextlib
import sys
import shutil
import random
from helpers.eval_tools import save_classification_report, save_confusion_matrix
from helpers.cv_grid import cross_validate_models
from models.naive_bayes import run_naive_bayes
from models.decision_tree import run_decision_tree
from models.mlp_model import run_mlp_model

#to manage printing while using cross validation
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def auto_sort_images(source_dir='all_data', train_dir='train', test_dir='test', split_ratio=0.8):
    # Check if the dataset already sorted
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("Training and testing folders already exist — skipping sorting.")
        return

    print("--- Sorting images into train/test folders...")

    random.seed(42)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

        for img in test_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

        print(f"[INFO] {class_name}: {len(train_images)} train, {len(test_images)} test")

    print("|| Dataset successfully split into /train and /test folders! ||")

def parse_args():
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Image Classification Comparison')
    parser.add_argument('--size', type=int, default=32, help='resize images to SIZE x SIZE')
    parser.add_argument('--cv', action='store_true', help='run cross-validation')
    parser.add_argument('--model', choices=['all', 'nb', 'dt', 'mlp'], default='all',
                        help='which model(s) to run: nb = Naive Bayes, dt = Decision Tree, mlp = MLP')
    return parser.parse_args()

def main():
    args = parse_args()
    auto_sort_images()
    print('Loading Dataset, Please Wait ...\n')
    # Load training and testing images from separate folders.
    # Each folder should have class-named subfolders (train/stop/, test/stop/, etc...)and we achieve that by using sort_test_image.py (Python Sript)
    
    X_train, y_train, label_map = load_images('train', image_size=(args.size, args.size))
    X_test, y_test, _ = load_images('test', image_size=(args.size, args.size))
    class_names = list(label_map.keys())  # Get class names from the label map

    # Run selected models (Naive Bayes, Decision Tree, MLP) default is all models:
    
    if args.model in ('all', 'nb'):
        print("\n----- Running Naive Bayes -----")
        y_true, y_pred, report = run_naive_bayes(X_train, X_test, y_train, y_test, class_names)
        save_classification_report(report, 'naive_bayes')
        save_confusion_matrix(y_true, y_pred, class_names, 'naive_bayes')

    if args.model in ('all', 'dt'):
        print("\n----- Running Decision Tree -----")
        y_true, y_pred, report = run_decision_tree(X_train, X_test, y_train, y_test, class_names)
        save_classification_report(report, 'decision_tree')
        save_confusion_matrix(y_true, y_pred, class_names, 'decision_tree')

    if args.model in ('all', 'mlp'):
        print("\n----- Running MLP (Feedforward Neural Network) -----")
        y_true, y_pred, report = run_mlp_model(X_train, X_test, y_train, y_test, class_names)
        save_classification_report(report, 'mlp')
        save_confusion_matrix(y_true, y_pred, class_names, 'mlp')

    # Performing 5-fold cross validation for model robustness:
    if args.cv:
        print('\n--- Running cross validation (5-fold) ---')
        with suppress_stdout():  # This silences all print() calls inside CV
            cv_results = cross_validate_models(X_train, y_train, cv_splits=5)
            
        for model, (mean_acc, std_acc) in cv_results.items():
            print(f'  --> {model}: {mean_acc:.2%} ± {std_acc:.2%}')

    
if __name__ == '__main__':
    main()
