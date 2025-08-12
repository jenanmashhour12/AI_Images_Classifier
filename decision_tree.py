# Jenan, Taleen | Section:3 #
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

def run_decision_tree(X_train, X_test, y_train, y_test, class_names=None, save_plot=True):
    
    print('Training Decision Tree...')

    # Initialize Decision Tree classifier with fixed random state for reproducibility
    dt = DecisionTreeClassifier(random_state=42)

    # Train the model on the training data
    dt.fit(X_train, y_train)

    # Predict labels for the test data
    y_pred = dt.predict(X_test)

    #classification report with precision, recall, F1-score, and accuracy
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    # Extract overall accuracy value from the report
    accuracy = report_dict['accuracy']
    
    print(f'\nDecision Tree [DT] Accuracy: {accuracy:.2%}')
    # Print summary metrics
    precision = report_dict['weighted avg']['precision']
    recall = report_dict['weighted avg']['recall']
    f1 = report_dict['weighted avg']['f1-score']
    # Print per-class evaluation metrics
    for cls in class_names:
        print(f"\nClass: {cls}")
        print(f"  Precision: {report_dict[cls]['precision']:.2%}")
        print(f"  Recall:    {report_dict[cls]['recall']:.2%}")
        print(f"  F1-Score:  {report_dict[cls]['f1-score']:.2%}")

    # optional: export tree plot
    if save_plot:
        from sklearn.tree import plot_tree
        os.makedirs('outputs', exist_ok=True)
        plt.figure(figsize=(12, 8))
        plot_tree(dt, filled=True, class_names=class_names, max_depth=2)
        plt.title('Decision Tree (Depth ≤ 2)')
        plt.savefig('outputs/decision_tree_structure.png')
        plt.close()
    return y_test, y_pred, report_dict