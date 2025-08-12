# Jenan, Taleen | Section:3 #
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def run_mlp_model(X_train, X_test, y_train, y_test, class_names=None):
    print('Training MLP (feedforward NN)...')
    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # We've this MLPClassifier with 1 hidden layer with 100 neurons:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                        early_stopping=True, random_state=42, verbose=False)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    
    # Print evaluation metrics
    printed = set()
    for cls in class_names:
        if cls in report_dict and cls not in printed:
            print(f"\nClass: {cls}")
            print(f"  Precision: {report_dict[cls]['precision']:.2%}")
            print(f"  Recall:    {report_dict[cls]['recall']:.2%}")
            print(f"  F1-Score:  {report_dict[cls]['f1-score']:.2%}")
            printed.add(cls)

    print(f"\n MLP Overall Accuracy: {report_dict['accuracy']:.2%}")
    return y_test, y_pred, report_dict