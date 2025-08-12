# Jenan, Taleen | Section:3 #
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

def run_naive_bayes(X_train, X_test, y_train, y_test, class_names=None):
   
    print('Training Naive Bayes (NB)...')

    # Initialize Naive Bayes classifier
    model = GaussianNB()

    # Train the model on the training data
    model.fit(X_train, y_train)
    # Predict labels for the test data
    y_pred = model.predict(X_test)

    # Generate classification report with precision, recall, F1-score, and accuracy
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    # Print per-class evaluation metrics
    for cls in class_names:
        print(f"\nClass: {cls}")
        print(f"  Precision: {report_dict[cls]['precision']:.2%}")
        print(f"  Recall:    {report_dict[cls]['recall']:.2%}")
        print(f"  F1-Score:  {report_dict[cls]['f1-score']:.2%}")

    print(f"\nNaive Bayes Overall Accuracy: {report_dict['accuracy']:.2%}")
 
    return y_test, y_pred, report_dict