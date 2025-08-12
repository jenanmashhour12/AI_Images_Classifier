import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def save_classification_report(report_dict, model_name, out_dir='outputs'):
    
    """Save classification report dictionary to CSV."""
    os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    df = pd.DataFrame(report_dict).transpose()  # Convert the classification report dict to a DataFrame
    
    csv_path = os.path.join(out_dir, f'{model_name}_classification_report.csv')  # Define the CSV file path
    df.to_csv(csv_path, index=True)  # Save the DataFrame to a CSV file
    print(f'[INFO] Saved report → {csv_path}')

def save_confusion_matrix(y_true, y_pred, class_names, model_name, out_dir='outputs'):
    
    """Plot and save confusion matrix heatmap."""
    os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names) 
    # Plot the confusion matrix with formatting
    disp.plot(cmap='BuGn', xticks_rotation=45, values_format='d')  # Plot the confusion matrix with formatting
    plt.title(f'Confusion Matrix - {model_name}')  # Set plot title
    plt.tight_layout()  # Adjust layout to prevent clipping
    file_path = os.path.join(out_dir, f'{model_name}_confusion_matrix.png') # Define output image path
    plt.savefig(file_path)  # Save the plot to a file
    plt.close()  # Close the plot to free memory
    print(f'[INFO] Saved confusion matrix → {file_path}') 
