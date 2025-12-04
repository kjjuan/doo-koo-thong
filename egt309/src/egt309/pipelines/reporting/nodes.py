import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_confusion_matrices(results_df: pd.DataFrame, output_dir: str = "data/08_reporting") -> None:
    """
    Generates and saves confusion matrix plots as PNG files for each model and threshold 
    found in the results DataFrame.
    
    Args:
        results_df: The DataFrame returned by 'evaluate_models' containing 'Model', 
                    'Threshold', and 'Confusion Matrix' columns
        output_dir: The directory where PNG files will be saved
                    Defaults to 'data/08_reporting'
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each row in the evaluation results
    for index, row in results_df.iterrows():
        model_name = row['Model']
        threshold = row['Threshold']
        cm_data = np.array(row['Confusion Matrix']) # Convert list back to numpy array

        # Initialize the figure
        plt.figure(figsize=(8, 6))
        
        # Create Heatmap
        # fmt='d' ensures numbers are integers, cmap='Blues' gives a standard blue scale
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        
        # Add labels and title
        plt.title(f'Confusion Matrix: {model_name} (Threshold: {threshold})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Construct a filename that includes model name and threshold
        # Replaces decimal points with 'p' to avoid filesystem issues (e.g., 0.5 -> 0p5)
        thresh_str = str(threshold).replace('.', 'p')
        filename = f"cm_{model_name}_thresh_{thresh_str}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save the plot
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        
        # Close the plot to free up memory
        plt.close()