# evaluate.py

import torch
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Import our custom model architecture
from architectures.row_transformer import RowBasedTransformer

# --- Configuration ---
# IMPORTANT: This configuration MUST EXACTLY MATCH the configuration used for training the model.
# If these values are different, the saved weights will not load correctly.
CONFIG = {
    "model_path": "row_transformer_pretrained.pth",

    # Model Config (MATCHED TO THE NEW, LARGER MODEL)
    "num_features": 32,
    "num_classes": 5,
    "embedding_dim": 256,      # Updated
    "nhead": 8,
    "num_encoder_layers": 6,   # Updated
    "dim_feedforward": 512,    # Updated
    "dropout": 0.1,

    # Evaluation Config
    "test_size": 0.25,
}

def load_pretrained_model():
    """Loads the pre-trained model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ... (model instantiation is correct) ...
    model = RowBasedTransformer(...)
    
    # Load the saved weights
    try:
        # --- THE FIX FOR THE WARNING ---
        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{CONFIG['model_path']}'.")
        print("Please run train.py first to generate the model file.")
        exit()
        
    model.eval()
    
    print(f"Model loaded successfully from {CONFIG['model_path']} and set to evaluation mode.")
    return model, device


def preprocess_and_pad(X, y):
    """
    Prepares a real-world dataset for our model.
    1. Scales the features.
    2. Pads the feature dimension to match what the model expects.
    """
    # 1. Scale features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Pad features if necessary
    current_num_features = X_scaled.shape[1]
    expected_num_features = CONFIG["num_features"]
    
    if current_num_features < expected_num_features:
        # Create a matrix of zeros to pad with
        padding_width = expected_num_features - current_num_features
        padding = np.zeros((X_scaled.shape[0], padding_width))
        # Concatenate the original data with the padding
        X_padded = np.hstack([X_scaled, padding])
        print(f"Padded dataset from {current_num_features} to {expected_num_features} features.")
    elif current_num_features > expected_num_features:
        print(f"WARNING: Dataset has more features ({current_num_features}) than model expects ({expected_num_features}). Truncating features.")
        X_padded = X_scaled[:, :expected_num_features]
    else:
        X_padded = X_scaled

    return pd.DataFrame(X_padded), pd.Series(y)


def evaluate_on_dataset(model, device, X, y):
    """
    Performs the full in-context evaluation on a single dataset.
    """
    # Split the dataset into a context (for the prompt) and a test set (for evaluation)
    X_context, X_test, y_context, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=y, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_context_tensor = torch.from_numpy(X_context.values).float()
    X_test_tensor = torch.from_numpy(X_test.values).float()
    
    # Create the full prompt for the model. Shape: (1, num_total_rows, num_features)
    # We add a batch dimension of 1.
    prompt_x = torch.cat([X_context_tensor, X_test_tensor], dim=0).unsqueeze(0).to(device)
    
    test_start_index = len(X_context_tensor)
    
    # --- Perform Inference ---
    # `torch.no_grad()` is a context manager that disables gradient calculation.
    # This is essential for evaluation as it reduces memory consumption and speeds up computation.
    with torch.no_grad():
        output = model(prompt_x)
    
    # Extract the predictions for the test portion of the prompt
    # Output shape is (1, seq_len, num_classes), so we access the first batch item.
    test_predictions_logits = output[0, test_start_index:, :]
    
    # Convert logits to final class predictions by taking the argmax
    predicted_labels = torch.argmax(test_predictions_logits, dim=1).cpu().numpy()
    
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    
    return accuracy


def main():
    """Main evaluation script."""
    model, device = load_pretrained_model()
    
    # --- Define the datasets to evaluate on ---
    datasets_to_test = {
        "Breast Cancer": load_breast_cancer(as_frame=True),
        "Wine": load_wine(as_frame=True)
    }
    
    all_accuracies = []
    
    print("\n--- Starting Evaluation ---")
    for name, data in datasets_to_test.items():
        print(f"\nEvaluating on: {name} dataset")
        X = data.data
        y = data.target
        
        # Prepare the data
        X_padded, y_processed = preprocess_and_pad(X, y)
        
        # Get the accuracy
        accuracy = evaluate_on_dataset(model, device, X_padded, y_processed)
        all_accuracies.append(accuracy)
        
        print(f"==> Zero-Shot Accuracy on {name}: {accuracy:.4f}")
        
    print("\n--- Evaluation Finished ---")
    print(f"Average Zero-Shot Accuracy across all datasets: {np.mean(all_accuracies):.4f}")


if __name__ == "__main__":
    main()