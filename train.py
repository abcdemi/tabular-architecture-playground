# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import numpy as np

# Import our custom modules
from data_generator import SyntheticDataGenerator
from architectures.row_transformer import RowBasedTransformer

# --- Configuration ---
# We'll use a dictionary for configuration for simplicity.
CONFIG = {
    # Data Generation Config
    "max_samples": 512,        # Max rows per dataset (controls sequence length)
    "max_features": 32,        # Max features per dataset
    "max_classes": 5,          # Max number of classes
    "test_size": 0.25,         # Proportion of each dataset to use for the test set

    # Model Config (must match the data config)
    "num_features": 32,        # Must be == max_features
    "num_classes": 5,          # Must be == max_classes
    "embedding_dim": 128,      
    "nhead": 8,
    "num_encoder_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.1,

    # Training Config
    "learning_rate": 1e-4,
    "batch_size": 16,          # Number of synthetic datasets per batch
    "num_training_steps": 5000,
    "log_interval": 100,       # How often to print the loss
    "grad_clip_value": 1.0,    # Helps prevent exploding gradients
}

def create_prompt_from_dataset(X, y, test_size):
    """
    Takes a generated dataset and prepares it for in-context learning.
    It splits the data into a training context and a test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=None
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train.values).float()
    y_train_tensor = torch.from_numpy(y_train.values).long()
    X_test_tensor = torch.from_numpy(X_test.values).float()
    y_test_tensor = torch.from_numpy(y_test.values).long()

    # The "prompt" is the concatenation of the training and test features.
    # The model will see all the features, but only the training labels.
    # We will handle passing the training labels to the model in a future step if needed.
    # For now, the model architecture only takes X as input.
    prompt_x = torch.cat([X_train_tensor, X_test_tensor], dim=0)

    # The target for the loss function is ONLY the test labels.
    target_y = y_test_tensor
    
    # We need to know where the test set begins to calculate the loss correctly.
    test_start_index = len(X_train_tensor)

    return prompt_x, target_y, test_start_index


def train():
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialization ---
    # Note: To make batching work, all datasets in a batch must have the same number of features.
    # We enforce this by setting min_features = max_features.
    generator = SyntheticDataGenerator(
        min_samples=200, max_samples=CONFIG["max_samples"],
        min_features=CONFIG["max_features"], max_features=CONFIG["max_features"],
        min_classes=2, max_classes=CONFIG["max_classes"]
    )

    model = RowBasedTransformer(
        num_features=CONFIG["num_features"],
        num_classes=CONFIG["num_classes"],
        embedding_dim=CONFIG["embedding_dim"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"]
    ).to(device)

    print(f"Model created. Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    model.train() # Set the model to training mode
    for step in range(CONFIG["num_training_steps"]):
        batch_prompts = []
        batch_targets = []
        batch_test_indices = []

        # 1. Create a batch of synthetic datasets
        for _ in range(CONFIG["batch_size"]):
            X, y = generator.generate()
            prompt_x, target_y, test_start_index = create_prompt_from_dataset(X, y, CONFIG["test_size"])
            batch_prompts.append(prompt_x)
            batch_targets.append(target_y)
            batch_test_indices.append(test_start_index)
        
        # 2. Pad the prompts to the same length within the batch
        # pad_sequence expects a list of tensors and pads them to the length of the longest tensor.
        padded_prompts = pad_sequence(batch_prompts, batch_first=True, padding_value=0).to(device)
        
        # 3. Perform the forward pass
        optimizer.zero_grad()
        output = model(padded_prompts) # Shape: (batch_size, seq_len, num_classes)
        
        # 4. Calculate the loss (the tricky part)
        total_loss = 0
        for i in range(CONFIG["batch_size"]):
            # Get the predictions for the test portion of this specific prompt
            start_idx = batch_test_indices[i]
            # Slicing the output tensor for the i-th item in the batch
            test_predictions = output[i, start_idx:, :]
            
            # Get the corresponding true labels
            true_labels = batch_targets[i].to(device)

            # Ensure the number of predictions matches the number of labels
            # This is important because padding could add extra length
            num_targets = len(true_labels)
            test_predictions = test_predictions[:num_targets, :]
            
            # Calculate loss for this single dataset and add it to the total
            loss = criterion(test_predictions, true_labels)
            total_loss += loss

        # Average the loss across the batch
        average_loss = total_loss / CONFIG["batch_size"]
        
        # 5. Backpropagation
        average_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_value"])
        optimizer.step()
        
        # 6. Logging
        if (step + 1) % CONFIG["log_interval"] == 0:
            print(f"Step {step+1}/{CONFIG['num_training_steps']}, Loss: {average_loss.item():.4f}")

    print("Training finished!")
    
    # You would typically save your model here
    # torch.save(model.state_dict(), 'row_transformer_pretrained.pth')


if __name__ == '__main__':
    train()