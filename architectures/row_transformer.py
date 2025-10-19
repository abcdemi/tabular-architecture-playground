# architectures/row_transformer.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds information about the position of each token in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register 'pe' as a buffer, so it's part of the model's state but not a parameter.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # We add the positional encoding to the input tensor.
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class RowBasedTransformer(nn.Module):
    """
    The main model for our experiment.
    Architecture A: A Transformer that treats each row as a token.
    """
    def __init__(self, 
                 num_features: int, 
                 num_classes: int, 
                 embedding_dim: int = 256, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6, 
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        """
        Args:
            num_features (int): The number of columns in the input tabular data.
            num_classes (int): The number of target classes for the output.
            embedding_dim (int): The dimensionality of the internal embeddings. Must be divisible by nhead.
            nhead (int): The number of heads in the multi-head attention mechanism.
            num_encoder_layers (int): The number of Transformer encoder layers.
            dim_feedforward (int): The dimension of the feedforward network model in the encoder.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding_dim = embedding_dim

        # 1. The Row Embedder (an MLP)
        # Takes a row (num_features) and projects it to the embedding dimension.
        self.row_embedder = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # 3. The Transformer Core
        # PyTorch's built-in TransformerEncoder is composed of multiple TransformerEncoderLayers.
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # This is an important flag!
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 4. The Output Head (a Linear layer)
        # Maps the contextualized embedding of each row to the output classes.
        self.output_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.
        
        Args:
            src (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_features).
                                `seq_len` is the number of rows in the context.
                                
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, num_classes).
                          These are the raw logits for each row in the sequence.
        """
        # Step 1: Pass through the row embedder.
        # Shape: (batch_size, seq_len, num_features) -> (batch_size, seq_len, embedding_dim)
        embedded_src = self.row_embedder(src)
        
        # Step 2: Add positional encoding.
        # The PositionalEncoding class expects (seq_len, batch_size, dim), but our model uses
        # (batch_size, seq_len, dim) because of `batch_first=True`. We need to adjust.
        # Let's permute the dimensions for the pos_encoder and then permute back.
        embedded_src = embedded_src.permute(1, 0, 2)
        embedded_src = self.pos_encoder(embedded_src)
        embedded_src = embedded_src.permute(1, 0, 2)

        # Step 3: Pass through the Transformer Encoder.
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        contextualized_src = self.transformer_encoder(embedded_src)

        # Step 4: Pass through the output head.
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_classes)
        output = self.output_head(contextualized_src)
        
        return output


# --- HOW TO USE IT (A simple test) ---
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    N_FEATURES = 50       # Corresponds to max_features in your data generator
    N_CLASSES = 10        # Corresponds to max_classes
    EMBEDDING_DIM = 128   # Must be divisible by N_HEAD
    N_HEAD = 8            # Number of attention heads
    N_ENCODER_LAYERS = 3  # Number of Transformer layers
    DIM_FEEDFORWARD = 256 # Hidden layer size in the Transformer

    # --- Dummy Data Parameters ---
    BATCH_SIZE = 16
    SEQ_LENGTH = 512      # The number of rows in your "prompt" (train + test)

    # Instantiate the model
    model = RowBasedTransformer(
        num_features=N_FEATURES,
        num_classes=N_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        nhead=N_HEAD,
        num_encoder_layers=N_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD
    )

    print(f"Model created. Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create a dummy input tensor
    # This simulates a batch of 16 "prompts", each containing 512 rows with 50 features.
    dummy_data = torch.randn(BATCH_SIZE, SEQ_LENGTH, N_FEATURES)
    print(f"\nShape of dummy input data: {dummy_data.shape}")

    # Perform a forward pass
    output = model(dummy_data)

    # Inspect the output
    print(f"Shape of model output: {output.shape}")
    print("This matches our expectation of (batch_size, seq_len, num_classes).")