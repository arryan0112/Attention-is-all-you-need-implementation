import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    Returns:
        PE matrix of shape (seq_length, d_model)
    """
    
    # Create position column vector (seq_length, 1)
    position = np.arange(seq_length)[:, np.newaxis]
    
    # Create dimension indices (0, 2, 4, ..., d_model-2)
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )
    
    # Initialize PE matrix
    pe = np.zeros((seq_length, d_model))
    
    # Apply sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd indices
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe
