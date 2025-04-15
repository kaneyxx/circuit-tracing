#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Usage Example

This script demonstrates the basic usage of the circuit analysis toolkit:
1. Creating a transformer model
2. Training the model on synthetic data
3. Creating and training transcoders
4. Analyzing the model's behavior

Usage:
    python basic_usage.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path so we can import the modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from model import Transformer
from transcoder import TranscoderConfig, Transcoder, create_transcoder_pipeline
from circuit_analysis import CircuitTracer, FeatureVector

# ============================================================================
# Configuration (modify these parameters according to your needs)
# ============================================================================

# Model parameters
MODEL_PARAMS = {
    'd_model': 128,        # Dimension of embeddings and hidden layers
    'n_layers': 4,         # Number of transformer layers
    'n_heads': 4,          # Number of attention heads
    'd_mlp': 256,          # Dimension of MLP hidden layers
    'vocab_size': 1000,    # Vocabulary size
    'max_seq_len': 64      # Maximum sequence length
}

# Training parameters
TRAINING_PARAMS = {
    'epochs': 3,           # Number of training epochs
    'batch_size': 32,      # Batch size for training
    'learning_rate': 1e-3, # Learning rate for model training
    'num_samples': 500,    # Number of training samples
    'seq_len': 16          # Length of each training sequence
}

# Transcoder parameters
TRANSCODER_PARAMS = {
    'l1_coefficient': 1e-3,    # L1 sparsity coefficient
    'epochs': 5,               # Epochs for training transcoders
    'learning_rate': 1e-3,     # Learning rate for transcoder training
    'batch_size': 16           # Batch size for transcoder training
}

# Output directory for saving results
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Utility Functions
# ============================================================================

def create_synthetic_data(vocab_size, seq_len, num_samples):
    """
    Create synthetic token data for training.
    
    Args:
        vocab_size: Size of the vocabulary
        seq_len: Length of each sequence
        num_samples: Number of sequences to generate
        
    Returns:
        Tensor of shape [num_samples, seq_len] with token IDs
    """
    print(f"Generating synthetic data: {num_samples} sequences of length {seq_len}")
    return torch.randint(0, vocab_size, (num_samples, seq_len))

def train_model(model, data, epochs, batch_size, learning_rate):
    """
    Train the transformer model on synthetic data.
    
    Args:
        model: Transformer model to train
        data: Training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of average losses per epoch
    """
    print(f"Training model for {epochs} epochs")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for i in range(0, len(data), batch_size):
            # Get batch
            batch = data[i:i+batch_size]
            
            # Split into inputs and targets (predict next token)
            inputs = batch[:, :-1]       # All but the last token
            targets = batch[:, 1:]       # All but the first token
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Reshape for cross entropy loss
            logits_flat = logits.reshape(-1, model.vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Compute loss
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Average loss for this epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "model_training_loss.png"))
    plt.close()
    
    return losses

def train_transcoders(model, data, params):
    """
    Create and train transcoders for the model.
    
    Args:
        model: Trained transformer model
        data: Training data
        params: Dictionary of transcoder training parameters
        
    Returns:
        Dictionary mapping layer indices to trained transcoders
    """
    print(f"Creating and training transcoders for {model.n_layers} layers")
    
    # Create transcoders for each layer
    transcoders = {}
    
    for layer in range(model.n_layers):
        print(f"  Training transcoder for layer {layer}")
        
        # Create transcoder configuration
        cfg = TranscoderConfig(
            d_model=model.d_model,
            d_hidden=model.d_mlp,
            hook_point=f'normalized_{layer}_ln2',
            hook_point_layer=layer,
            l1_coefficient=params['l1_coefficient'],
            is_post_ln=True,
            # Specify that this transcoder affects all subsequent layers
            downstream_layers=list(range(layer+1, model.n_layers))
        )
        
        # Create transcoder
        transcoder = Transcoder(cfg)
        
        # Create optimizer
        optimizer = optim.Adam(transcoder.parameters(), lr=params['learning_rate'])
        
        # Select a subset of data for training this transcoder
        train_subset = data[:min(100, len(data))]
        
        # Train the transcoder with downstream propagation
        from transcoder import train_transcoder
        losses = train_transcoder(
            model,
            transcoder,
            train_subset,
            optimizer,
            batch_size=params['batch_size'],
            num_epochs=params['epochs'],
            downstream_layers=list(range(layer+1, model.n_layers))
        )
        
        transcoders[layer] = transcoder
        print(f"    Final loss: {losses[-1]:.6f}")
    
    return transcoders

def analyze_model(model, transcoders, input_data):
    """
    Perform basic analysis of the model using transcoders.
    
    Args:
        model: Trained transformer model
        transcoders: Dictionary of trained transcoders
        input_data: Input data for analysis
        
    Returns:
        CircuitTracer object with results
    """
    print("Analyzing model with circuit tracer")
    
    # Create circuit tracer
    tracer = CircuitTracer(model, transcoders)
    
    # Run model on input data
    logits = tracer.run_model(input_data)
    
    # Select a feature to analyze (last token of the first sequence)
    token_idx = -1
    layer = model.n_layers - 1
    
    # Get activation at the specified position
    activation = tracer.cache[f'resid_post_{layer}'][0, token_idx]
    
    # Create feature vector
    feature_vector = FeatureVector(
        component_path=[],
        vector=activation,
        layer=layer,
        sublayer='resid_post',
        token=token_idx
    )
    
    # Trace the circuit
    print(f"Tracing circuit for layer {layer}, token {token_idx}")
    paths = tracer.trace_circuit(
        feature_vector,
        num_iters=2,
        num_branches=3
    )
    
    # Get and visualize circuit graph
    edges, nodes = tracer.get_circuit_graph(paths, add_error_nodes=True)
    fig = tracer.visualize_circuit(
        edges, nodes,
        title=f"Circuit for Layer {layer}, Token {token_idx}",
        width=1000,
        height=600
    )
    
    # Save visualization
    plt.savefig(os.path.join(OUTPUT_DIR, "circuit_visualization.png"))
    plt.close()
    
    return tracer, paths, edges, nodes

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to demonstrate basic usage."""
    print("=" * 70)
    print("Circuit Analysis Toolkit - Basic Usage Example")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Create a transformer model
    print("\n1. Creating transformer model")
    model = Transformer(**MODEL_PARAMS)
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. Create synthetic training data
    data = create_synthetic_data(
        MODEL_PARAMS['vocab_size'],
        TRAINING_PARAMS['seq_len'],
        TRAINING_PARAMS['num_samples']
    )
    
    # 3. Train the model
    print("\n2. Training the model")
    train_model(
        model, 
        data,
        TRAINING_PARAMS['epochs'],
        TRAINING_PARAMS['batch_size'],
        TRAINING_PARAMS['learning_rate']
    )
    
    # 4. Train transcoders
    print("\n3. Training transcoders")
    transcoders = train_transcoders(model, data, TRANSCODER_PARAMS)
    
    # 5. Analyze the model
    print("\n4. Analyzing the model")
    # Create a small batch for analysis
    analysis_data = data[:1, :8]  # First sequence, truncated to 8 tokens
    tracer, paths, edges, nodes = analyze_model(model, transcoders, analysis_data)
    
    # Print model prediction
    logits = model(analysis_data)
    predicted_token = torch.argmax(logits[0, -1]).item()
    
    print("\nAnalysis results:")
    print(f"Model predicted token: {predicted_token}")
    print(f"Number of paths in circuit: {len(paths)}")
    print(f"Visualization saved to: {os.path.join(OUTPUT_DIR, 'circuit_visualization.png')}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 