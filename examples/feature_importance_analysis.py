#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Importance Analysis Example

This script demonstrates how to analyze feature importance in a model using the circuit
analysis toolkit. It covers:
1. Creating a model and transcoders
2. Identifying important features across layers
3. Running feature ablation experiments
4. Visualizing feature importance

Usage:
    python feature_importance_analysis.py
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from datetime import datetime

# Add parent directory to path so we can import the modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from model import Transformer
from transcoder import (
    TranscoderConfig, Transcoder, create_transcoder_pipeline,
    visualize_feature_importance, ablate_features, TranscoderReplacementContext
)
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

# Analysis parameters
ANALYSIS_PARAMS = {
    'top_k': 10,               # Number of top features to analyze
    'num_ablations': 5,        # Number of features to ablate in experiment
    'layer_to_focus': None,    # Specific layer to focus on (None for all layers)
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

def quick_train_model(model, data, epochs, batch_size, learning_rate):
    """
    Train the transformer model on synthetic data (simplified).
    
    Args:
        model: Transformer model to train
        data: Training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print(f"Quick training model for {epochs} epochs")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for i in range(0, min(100, len(data)), batch_size):
            # Get batch
            batch = data[i:i+batch_size]
            
            # Split into inputs and targets (predict next token)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Print progress
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def setup_model_and_transcoders():
    """
    Create and initialize a model and transcoders.
    
    This function sets up a transformer model and creates transcoders
    for each layer, without going through full training.
    
    Returns:
        Tuple of (model, transcoders, data)
    """
    # Create model
    model = Transformer(**MODEL_PARAMS)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create synthetic data
    data = create_synthetic_data(
        MODEL_PARAMS['vocab_size'],
        TRAINING_PARAMS['seq_len'],
        TRAINING_PARAMS['num_samples']
    )
    
    # Quick train the model
    quick_train_model(
        model,
        data,
        TRAINING_PARAMS['epochs'],
        TRAINING_PARAMS['batch_size'],
        TRAINING_PARAMS['learning_rate']
    )
    
    # Create transcoders for all layers
    print("Creating transcoders for all layers")
    transcoders = create_transcoder_pipeline(
        model,
        downstream_propagation=True  # Enable multi-layer effects
    )
    
    # Initialize transcoders (simplified, no full training)
    print("Initializing transcoders (simplified)")
    for layer, transcoder in transcoders.items():
        # Run a few steps of optimization on a small batch
        optimizer = optim.Adam(transcoder.parameters(), lr=TRANSCODER_PARAMS['learning_rate'])
        sample_batch = data[:4]
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                sample_batch,
                names_filter=[transcoder.cfg.hook_point]
            )
            activations = cache[transcoder.cfg.hook_point]
        
        # Just initialize the transcoder with a few steps
        for _ in range(10):
            optimizer.zero_grad()
            reconstructed, features = transcoder(activations)
            loss_dict = transcoder.loss_fn(activations, reconstructed, features)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
    
    return model, transcoders, data

def analyze_feature_importance(model, transcoders, data, top_k=10):
    """
    Analyze and visualize feature importance across layers.
    
    Args:
        model: Trained transformer model
        transcoders: Dictionary of trained transcoders
        data: Sample data to analyze
        top_k: Number of top features to display per layer
        
    Returns:
        Dictionary of importance scores by (layer, feature_idx)
    """
    print(f"Analyzing feature importance (top {top_k} features per layer)")
    
    # Use a small batch for analysis
    analysis_data = data[:10]
    
    # Get feature importance scores
    importance_scores = visualize_feature_importance(
        model,
        transcoders,
        analysis_data,
        top_k=top_k
    )
    
    # Print top features by importance
    print("\nTop features by importance:")
    for i, ((layer, feature_idx), score) in enumerate(
        sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    ):
        print(f"{i+1}. Layer {layer}, Feature {feature_idx}: {score:.4f}")
    
    # Visualize importance by layer
    layer_scores = {}
    for (layer, _), score in importance_scores.items():
        if layer not in layer_scores:
            layer_scores[layer] = []
        layer_scores[layer].append(score)
    
    # Calculate average importance by layer
    layer_avg_importance = {
        layer: sum(scores) / len(scores) 
        for layer, scores in layer_scores.items()
    }
    
    # Plot average importance by layer
    plt.figure(figsize=(10, 6))
    layers = sorted(layer_avg_importance.keys())
    avg_scores = [layer_avg_importance[layer] for layer in layers]
    
    plt.bar(layers, avg_scores, color='skyblue')
    plt.xlabel('Layer')
    plt.ylabel('Average Feature Importance')
    plt.title('Average Feature Importance by Layer')
    plt.xticks(layers)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_by_layer.png"))
    plt.close()
    
    # Plot feature distribution for the most important layer
    most_important_layer = max(layer_avg_importance.items(), key=lambda x: x[1])[0]
    
    # Get feature scores for this layer
    layer_feature_scores = {
        feature_idx: score 
        for (layer, feature_idx), score in importance_scores.items() 
        if layer == most_important_layer
    }
    
    plt.figure(figsize=(12, 6))
    feature_indices = sorted(layer_feature_scores.keys())
    feature_scores = [layer_feature_scores[idx] for idx in feature_indices]
    
    plt.bar(feature_indices, feature_scores, color='lightgreen')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title(f'Feature Importance Distribution (Layer {most_important_layer})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"feature_distribution_layer{most_important_layer}.png"))
    plt.close()
    
    return importance_scores

def run_ablation_experiment(model, transcoders, data, importance_scores, num_ablations=5):
    """
    Run a feature ablation experiment.
    
    Args:
        model: Trained transformer model
        transcoders: Dictionary of trained transcoders
        data: Sample data to analyze
        importance_scores: Dictionary of importance scores by (layer, feature_idx)
        num_ablations: Number of top features to ablate
        
    Returns:
        Dictionary of ablation results
    """
    print(f"\nRunning ablation experiment on top {num_ablations} features")
    
    # Use a small batch for analysis
    analysis_data = data[:5]
    
    # Select top features to ablate
    features_to_ablate = [
        key for key, _ in sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_ablations]
    ]
    
    print(f"Features selected for ablation: {features_to_ablate}")
    
    # Run ablation experiment
    ablation_results = ablate_features(
        model,
        transcoders,
        analysis_data,
        features_to_ablate
    )
    
    print(f"Ablation impact (L2 norm): {ablation_results['l2_norm']:.4f}")
    print("Top affected tokens:")
    for token_idx, impact in sorted(
        ablation_results['top_affected_tokens'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  Token {token_idx}: {impact:.4f}")
    
    # Visualize the effect of progressive ablation
    progressive_results = []
    
    # Run ablation with progressively more features
    for i in range(1, num_ablations + 1):
        current_features = features_to_ablate[:i]
        result = ablate_features(
            model,
            transcoders,
            analysis_data,
            current_features
        )
        progressive_results.append((i, result['l2_norm']))
    
    # Plot progressive ablation impact
    plt.figure(figsize=(10, 6))
    num_features, impact = zip(*progressive_results)
    
    plt.plot(num_features, impact, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Ablated Features')
    plt.ylabel('Impact (L2 Norm)')
    plt.title('Progressive Feature Ablation Impact')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(num_features)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "progressive_ablation_impact.png"))
    plt.close()
    
    return ablation_results

def visualize_feature_activation_patterns(model, transcoders, data):
    """
    Visualize activation patterns of important features.
    
    Args:
        model: Trained transformer model
        transcoders: Dictionary of trained transcoders
        data: Data to analyze
    """
    print("\nVisualizing feature activation patterns")
    
    # Use a small batch for analysis
    sample_batch = data[:10]
    
    # Select a specific layer for detailed analysis
    if ANALYSIS_PARAMS['layer_to_focus'] is not None:
        layer = ANALYSIS_PARAMS['layer_to_focus']
    else:
        # Pick the middle layer
        layer = model.n_layers // 2
    
    print(f"Focusing on layer {layer}")
    transcoder = transcoders[layer]
    
    # Get activations for this layer
    with torch.no_grad():
        _, cache = model.run_with_cache(
            sample_batch,
            names_filter=[transcoder.cfg.hook_point]
        )
        activations = cache[transcoder.cfg.hook_point]
        
        # Apply transcoder to get feature activations
        _, feature_activations = transcoder(activations)
    
    # Get top features
    feature_importance = torch.sum(feature_activations, dim=(0, 1))
    _, top_indices = torch.topk(feature_importance, k=5)
    
    # Plot activation patterns across tokens for top features
    plt.figure(figsize=(12, 8))
    
    # Get activation of top features across tokens in first sequence
    seq_activations = feature_activations[0].detach().cpu().numpy()
    
    # Plot each top feature
    for i, feature_idx in enumerate(top_indices):
        feature_idx = feature_idx.item()
        plt.subplot(len(top_indices), 1, i+1)
        plt.plot(seq_activations[:, feature_idx])
        plt.title(f'Feature {feature_idx} Activation Pattern')
        if i == len(top_indices) - 1:
            plt.xlabel('Token Position')
        plt.ylabel('Activation')
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"activation_patterns_layer{layer}.png"))
    plt.close()
    
    # Create a heatmap of top feature activations across tokens
    plt.figure(figsize=(10, 6))
    top_features_activations = seq_activations[:, top_indices.cpu().numpy()]
    
    plt.imshow(
        top_features_activations.T,
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Activation Strength')
    plt.xlabel('Token Position')
    plt.ylabel('Feature Index')
    plt.title(f'Top Feature Activations Across Tokens (Layer {layer})')
    plt.yticks(range(len(top_indices)), [f"Feature {idx.item()}" for idx in top_indices])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"feature_heatmap_layer{layer}.png"))
    plt.close()

def compare_model_outputs_with_ablation(model, transcoders, data, importance_scores):
    """
    Compare model outputs with and without specific features ablated.
    
    Args:
        model: Trained transformer model
        transcoders: Dictionary of trained transcoders
        data: Sample data to analyze
        importance_scores: Dictionary of importance scores
    """
    print("\nComparing model outputs with and without feature ablation")
    
    # Use a single sequence for comparison
    sample_data = data[:1]
    
    # Get original model output
    with torch.no_grad():
        original_output = model(sample_data)
    
    # Get top feature by importance
    top_feature = max(importance_scores.items(), key=lambda x: x[1])[0]
    top_layer, top_feature_idx = top_feature
    
    print(f"Top feature: Layer {top_layer}, Feature {top_feature_idx}")
    
    # Set up feature weights for ablation
    feature_weights = {
        top_feature: 0.0  # Completely ablate the top feature
    }
    
    # Get output with the top feature ablated
    with torch.no_grad(), TranscoderReplacementContext(
        model, transcoders, apply_feature_weights=feature_weights
    ):
        ablated_output = model(sample_data)
    
    # Calculate the difference
    diff = ablated_output - original_output
    diff_norm = torch.norm(diff).item()
    
    print(f"Output difference (L2 norm): {diff_norm:.4f}")
    
    # Plot the difference in outputs for the last token
    plt.figure(figsize=(12, 6))
    
    last_token_diff = diff[0, -1].detach().cpu().numpy()
    
    # Plot top 50 dimensions with highest difference
    top_dims = np.argsort(np.abs(last_token_diff))[-50:]
    top_diffs = last_token_diff[top_dims]
    
    plt.bar(range(len(top_dims)), top_diffs, color=[
        'red' if x < 0 else 'green' for x in top_diffs
    ])
    plt.xlabel('Dimension Index')
    plt.ylabel('Output Difference')
    plt.title(f'Output Difference After Ablating Feature {top_feature_idx} in Layer {top_layer}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_output_difference.png"))
    plt.close()

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to demonstrate feature importance analysis."""
    print("=" * 70)
    print("Circuit Analysis Toolkit - Feature Importance Analysis Example")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Create model and transcoders
    print("\n1. Setting up model and transcoders")
    model, transcoders, data = setup_model_and_transcoders()
    
    # 2. Analyze feature importance
    print("\n2. Analyzing feature importance")
    importance_scores = analyze_feature_importance(
        model, 
        transcoders, 
        data,
        top_k=ANALYSIS_PARAMS['top_k']
    )
    
    # 3. Run ablation experiment
    print("\n3. Running ablation experiment")
    ablation_results = run_ablation_experiment(
        model,
        transcoders,
        data,
        importance_scores,
        num_ablations=ANALYSIS_PARAMS['num_ablations']
    )
    
    # 4. Visualize feature activation patterns
    print("\n4. Visualizing feature activation patterns")
    visualize_feature_activation_patterns(model, transcoders, data)
    
    # 5. Compare model outputs with and without ablation
    print("\n5. Comparing model outputs with and without ablation")
    compare_model_outputs_with_ablation(model, transcoders, data, importance_scores)
    
    print("\nExample completed successfully!")
    print(f"Visualizations saved to {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main() 