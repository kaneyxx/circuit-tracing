#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pre-trained Model Integration Example

This script demonstrates how to integrate pre-trained models from Hugging Face
with the circuit analysis toolkit. It covers:
1. Loading a pre-trained model
2. Adapting it to work with the toolkit
3. Training transcoders on the model
4. Analyzing specific behaviors

Usage:
    python pretrained_model_integration.py
"""

import os
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

# Add parent directory to path so we can import the modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from transcoder import TranscoderConfig, Transcoder, TranscoderReplacementContext
from circuit_analysis import CircuitTracer, FeatureVector

# ============================================================================
# Configuration (modify these parameters according to your needs)
# ============================================================================

# Pre-trained model parameters
PRETRAINED_PARAMS = {
    'model_name': 'bert-base-uncased',  # Options: 'bert-base-uncased', 'gpt2', etc.
    'max_length': 32                    # Maximum sequence length for tokenization
}

# Transcoder parameters
TRANSCODER_PARAMS = {
    'l1_coefficient': 1e-3,    # L1 sparsity coefficient
    'epochs': 3,               # Epochs for training transcoders
    'learning_rate': 1e-3,     # Learning rate for transcoder training
    'batch_size': 4            # Batch size for transcoder training
}

# Example text for analysis
ANALYSIS_TEXTS = [
    "The cat sat on the mat.",
    "The dog played in the yard."
]

# Output directory for saving results
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Hugging Face Model Adapter
# ============================================================================

class HuggingFaceModelAdapter:
    """
    Adapter class to make Hugging Face models compatible with our toolkit.
    
    This class wraps a Hugging Face model and provides the interface expected
    by our toolkit, including activation caching and layer access.
    """
    
    def __init__(self, model_name):
        """
        Initialize the adapter with a Hugging Face model.
        
        Args:
            model_name: Name of the pre-trained model to load from Hugging Face
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            print("Error: The transformers library is not installed.")
            print("Please install it with: pip install transformers")
            sys.exit(1)
            
        print(f"Loading pre-trained model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Extract model configuration
        self.config = self.model.config
        
        # Get architecture-specific parameters
        if 'bert' in model_name.lower():
            self._setup_bert_params()
        elif 'gpt' in model_name.lower():
            self._setup_gpt_params()
        else:
            print(f"Warning: Model {model_name} not specifically supported.")
            print("Attempting to use generic parameters.")
            self._setup_generic_params()
            
        print(f"Model loaded with {self.n_layers} layers, d_model={self.d_model}, d_mlp={self.d_mlp}")
        
        # Create placeholder for blocks to make compatible with toolkit
        self.blocks = self._create_blocks()
    
    def _setup_bert_params(self):
        """Setup parameters specific to BERT models."""
        self.n_layers = self.config.num_hidden_layers
        self.d_model = self.config.hidden_size
        self.d_mlp = self.config.intermediate_size
        self.vocab_size = self.config.vocab_size
        
        # Define layer name patterns for hooks
        self.layer_module_pattern = "encoder.layer"
        self.ln_pattern = "attention.output.LayerNorm"
        self.mlp_pattern = "intermediate"
    
    def _setup_gpt_params(self):
        """Setup parameters specific to GPT models."""
        self.n_layers = self.config.n_layer
        self.d_model = self.config.n_embd
        self.d_mlp = self.config.n_embd * 4  # GPT typically uses 4x embedding dim for MLP
        self.vocab_size = self.config.vocab_size
        
        # Define layer name patterns for hooks
        self.layer_module_pattern = "h"
        self.ln_pattern = "ln_2"
        self.mlp_pattern = "mlp"
    
    def _setup_generic_params(self):
        """Setup generic parameters as a fallback."""
        # Try to extract common parameters from config
        self.n_layers = getattr(self.config, "num_hidden_layers", 
                              getattr(self.config, "n_layer", 12))
        self.d_model = getattr(self.config, "hidden_size", 
                             getattr(self.config, "n_embd", 768))
        self.d_mlp = getattr(self.config, "intermediate_size", 
                           getattr(self.config, "n_inner", self.d_model * 4))
        self.vocab_size = getattr(self.config, "vocab_size", 30000)
        
        # Use generic pattern names
        self.layer_module_pattern = None
        self.ln_pattern = None
        self.mlp_pattern = None
    
    def _create_blocks(self):
        """
        Create block representations compatible with the toolkit.
        
        Returns:
            List of block objects with ln2 and mlp attributes
        """
        blocks = []
        for i in range(self.n_layers):
            # For BERT-like models
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                ln2 = self.model.encoder.layer[i].attention.output.LayerNorm
                mlp = self.model.encoder.layer[i].intermediate
            # For GPT-like models
            elif hasattr(self.model, 'h'):
                ln2 = self.model.h[i].ln_2
                mlp = self.model.h[i].mlp
            # Generic fallback
            else:
                # Create dummy objects
                ln2 = type('DummyLN', (), {'forward': lambda x: x})
                mlp = type('DummyMLP', (), {'forward': lambda x: x})
                print(f"Warning: Could not find layer components for layer {i}")
            
            block = type('Block', (), {'ln2': ln2, 'mlp': mlp})
            blocks.append(block)
        
        return blocks
    
    def run_with_cache(self, input_ids, names_filter=None):
        """
        Run the model and cache intermediate activations.
        
        Args:
            input_ids: Input token IDs
            names_filter: Optional list of activation names to cache
            
        Returns:
            Tuple of (outputs, cache)
        """
        # Initialize cache
        cache = {}
        
        # Create hooks for each layer
        hooks = []
        
        for i in range(self.n_layers):
            # Hook names to register based on names_filter
            hook_names = []
            
            if names_filter is None or f"normalized_{i}_ln2" in names_filter:
                hook_names.append((f"normalized_{i}_ln2", self._get_ln_module(i)))
                
            if names_filter is None or f"resid_post_{i}" in names_filter:
                hook_names.append((f"resid_post_{i}", self._get_output_module(i)))
            
            # Register hooks
            for name, module in hook_names:
                if module is not None:
                    hook = module.register_forward_hook(
                        self._create_hook_fn(name, cache)
                    )
                    hooks.append(hook)
        
        # Run the model
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return outputs, cache
    
    def _get_ln_module(self, layer_idx):
        """Get the LayerNorm module for a specific layer."""
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return self.model.encoder.layer[layer_idx].attention.output.LayerNorm
        elif hasattr(self.model, 'h'):
            return self.model.h[layer_idx].ln_2
        return None
    
    def _get_output_module(self, layer_idx):
        """Get the output module for a specific layer."""
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return self.model.encoder.layer[layer_idx].output
        elif hasattr(self.model, 'h'):
            return self.model.h[layer_idx].mlp
        return None
    
    def _create_hook_fn(self, name, cache):
        """Create a hook function that stores activations in the cache."""
        def hook_fn(module, input, output):
            cache[name] = output
            return output
        return hook_fn
    
    def __call__(self, input_ids):
        """
        Run the model on input IDs.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Model hidden states
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            
        # Return last hidden states to match toolkit expectations
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        else:
            return outputs[0]  # Assume first output is hidden states
    
    def tokenize(self, texts, max_length=None):
        """
        Tokenize texts using the model's tokenizer.
        
        Args:
            texts: List of text strings to tokenize
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of tokenizer outputs
        """
        max_length = max_length or PRETRAINED_PARAMS['max_length']
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

# ============================================================================
# Utility Functions
# ============================================================================

def train_transcoders_for_pretrained(adapted_model, texts, params):
    """
    Create and train transcoders for a pre-trained model.
    
    Args:
        adapted_model: Adapted Hugging Face model
        texts: List of example texts for training
        params: Dictionary of transcoder training parameters
        
    Returns:
        Dictionary mapping layer indices to trained transcoders
    """
    print(f"Creating and training transcoders for {adapted_model.n_layers} layers")
    
    # Tokenize the texts
    tokens = adapted_model.tokenize(texts)
    input_ids = tokens.input_ids
    
    # Create transcoders for each layer
    transcoders = {}
    
    for layer in range(adapted_model.n_layers):
        print(f"  Training transcoder for layer {layer}")
        
        # Create transcoder configuration
        cfg = TranscoderConfig(
            d_model=adapted_model.d_model,
            d_hidden=adapted_model.d_mlp,
            hook_point=f'normalized_{layer}_ln2',
            hook_point_layer=layer,
            l1_coefficient=params['l1_coefficient'],
            is_post_ln=True,
            # Specify that this transcoder affects all subsequent layers
            downstream_layers=list(range(layer+1, adapted_model.n_layers))
        )
        
        # Create transcoder
        transcoder = Transcoder(cfg)
        
        # Create optimizer
        optimizer = optim.Adam(transcoder.parameters(), lr=params['learning_rate'])
        
        # Train the transcoder with downstream propagation
        from transcoder import train_transcoder
        
        # Get activations for this layer
        with torch.no_grad():
            _, cache = adapted_model.run_with_cache(
                input_ids, 
                names_filter=[f'normalized_{layer}_ln2']
            )
            activations = cache[f'normalized_{layer}_ln2']
        
        # Set up simplified training for pre-trained model
        # Note: We're skipping full downstream propagation for simplicity
        # In a real use case, you'd want to implement proper multi-layer training
        for epoch in range(params['epochs']):
            optimizer.zero_grad()
            reconstructed, features = transcoder(activations)
            loss_dict = transcoder.loss_fn(activations, reconstructed, features)
            total_loss = loss_dict['total']
            total_loss.backward()
            optimizer.step()
            
            if epoch % 1 == 0:
                print(f"    Epoch {epoch+1}/{params['epochs']}, Loss: {total_loss.item():.6f}")
        
        transcoders[layer] = transcoder
        
    return transcoders

def analyze_pretrained_model(adapted_model, transcoders, texts):
    """
    Analyze the pre-trained model using transcoders.
    
    Args:
        adapted_model: Adapted Hugging Face model
        transcoders: Dictionary of trained transcoders
        texts: List of texts for analysis
        
    Returns:
        CircuitTracer object with results
    """
    print("Analyzing pre-trained model with circuit tracer")
    
    # Tokenize the texts
    tokens = adapted_model.tokenize(texts)
    input_ids = tokens.input_ids
    
    # Create circuit tracer
    tracer = CircuitTracer(adapted_model, transcoders)
    
    # Run model on input data
    tracer.run_model(input_ids)
    
    # Select a feature to analyze (last token of the first sequence)
    token_idx = -1
    layer = adapted_model.n_layers - 1
    
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
    plt.savefig(os.path.join(OUTPUT_DIR, "pretrained_circuit_visualization.png"))
    plt.close()
    
    return tracer, paths, edges, nodes

def compare_with_transcoders(adapted_model, transcoders, texts):
    """
    Compare model outputs with and without transcoders.
    
    Args:
        adapted_model: Adapted Hugging Face model
        transcoders: Dictionary of trained transcoders
        texts: List of texts for analysis
    """
    print("Comparing model with and without transcoders")
    
    # Tokenize the texts
    tokens = adapted_model.tokenize(texts)
    input_ids = tokens.input_ids
    
    # Get original model output
    with torch.no_grad():
        original_output = adapted_model(input_ids)
    
    # Get output with transcoders
    with torch.no_grad(), TranscoderReplacementContext(adapted_model, transcoders):
        transcoder_output = adapted_model(input_ids)
    
    # Calculate difference
    diff = torch.norm(transcoder_output - original_output).item()
    
    print(f"Transcoder replacement difference (L2 norm): {diff:.6f}")
    
    # Detailed comparison for each text
    for i, text in enumerate(texts):
        print(f"\nText: {text}")
        print(f"  Original output norm: {torch.norm(original_output[i]).item():.4f}")
        print(f"  Transcoder output norm: {torch.norm(transcoder_output[i]).item():.4f}")
    
    # Plot a comparison of the outputs for the first sequence
    plt.figure(figsize=(12, 6))
    
    # Get the last token representation for comparison
    orig_last = original_output[0, -1].detach().cpu().numpy()
    trans_last = transcoder_output[0, -1].detach().cpu().numpy()
    
    # Plot the first 20 dimensions
    dims = min(20, orig_last.shape[0])
    x = np.arange(dims)
    width = 0.35
    
    plt.bar(x - width/2, orig_last[:dims], width, label='Original')
    plt.bar(x + width/2, trans_last[:dims], width, label='Transcoder')
    
    plt.xlabel('Dimension')
    plt.ylabel('Activation')
    plt.title('Comparison of Original vs. Transcoder Outputs (First 20 Dimensions)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "pretrained_comparison.png"))
    plt.close()

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to demonstrate pre-trained model integration."""
    print("=" * 70)
    print("Circuit Analysis Toolkit - Pre-trained Model Integration Example")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load and adapt a pre-trained model
    print("\n1. Loading and adapting pre-trained model")
    adapted_model = HuggingFaceModelAdapter(PRETRAINED_PARAMS['model_name'])
    
    # 2. Train transcoders for the pre-trained model
    print("\n2. Training transcoders for the pre-trained model")
    transcoders = train_transcoders_for_pretrained(
        adapted_model,
        ANALYSIS_TEXTS,
        TRANSCODER_PARAMS
    )
    
    # 3. Analyze the model
    print("\n3. Analyzing the pre-trained model")
    tracer, paths, edges, nodes = analyze_pretrained_model(
        adapted_model,
        transcoders,
        ANALYSIS_TEXTS
    )
    
    # 4. Compare model with and without transcoders
    print("\n4. Comparing model with and without transcoders")
    compare_with_transcoders(adapted_model, transcoders, ANALYSIS_TEXTS)
    
    print("\nExample completed successfully!")
    print(f"Visualizations saved to {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main() 