"""
Quickstart script for Circuit Tracing tools.

This script demonstrates the basic usage of the circuit tracing tools with a simple example.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from model import Transformer
from transcoder import TranscoderConfig, Transcoder, TranscoderReplacementContext
from circuit_analysis import CircuitTracer, FeatureVector, ComponentType, FeatureType

def main():
    """Run a basic circuit tracing example."""
    print("Circuit Tracing Quickstart")
    print("=========================")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Create a simple transformer model
    print("\n1. Creating a simple transformer model...")
    model = Transformer(
        d_model=64,       # Small embedding dimension
        n_layers=2,       # Two transformer layers
        n_heads=4,        # Four attention heads
        d_mlp=128,        # MLP hidden dimension
        vocab_size=1000,  # Small vocabulary
        max_seq_len=128   # Maximum sequence length
    )
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. Create transcoders for each layer
    print("\n2. Creating transcoders...")
    transcoders = {}
    for layer in range(model.n_layers):
        cfg = TranscoderConfig(
            d_model=model.d_model,
            d_hidden=model.d_mlp,
            hook_point=f'normalized_{layer}_ln2',
            hook_point_layer=layer,
            l1_coefficient=1e-3,
            is_post_ln=True
        )
        transcoders[layer] = Transcoder(cfg)
        print(f"   Created transcoder for layer {layer}")
    
    # 3. Create a sample input
    print("\n3. Creating a sample input...")
    input_ids = torch.randint(0, 1000, (1, 10))
    print(f"   Input shape: {input_ids.shape}")
    
    # 4. Create a circuit tracer
    print("\n4. Creating a circuit tracer...")
    tracer = CircuitTracer(model, transcoders)
    
    # 5. Run the model
    print("\n5. Running the model...")
    logits = tracer.run_model(input_ids)
    print(f"   Model output shape: {logits.shape}")
    
    # 6. Select a feature to analyze
    print("\n6. Selecting a feature to analyze...")
    token_idx = 5  # 6th token
    layer = 1      # 2nd layer
    
    # Get the activation at the specified position
    activation = tracer.cache[f'resid_post_{layer}'][0, token_idx]
    
    # Create a feature vector for this activation
    feature_vector = FeatureVector(
        component_path=[],
        vector=activation,
        layer=layer,
        sublayer='resid_post',
        token=token_idx
    )
    print(f"   Selected activation at layer {layer}, token {token_idx}")
    
    # 7. Trace the circuit
    print("\n7. Tracing the circuit...")
    paths = tracer.trace_circuit(
        feature_vector,
        num_iters=2,
        num_branches=3
    )
    print(f"   Found {len(paths)} paths")
    
    # Print the top paths
    print("\n   Top 5 paths:")
    for i, path in enumerate(paths[:5]):
        print(f"   Path {i+1}: {path}")
    
    # 8. Visualize the circuit
    print("\n8. Visualizing the circuit...")
    edges, nodes = tracer.get_circuit_graph(paths, add_error_nodes=True)
    fig = tracer.visualize_circuit(
        edges, nodes,
        title=f"Circuit for Layer {layer}, Token {token_idx}",
        width=1000,
        height=800
    )
    
    # Save the figure
    output_file = "quickstart_circuit.png"
    plt.savefig(output_file)
    plt.close()
    print(f"   Circuit visualization saved to {output_file}")
    
    # 9. Demonstrate replacement
    print("\n9. Demonstrating replacement with transcoders...")
    
    # Run model with original MLPs
    orig_logits, _ = model.run_with_cache(input_ids)
    
    # Run model with MLPs replaced by transcoders
    with TranscoderReplacementContext(model, list(transcoders.values())):
        tc_logits, _ = model.run_with_cache(input_ids)
    
    # Compute difference
    diff = tc_logits - orig_logits
    error = torch.norm(diff).item()
    
    print(f"   Replacement error: {error:.4f}")
    
    # Plot difference in logits
    plt.figure(figsize=(10, 5))
    plt.bar(range(min(20, model.vocab_size)), diff[0, -1, :20].detach().cpu().numpy())
    plt.title("Difference in logits after transcoder replacement (top 20 tokens)")
    plt.xlabel("Token index")
    plt.ylabel("Logit difference")
    
    # Save the figure
    output_file = "quickstart_replacement.png"
    plt.savefig(output_file)
    plt.close()
    print(f"   Replacement visualization saved to {output_file}")
    
    print("\nQuickstart completed successfully!")

if __name__ == "__main__":
    main() 