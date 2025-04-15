import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import Transformer
from transcoder import TranscoderConfig, Transcoder, TranscoderReplacementContext
from circuit_analysis import CircuitTracer, FeatureVector, ComponentType, FeatureType

def create_toy_model():
    """Create a small transformer model for demonstration purposes."""
    model = Transformer(
        d_model=64,       # Small embedding dimension
        n_layers=2,       # Two transformer layers
        n_heads=4,        # Four attention heads
        d_mlp=128,        # MLP hidden dimension
        vocab_size=1000,  # Small vocabulary
        max_seq_len=128   # Maximum sequence length
    )
    return model

def create_synthetic_data(vocab_size=1000, seq_len=20, num_samples=100):
    """Create synthetic token data for training."""
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data

def train_toy_model(model, data, num_epochs=5):
    """Train the toy model to predict the next token."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for i in range(len(data)):
            # Get input and target
            x = data[i, :-1]  # All but the last token
            y = data[i, 1:]   # All but the first token
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x.unsqueeze(0))
            loss = criterion(logits[0], y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

def train_transcoders(model, data, layers=(0, 1)):
    """Train transcoders for MLP layers."""
    transcoders = {}
    
    for layer in layers:
        # Create transcoder config
        cfg = TranscoderConfig(
            d_model=model.d_model,
            d_hidden=model.d_mlp,
            hook_point=f'normalized_{layer}_ln2',
            hook_point_layer=layer,
            l1_coefficient=1e-3,
            is_post_ln=True
        )
        
        # Create and initialize transcoder
        transcoder = Transcoder(cfg)
        transcoders[layer] = transcoder
        
        # Create optimizer
        optimizer = optim.Adam(transcoder.parameters(), lr=0.001)
        
        # Train transcoder
        print(f"Training transcoder for layer {layer}...")
        for sample_idx in range(min(10, len(data))):  # Train on a few samples
            # Run model and get activations
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    data[sample_idx].unsqueeze(0), 
                    names_filter=[f'normalized_{layer}_ln2', f'pre_act_{layer}']
                )
                activations = cache[f'normalized_{layer}_ln2']
            
            # Train for a few epochs
            for epoch in range(20):
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, features = transcoder(activations)
                
                # Compute loss
                loss_dict = transcoder.loss_fn(activations, reconstructed, features)
                total_loss = loss_dict['total']
                
                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    print(f"  Sample {sample_idx+1}, Epoch {epoch+1}, Loss: {total_loss.item():.6f}")
    
    return transcoders

def find_important_directions(model, data, transcoders):
    """Find important directions in the model's activation space."""
    important_directions = []
    
    # Run the model on a data sample
    sample_idx = 0
    sample = data[sample_idx].unsqueeze(0)
    
    # Get activations
    logits, cache = model.run_with_cache(sample)
    
    # For each layer, find important features in the output
    for layer in range(model.n_layers):
        # Get MLP output
        ln2_out = cache[f'normalized_{layer}_ln2']
        mlp_out, _ = model.blocks[layer].mlp(ln2_out, layer_idx=layer)
        
        # Compute L2 norm of each output dimension
        norms = torch.norm(mlp_out, dim=2)
        
        # Find the token with the highest norm
        token_idx = torch.argmax(norms[0]).item()
        
        # Get the feature vector with the highest magnitude
        feature_vector = mlp_out[0, token_idx]
        
        # Get direction by normalizing
        direction = feature_vector / torch.norm(feature_vector)
        
        important_directions.append((direction, token_idx, layer))
    
    return important_directions

def visualize_circuit(model, direction, token_idx, layer, transcoders):
    """Visualize the circuit for a given direction."""
    # Get sample data
    sample = torch.randint(0, model.vocab_size, (1, token_idx + 1))
    
    # Create circuit tracer
    tracer = CircuitTracer(model, transcoders)
    
    # Run model
    tracer.run_model(sample)
    
    # Create feature vector for the direction
    feature_vector = FeatureVector(
        component_path=[],
        vector=direction,
        layer=layer,
        sublayer='resid_post',
        token=token_idx
    )
    
    # Trace circuit
    all_paths = tracer.trace_circuit(
        feature_vector, 
        num_iters=3, 
        num_branches=5
    )
    
    # Get circuit graph with default settings (no Jacobian, no pruning)
    edges, nodes = tracer.get_circuit_graph(
        all_paths, 
        add_error_nodes=True, 
        sum_over_tokens=False,
        use_jacobian=False,
        prune=False
    )
    
    # Visualize circuit
    fig = tracer.visualize_circuit(
        edges, nodes, 
        title=f"Circuit for Layer {layer}, Token {token_idx} (Basic)"
    )
    
    # Save plot
    plt.savefig(f"circuit_layer{layer}_token{token_idx}_basic.png")
    plt.close()
    
    return edges, nodes, fig

def visualize_circuit_advanced(model, direction, token_idx, layer, transcoders):
    """Visualize the circuit using advanced features (Jacobian weights and pruning)."""
    # Get sample data
    sample = torch.randint(0, model.vocab_size, (1, token_idx + 1))
    
    # Create circuit tracer
    tracer = CircuitTracer(model, transcoders)
    
    # Run model
    tracer.run_model(sample)
    
    # Create feature vector for the direction
    feature_vector = FeatureVector(
        component_path=[],
        vector=direction,
        layer=layer,
        sublayer='resid_post',
        token=token_idx
    )
    
    # Trace circuit with more branches (will be pruned later)
    all_paths = tracer.trace_circuit(
        feature_vector, 
        num_iters=3, 
        num_branches=10  # More branches for a richer initial graph
    )
    
    # Get circuit graph with Jacobian-based weights
    print("Generating circuit graph with Jacobian-based weights...")
    edges_jacobian, nodes_jacobian = tracer.get_circuit_graph(
        all_paths, 
        add_error_nodes=True, 
        sum_over_tokens=False,
        use_jacobian=True,  # Use Jacobian-based weights
        prune=False         # No pruning yet
    )
    
    # Visualize circuit with Jacobian weights
    fig_jacobian = tracer.visualize_circuit(
        edges_jacobian, nodes_jacobian, 
        title=f"Circuit for Layer {layer}, Token {token_idx} (Jacobian Weights)"
    )
    
    # Save plot
    plt.savefig(f"circuit_layer{layer}_token{token_idx}_jacobian.png")
    plt.close()
    
    # Get circuit graph with Jacobian-based weights and pruning
    print("Generating pruned circuit graph...")
    edges_pruned, nodes_pruned, metrics = tracer.get_circuit_graph(
        all_paths, 
        add_error_nodes=True, 
        sum_over_tokens=False,
        use_jacobian=True,   # Use Jacobian-based weights
        prune=True,          # Apply pruning
        threshold=0.01,      # Minimum absolute weight to keep
        max_edges=15         # Maximum number of edges to keep
    )
    
    # Print pruning metrics
    print(f"Pruning metrics:")
    print(f"  Original edges: {metrics['original_edges']}")
    print(f"  Kept edges: {metrics['kept_edges']}")
    print(f"  Pruned weight ratio: {metrics['pruned_weight_ratio']:.2%}")
    
    # Visualize pruned circuit
    fig_pruned = tracer.visualize_circuit(
        edges_pruned, nodes_pruned, 
        title=f"Circuit for Layer {layer}, Token {token_idx} (Pruned)",
        width=1200,  # Larger size for better visibility
        height=800
    )
    
    # Save plot
    plt.savefig(f"circuit_layer{layer}_token{token_idx}_pruned.png")
    plt.close()
    
    return {
        'jacobian': (edges_jacobian, nodes_jacobian, fig_jacobian),
        'pruned': (edges_pruned, nodes_pruned, fig_pruned, metrics)
    }

def visualize_replacement(model, direction, token_idx, layer, transcoders):
    """Visualize the effect of replacing an MLP with a transcoder."""
    # Get sample data
    sample = torch.randint(0, model.vocab_size, (1, token_idx + 1))
    
    # Run model with original MLPs
    orig_logits, _ = model.run_with_cache(sample)
    
    # Run model with transcoders replacing MLPs
    with TranscoderReplacementContext(model, list(transcoders.values())):
        tc_logits, _ = model.run_with_cache(sample)
    
    # Compute difference
    diff = tc_logits - orig_logits
    
    # Calculate error
    error = torch.norm(diff).item()
    
    print(f"Replacement error: {error:.4f}")
    
    # Plot difference in logits
    plt.figure(figsize=(10, 5))
    plt.bar(range(min(20, model.vocab_size)), diff[0, -1, :20].detach().cpu().numpy())
    plt.title(f"Difference in logits after transcoder replacement (top 20 tokens)")
    plt.xlabel("Token index")
    plt.ylabel("Logit difference")
    plt.savefig(f"replacement_diff_layer{layer}.png")
    plt.close()
    
    return error

def demonstrate_downstream_propagation_with_jacobian(model, data, transcoders):
    """
    Demonstrate how downstream propagation in transcoders affects circuit analysis
    using Jacobian-based connection weights.
    """
    print("\nDemonstrating downstream propagation with Jacobian-based analysis...")
    
    # Get a sample for analysis
    sample_idx = 0
    sample = data[sample_idx].unsqueeze(0)
    
    # Choose a layer and token to analyze
    layer = model.n_layers - 1  # Last layer
    token_idx = -1              # Last token
    
    # Create circuit tracer
    tracer = CircuitTracer(model, transcoders)
    
    # Run model
    logits = tracer.run_model(sample)
    
    # Get the output vector to analyze
    output_vector = logits[0, token_idx]
    
    # Create feature vector for the direction
    feature_vector = FeatureVector(
        component_path=[],
        vector=output_vector,
        layer=layer,
        sublayer='resid_post',
        token=token_idx
    )
    
    # Configure different propagation settings
    propagation_settings = [
        {"name": "no_propagation", "propagate_downstream": False, "excluded_layers": []},
        {"name": "full_propagation", "propagate_downstream": True, "excluded_layers": []},
        {"name": "selective_propagation", "propagate_downstream": True, "excluded_layers": [1]}  # Exclude layer 1
    ]
    
    results = {}
    
    # Test each propagation setting
    for setting in propagation_settings:
        print(f"\nTesting {setting['name']} setting...")
        
        # Apply the propagation setting using TranscoderReplacementContext
        from transcoder import TranscoderReplacementContext
        
        # Convert transcoders dict to single transcoder for simplicity
        # In a real scenario, you might use multiple transcoders
        if isinstance(transcoders, dict) and len(transcoders) > 0:
            main_transcoder = list(transcoders.values())[0]
        else:
            main_transcoder = transcoders
        
        with TranscoderReplacementContext(
            model, 
            main_transcoder, 
            hook_point="mlp",
            hook_point_layer=0,  # Apply to first layer
            propagate_downstream=setting["propagate_downstream"],
            excluded_layers=setting["excluded_layers"]
        ):
            # Run model with this setting
            setting_logits = model(sample)
            
            # Trace circuit with this setting
            setting_paths = tracer.trace_circuit(
                feature_vector, 
                num_iters=3, 
                num_branches=8
            )
            
            # Get circuit graph with Jacobian weights and pruning
            setting_edges, setting_nodes, metrics = tracer.get_circuit_graph(
                setting_paths, 
                add_error_nodes=True,
                use_jacobian=True,
                prune=True,
                threshold=0.01,
                max_edges=20
            )
            
            # Visualize circuit
            setting_fig = tracer.visualize_circuit(
                setting_edges, setting_nodes, 
                title=f"Circuit with {setting['name'].replace('_', ' ').title()}",
                width=1200,
                height=800
            )
            
            # Save plot
            plt.savefig(f"circuit_{setting['name']}.png")
            plt.close()
            
            # Calculate output difference from original
            if 'original_output' not in locals():
                original_output = logits[0, token_idx]
                
            output_diff = torch.norm(setting_logits[0, token_idx] - original_output).item()
            
            # Store results
            results[setting['name']] = {
                'edges': setting_edges,
                'nodes': setting_nodes,
                'fig': setting_fig,
                'metrics': metrics,
                'output_diff': output_diff
            }
            
            # Print metrics
            print(f"  Circuit complexity: {len(setting_edges)} edges")
            print(f"  Pruned weight ratio: {metrics['pruned_weight_ratio']:.2%}")
            print(f"  Output difference: {output_diff:.6f}")
    
    # Compare the circuits
    print("\nPropagation effect comparison:")
    
    no_prop = results['no_propagation']
    full_prop = results['full_propagation']
    selective_prop = results['selective_propagation']
    
    # Compare number of edges
    print(f"  No propagation edges: {len(no_prop['edges'])}")
    print(f"  Full propagation edges: {len(full_prop['edges'])}")
    print(f"  Selective propagation edges: {len(selective_prop['edges'])}")
    
    # Compare output differences
    print(f"  Full propagation output diff: {full_prop['output_diff']:.6f}")
    print(f"  Selective propagation output diff: {selective_prop['output_diff']:.6f}")
    
    # Find edges that exist in full propagation but not in no propagation
    prop_only_edges = set(full_prop['edges'].keys()) - set(no_prop['edges'].keys())
    print(f"  Edges unique to full propagation: {len(prop_only_edges)}")
    
    return results

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Creating toy model...")
    model = create_toy_model()
    
    print("Creating synthetic data...")
    data = create_synthetic_data()
    
    print("Training toy model...")
    train_toy_model(model, data, num_epochs=2)
    
    print("Training transcoders...")
    transcoders = train_transcoders(model, data)
    
    print("Finding important directions...")
    important_directions = find_important_directions(model, data, transcoders)
    
    print("Visualizing basic circuits...")
    for i, (direction, token_idx, layer) in enumerate(important_directions):
        print(f"Visualizing circuit {i+1}/{len(important_directions)}")
        visualize_circuit(model, direction, token_idx, layer, transcoders)
    
    print("Visualizing advanced circuits...")
    for i, (direction, token_idx, layer) in enumerate(important_directions):
        print(f"Visualizing advanced circuit {i+1}/{len(important_directions)}")
        results = visualize_circuit_advanced(model, direction, token_idx, layer, transcoders)
        
        # Print some comparison metrics
        jacobian_edges = results['jacobian'][0]
        pruned_edges = results['pruned'][0]
        metrics = results['pruned'][3]
        
        print(f"Circuit comparison:")
        print(f"  Basic edges count: {len(jacobian_edges)}")
        print(f"  Pruned edges count: {len(pruned_edges)}")
        print(f"  Pruning efficiency: {metrics['pruned_weight_ratio']:.2%} weight in {len(pruned_edges)}/{metrics['original_edges']} edges")
    
    print("Visualizing replacement effects...")
    for i, (direction, token_idx, layer) in enumerate(important_directions):
        print(f"Visualizing replacement {i+1}/{len(important_directions)}")
        visualize_replacement(model, direction, token_idx, layer, transcoders)
    
    # Demonstrate downstream propagation with Jacobian analysis
    demonstrate_downstream_propagation_with_jacobian(model, data, transcoders)
    
    print("Done!")

if __name__ == "__main__":
    main() 