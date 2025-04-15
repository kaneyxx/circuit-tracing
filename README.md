# Transformer Circuit Analysis Toolkit

This project provides a comprehensive toolkit for analyzing and visualizing circuits in transformer models. It allows you to decompose complex transformer behaviors into interpretable features and trace computational paths through the model.

## 🚀 Key Features

- **Multi-layer transcoders** that propagate effects across model layers
- **Selective layer control** to include or exclude specific layers from analysis
- **Feature weight management** to amplify or ablate specific features
- **Circuit visualization** to understand information flow through the model
- **Localization experiments** to identify critical components of a circuit

## 📋 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/interpretation.git
cd interpretation

# Install dependencies
pip install -r requirements.txt
```

## 🏁 Quick Start

Run the quickstart script to see the basic functionality in action:

```bash
python quickstart.py
```

This will demonstrate:
1. Creating a small transformer model
2. Setting up transcoders for feature decomposition
3. Tracing circuits through the model
4. Visualizing the results

## 🧩 Core Components

- `model.py`: Configurable transformer model implementation
- `transcoder.py`: Feature decomposition with multi-layer propagation
- `circuit_analysis.py`: Circuit tracing and visualization tools
- `circuit_demo.py`: Complete demonstration of the toolkit

## 🔄 Advanced Circuit Analysis Features

### Jacobian-based Connection Weights

The toolkit now includes Jacobian-based connection weight calculation, which provides a more accurate measure of how information flows between components in the model:

```python
from circuit_analysis import CircuitTracer

# Create a circuit tracer
tracer = CircuitTracer(model, transcoders)

# Run the model to populate the activation cache
tokens = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
logits = tracer.run_model(tokens.input_ids)

# Get the feature vector to analyze
output_vector = logits[0, -1]  # Last token logits
token_idx = -1
layer = model.n_layers - 1
sublayer = "resid_post"

# Trace the circuit with Jacobian-based weights
edges, nodes, fig = tracer.analyze_feature(
    output_vector,
    token_idx,
    layer,
    sublayer,
    use_jacobian=True,  # Enable Jacobian-based weighting
    num_iters=3,
    num_branches=5
)
```

Jacobian-based weights capture how changes in earlier components affect later components, providing a more accurate representation of causal relationships in the circuit. This is particularly important for:

- Understanding attention mechanisms where direct dot products may not capture the full influence
- Analyzing how MLP layers transform features in non-linear ways
- Determining the true strength of connections across multiple layers

### Circuit Graph Pruning

To manage complexity in large circuit graphs, the toolkit supports automatic pruning based on connection strength:

```python
# Trace and visualize a circuit with pruning
edges, nodes, fig, pruning_metrics = tracer.analyze_feature(
    output_vector,
    token_idx,
    layer,
    sublayer,
    prune=True,              # Enable pruning
    threshold=0.01,          # Minimum absolute weight to keep
    max_edges=100,           # Maximum number of edges to display
    num_iters=3,
    num_branches=10          # Explore more branches, pruning will simplify
)

# Print pruning metrics
print(f"Original edges: {pruning_metrics['original_edges']}")
print(f"Kept edges: {pruning_metrics['kept_edges']}")
print(f"Pruned weight ratio: {pruning_metrics['pruned_weight_ratio']:.2%}")
```

Benefits of pruning:

1. **Improved visualization clarity**: Focus on the most important connections
2. **Computational efficiency**: Reduce memory and processing requirements
3. **Noise reduction**: Filter out weak connections that may be statistical noise
4. **Interpretability**: Simpler graphs are easier to understand and explain

The pruning process automatically adds a "prune_error" node to account for the total weight of pruned connections, ensuring that the analysis maintains accuracy while simplifying the presentation.

## 🔍 Complete Usage Guide

### 1. Training Your Own Models and Transcoders

**Scenario**: You want to understand how a language model makes specific predictions by decomposing its internal representations into interpretable features.

In this scenario, you'll first build and train a transformer model, then train transcoders to decompose the model's internal representations:

```python
import torch
import torch.optim as optim
from model import Transformer
from transcoder import TranscoderConfig, Transcoder, train_transcoder

# 1. Create a transformer model
model = Transformer(d_model=768, n_layers=12, n_heads=12, d_mlp=3072, vocab_size=50000)

# 2. Prepare training data (e.g., tokenized text sequences)
train_data = torch.randint(0, 50000, (1000, 128))  # 1000 sequences of length 128

# 3. Train the transformer model (optional if you're using a pre-trained model)
def train_model(model, data, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for i in range(0, len(data), 32):
            batch = data[i:i+32]
            inputs = batch[:, :-1]  # All but the last token
            targets = batch[:, 1:]   # All but the first token
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
    return model

# 4. Train transcoders for each layer
transcoders = {}
for layer in range(model.n_layers):
    # Create transcoder configuration
    cfg = TranscoderConfig(
        d_model=model.d_model,
        d_hidden=model.d_mlp,
        hook_point=f'normalized_{layer}_ln2',
        hook_point_layer=layer,
        downstream_layers=list(range(layer+1, model.n_layers))  # Affect all subsequent layers
    )
    
    # Initialize transcoder
    transcoder = Transcoder(cfg)
    
    # Train the transcoder
    optimizer = optim.Adam(transcoder.parameters(), lr=1e-3)
    losses = train_transcoder(
        model, 
        transcoder, 
        train_data, 
        optimizer,
        batch_size=32,
        num_epochs=10,
        downstream_layers=list(range(layer+1, model.n_layers))
    )
    
    transcoders[layer] = transcoder
    print(f"Layer {layer} transcoder trained, final loss: {losses[-1]:.6f}")
```

**Benefits**: Training transcoders on your own model allows you to:
- Understand which features the model learns to detect
- Identify how information flows between model layers
- Decompose complex behaviors into simpler, interpretable features
- Find potential biases or shortcomings in the model's learned representations

### 2. Integrating with Pre-trained Models (e.g., from Hugging Face)

**Scenario**: You want to analyze how a pre-trained model like BERT or GPT processes information and makes predictions.

In this scenario, you'll load a pre-trained model from Hugging Face, adapt it to work with our toolkit, and train transcoders on it:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from transcoder import TranscoderConfig, Transcoder, create_transcoder_pipeline, TranscoderReplacementContext

# 1. Load a pre-trained model
model_name = "bert-base-uncased"  # or any other pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
pretrained_model = AutoModel.from_pretrained(model_name)

# 2. Create an adapter to make the Hugging Face model compatible with our toolkit
class HuggingFaceAdapter:
    def __init__(self, hf_model):
        self.model = hf_model
        self.n_layers = hf_model.config.num_hidden_layers
        self.d_model = hf_model.config.hidden_size
        self.d_mlp = hf_model.config.intermediate_size
        self.blocks = []
        
        # Get references to each layer
        for i in range(self.n_layers):
            # BERT-specific structure - adjust for other model types
            self.blocks.append(type('Block', (), {
                'ln2': self.model.encoder.layer[i].attention.output.LayerNorm,
                'mlp': self.model.encoder.layer[i].intermediate
            }))
    
    def run_with_cache(self, input_ids, names_filter=None):
        """Run the model and cache intermediate activations"""
        # Initialize cache
        cache = {}
        
        # Create hooks for each layer
        def create_hook(layer_idx, name):
            def hook_fn(module, input, output):
                cache[f"{name}_{layer_idx}"] = output
                return output
            return hook_fn
        
        # Register hooks
        handles = []
        for i in range(self.n_layers):
            if names_filter is None or f"normalized_{i}_ln2" in names_filter:
                h = self.model.encoder.layer[i].attention.output.LayerNorm.register_forward_hook(
                    create_hook(i, "normalized_ln2")
                )
                handles.append(h)
                
            if names_filter is None or f"resid_post_{i}" in names_filter:
                h = self.model.encoder.layer[i].output.register_forward_hook(
                    create_hook(i, "resid_post")
                )
                handles.append(h)
        
        # Run the model
        outputs = self.model(input_ids)
        
        # Remove hooks
        for h in handles:
            h.remove()
            
        return outputs, cache
    
    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        return outputs.last_hidden_state

# 3. Adapt the pre-trained model
adapted_model = HuggingFaceAdapter(pretrained_model)

# 4. Create transcoders for the adapted model
transcoders = create_transcoder_pipeline(adapted_model)

# 5. Train transcoders on the pre-trained model
# This step uses the train_transcoder function demonstrated earlier
# ...

# 6. Analyze the model using transcoders
text = "This is an example sentence for analysis."
input_ids = tokenizer(text, return_tensors="pt").input_ids

# Replace MLP layers with transcoders for analysis
with TranscoderReplacementContext(adapted_model, transcoders):
    outputs, cache = adapted_model.run_with_cache(input_ids)
    # Perform analysis...
```

**Benefits**: Integrating with pre-trained models allows you to:
- Leverage state-of-the-art models without training your own
- Investigate how these models represent knowledge
- Compare different model architectures and their internal mechanisms
- Analyze specific behaviors or failures in these widely-used models

### 3. Analyzing Feature Importance and Connections

**Scenario**: You want to understand how a model processes specific types of inputs and evaluate the importance of different features.

In this scenario, you'll use transcoders to analyze how a model represents and processes information:

```python
import torch
import matplotlib.pyplot as plt
from circuit_analysis import CircuitTracer, FeatureVector, ComponentType
from transcoder import ablate_features, visualize_feature_importance

# Assume model and transcoders are already set up

# 1. Prepare a representative dataset for analysis
test_examples = [
    "The movie was excellent, I loved it.",
    "The movie was terrible, I hated it.",
    "This restaurant has the best food.",
    "This restaurant has the worst food."
]
encoded_inputs = tokenizer(test_examples, padding=True, return_tensors="pt")

# 2. Create a circuit tracer
tracer = CircuitTracer(adapted_model, transcoders)

# 3. Run the model and analyze
tracer.run_model(encoded_inputs.input_ids)

# 4. Identify a direction of interest (e.g., positive vs. negative sentiment)
positive_output = tracer.cache["resid_post_11"][0, -1]  # Last token of first positive example
negative_output = tracer.cache["resid_post_11"][1, -1]  # Last token of first negative example
sentiment_direction = positive_output - negative_output
sentiment_direction = sentiment_direction / torch.norm(sentiment_direction)

# 5. Create a feature vector and trace the circuit
feature_vector = FeatureVector(
    component_path=[],
    vector=sentiment_direction,
    layer=11,  # Final layer
    sublayer='resid_post',
    token=-1  # Last token
)

# Trace the circuit
paths = tracer.trace_circuit(feature_vector, num_iters=3, num_branches=5)

# 6. Evaluate feature importance
importance_scores = visualize_feature_importance(
    adapted_model, 
    transcoders, 
    encoded_inputs.input_ids,
    top_k=10
)

print("Most important features:")
for (layer, feature_idx), score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Layer {layer}, Feature {feature_idx}: {score:.4f}")

# 7. Conduct feature ablation experiments
important_features = [(layer, feature) for (layer, feature), _ in 
                      sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]]

ablation_results = ablate_features(
    adapted_model,
    transcoders,
    encoded_inputs.input_ids,
    important_features
)

print(f"Ablation impact (L2 norm): {ablation_results['l2_norm']:.4f}")

# 8. Visualize the circuit
edges, nodes = tracer.get_circuit_graph(paths, add_error_nodes=True)
fig = tracer.visualize_circuit(
    edges, nodes, 
    title="Sentiment Analysis Circuit",
    width=1200, 
    height=800
)
plt.savefig("sentiment_circuit.png")
```

**Practical Applications**:

1. **Model Interpretability**: Understand why models make certain predictions by identifying key features and their interactions.

   *Example*: In a medical diagnosis system, identify which features detect symptoms and how they combine to predict diseases.

2. **Bias Detection**: Locate where biases manifest within the model's representations.

   *Example*: In a resume screening system, find features that inappropriately associate gender with job suitability.

3. **Knowledge Localization**: Determine where specific types of knowledge are stored in the model.

   *Example*: In a language model, locate features responsible for syntactic relationships, entity knowledge, or factual information.

4. **Model Debugging**: Find and fix issues in model behavior by identifying problematic features.

   *Example*: In a recommendation system, identify features that cause overemphasis on popularity rather than relevance.

5. **Model Distillation**: After identifying critical circuits, create smaller models that preserve key functionalities.

   *Example*: Distill a large language model into a specialized model for specific tasks by preserving only the most important circuits.

### 4. Performing Layer-wise Analysis

**Scenario**: You want to understand which layers of a model are crucial for specific tasks or predictions.

```python
from transcoder import TranscoderReplacementContext

# Assume model and transcoders are already set up

# 1. Get baseline output
input_text = "What is the capital of France?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
baseline_output = adapted_model(input_ids)

# 2. Check importance of each layer by excluding it from transcoder replacement
layer_importance = {}
for layer in range(adapted_model.n_layers):
    # Run model with this layer excluded from transcoder replacement
    with TranscoderReplacementContext(adapted_model, transcoders, excluded_layers={layer}):
        output = adapted_model(input_ids)
        
    # Calculate the difference from baseline output
    diff = torch.norm(output - baseline_output).item()
    layer_importance[layer] = diff
    
print("Layer importance (higher value = more important):")
for layer, impact in sorted(layer_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"Layer {layer}: Impact = {impact:.6f}")

# 3. Focus analysis on the most important layer
most_important_layer = max(layer_importance.items(), key=lambda x: x[1])[0]
most_important_transcoder = transcoders[most_important_layer]

# 4. Analyze feature activations in the important layer
with torch.no_grad():
    _, cache = adapted_model.run_with_cache(input_ids, 
        names_filter=[most_important_transcoder.cfg.hook_point])
    activations = cache[most_important_transcoder.cfg.hook_point]
    _, feature_activations = most_important_transcoder(activations)
    
# 5. Get top features
top_indices, top_values = most_important_transcoder.top_features(
    feature_activations[0, -1], k=10)

print(f"\nTop features in layer {most_important_layer}:")
for idx, val in zip(top_indices.tolist(), top_values.tolist()):
    print(f"Feature {idx}: Activation = {val:.6f}")
```

**Benefits**: Layer-wise analysis helps you:
- Determine which layers are most critical for specific tasks
- Optimize model size by potentially removing less important layers
- Understand how information flows and transforms through the model
- Target your analysis efforts on the most relevant parts of the model

## 💡 Usage Examples

### Example 1: Basic Setup and Circuit Tracing

```python
import torch
from model import Transformer
from transcoder import TranscoderConfig, Transcoder, create_transcoder_pipeline
from circuit_analysis import CircuitTracer, FeatureVector

# Create a transformer model
model = Transformer(d_model=64, n_layers=3, n_heads=4, d_mlp=128, vocab_size=1000)

# Create transcoders for all layers
transcoders = create_transcoder_pipeline(
    model, 
    downstream_propagation=True  # Enable multi-layer effects
)

# Create input data
tokens = torch.randint(0, 1000, (1, 10))

# Initialize circuit tracer
tracer = CircuitTracer(model, transcoders)

# Run model on input data
tracer.run_model(tokens)

# Create a feature vector representing a direction of interest
token_idx = 5
layer = 1
activation = tracer.cache[f'resid_post_{layer}'][0, token_idx]
feature_vector = FeatureVector(
    component_path=[],
    vector=activation,
    layer=layer,
    sublayer='resid_post',
    token=token_idx
)

# Trace circuit
paths = tracer.trace_circuit(feature_vector, num_iters=3, num_branches=5)

# Visualize circuit
edges, nodes = tracer.get_circuit_graph(paths, add_error_nodes=True)
fig = tracer.visualize_circuit(edges, nodes, title="Circuit Visualization")
```

### Example 2: Training Transcoders with Downstream Effects

```python
import torch
import torch.optim as optim
from model import Transformer
from transcoder import (
    TranscoderConfig, Transcoder, train_transcoder, train_multi_layer_transcoder
)

# Create model and data
model = Transformer(d_model=64, n_layers=3, n_heads=4, d_mlp=128, vocab_size=1000)
data = torch.randint(0, 1000, (100, 10))  # 100 sequences of length 10

# Create a transcoder for a specific layer
layer = 0
cfg = TranscoderConfig(
    d_model=model.d_model,
    d_hidden=model.d_mlp,
    hook_point=f'normalized_{layer}_ln2',
    hook_point_layer=layer,
    l1_coefficient=1e-3,
    is_post_ln=True
)
transcoder = Transcoder(cfg)

# Method 1: Train single transcoder with downstream effects
optimizer = optim.Adam(transcoder.parameters(), lr=0.001)
downstream_layers = [1, 2]  # Layers affected by this transcoder
losses = train_transcoder(
    model, 
    transcoder, 
    data, 
    optimizer, 
    batch_size=32, 
    num_epochs=5,
    downstream_layers=downstream_layers  # Include downstream effects
)

# Method 2: Create and train all transcoders jointly
transcoders = {
    0: Transcoder(TranscoderConfig(d_model=64, d_hidden=128, hook_point='normalized_0_ln2', hook_point_layer=0)),
    1: Transcoder(TranscoderConfig(d_model=64, d_hidden=128, hook_point='normalized_1_ln2', hook_point_layer=1)),
    2: Transcoder(TranscoderConfig(d_model=64, d_hidden=128, hook_point='normalized_2_ln2', hook_point_layer=2))
}

# Train all transcoders together with inter-layer effects
layer_losses = train_multi_layer_transcoder(
    model,
    transcoders,
    data,
    batch_size=32,
    num_epochs=5,
    excluded_layers=None  # Include all layers
)
```

### Example 3: Selectively Applying and Modifying Transcoders

```python
from model import Transformer
from transcoder import TranscoderReplacementContext, create_transcoder_pipeline

# Create model and transcoders
model = Transformer(d_model=64, n_layers=5, n_heads=4, d_mlp=128, vocab_size=1000)
transcoders = create_transcoder_pipeline(model)

# Example 1: Apply transcoders to all layers except 1 and 3
input_data = torch.randint(0, 1000, (1, 10))
excluded_layers = {1, 3}

with TranscoderReplacementContext(model, transcoders, excluded_layers=excluded_layers):
    output = model(input_data)
    # Layers 0, 2, and 4 use transcoders, layers 1 and 3 use original MLPs

# Example 2: Modify specific features during replacement
feature_weights = {
    (0, 5): 0.5,    # Layer 0, feature 5 at 50% strength
    (0, 10): 0.0,   # Layer 0, feature 10 completely ablated (zeroed)
    (2, 15): 2.0,   # Layer 2, feature 15 amplified (doubled)
}

with TranscoderReplacementContext(
    model, transcoders, excluded_layers=None, apply_feature_weights=feature_weights
):
    modified_output = model(input_data)
    # Features are modified according to the weights
```

### Example 4: Analyzing Feature Importance

```python
import torch
from model import Transformer
from transcoder import (
    create_transcoder_pipeline, visualize_feature_importance, ablate_features
)

# Setup model and transcoders
model = Transformer(d_model=64, n_layers=3, n_heads=4, d_mlp=128, vocab_size=1000)
transcoders = create_transcoder_pipeline(model)
data = torch.randint(0, 1000, (10, 10))

# Identify important features across layers
importance_scores = visualize_feature_importance(
    model, 
    transcoders, 
    data, 
    top_k=5,  # Show top 5 features per layer
    excluded_layers=None
)

print("Top features by importance:")
for (layer, feature_idx), score in sorted(
    importance_scores.items(), 
    key=lambda x: x[1], 
    reverse=True
)[:10]:
    print(f"Layer {layer}, Feature {feature_idx}: {score:.4f}")

# Ablation experiment: Measure impact of removing specific features
features_to_ablate = [(0, 5), (1, 10), (2, 15)]  # List of (layer, feature) pairs
ablation_results = ablate_features(
    model,
    transcoders,
    data,
    features_to_ablate
)

print(f"Ablation impact (L2 norm): {ablation_results['l2_norm']:.4f}")
print("Top affected tokens:")
for token_idx, impact in ablation_results['top_affected_tokens'].items():
    print(f"Token {token_idx}: {impact:.4f}")
```

### Example 5: Multi-Layer Transcoder for Complex Circuits

```python
import torch
from model import Transformer
from transcoder import (
    MultiLayerTranscoder, TranscoderConfig, Transcoder
)

# Create model
model = Transformer(d_model=64, n_layers=4, n_heads=4, d_mlp=128, vocab_size=1000)

# Create individual transcoders
transcoders = []
for layer in range(4):
    cfg = TranscoderConfig(
        d_model=model.d_model,
        d_hidden=model.d_mlp,
        hook_point=f'normalized_{layer}_ln2',
        hook_point_layer=layer,
        l1_coefficient=1e-3,
        is_post_ln=True,
        downstream_layers=list(range(layer+1, 4))  # All subsequent layers
    )
    transcoders.append(Transcoder(cfg))

# Create a multi-layer transcoder that propagates through the model
multi_layer_tc = MultiLayerTranscoder(model, transcoders)

# Process a batch of activations through multiple layers
input_data = torch.randn(1, 10, model.d_model)  # Batch size 1, seq len 10
output = multi_layer_tc(
    input_data,
    start_layer=0,
    end_layer=3,
    excluded_layers={2}  # Skip layer 2
)

# Modify specific features and propagate effects
multi_layer_tc.decode_with_weights({
    (0, 5): 0.0,   # Zero out feature 5 in layer 0
    (1, 10): 2.0,  # Amplify feature 10 in layer 1
})

# Reset all feature weights to default
multi_layer_tc.reset_all_weights()
```

## 🔍 Localization Experiments

One powerful application of transcoders is localizing important layers and features for specific tasks. This example demonstrates how to identify which layers and features are crucial for a specific prediction:

```python
import torch
from model import Transformer
from transcoder import TranscoderReplacementContext, create_transcoder_pipeline

# Create model and transcoders
model = Transformer(d_model=64, n_layers=4, n_heads=4, d_mlp=128, vocab_size=1000)
transcoders = create_transcoder_pipeline(model)

# Sample input
input_data = torch.randint(0, 1000, (1, 10))

# Get baseline output
baseline_output = model(input_data)
target_token_idx = 5
baseline_prob = torch.softmax(baseline_output[0, -1], dim=0)[target_token_idx].item()

# Test each layer's importance by excluding it
layer_importance = {}
for layer in range(model.n_layers):
    # Run model with this layer excluded from transcoder replacement
    with TranscoderReplacementContext(model, transcoders, excluded_layers={layer}):
        output = model(input_data)
        prob = torch.softmax(output[0, -1], dim=0)[target_token_idx].item()
        
    # Measure impact on target probability
    impact = baseline_prob - prob
    layer_importance[layer] = impact
    
print("Layer importance:")
for layer, impact in sorted(layer_importance.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"Layer {layer}: Impact = {impact:.6f}")

# Identify critical features in the most important layer
most_important_layer = max(layer_importance.items(), key=lambda x: abs(x[1]))[0]
transcoder = transcoders[most_important_layer]

# Get feature activations
with torch.no_grad():
    _, cache = model.run_with_cache(
        input_data, 
        names_filter=[transcoder.cfg.hook_point]
    )
    _, feature_acts = transcoder(cache[transcoder.cfg.hook_point])
    feature_acts = feature_acts[0, -1]  # Get last token

# Get top features
top_features, values = transcoder.top_features(feature_acts, k=5)

print(f"\nTop features in layer {most_important_layer}:")
for feature_idx, value in zip(top_features.tolist(), values.tolist()):
    print(f"Feature {feature_idx}: Activation = {value:.6f}")
```

## 🚧 Advanced Configuration Options

The toolkit provides various configuration options for fine-tuning the analysis:

```python
# TranscoderConfig options
config = TranscoderConfig(
    d_model=256,                      # Model dimension
    d_hidden=512,                     # Hidden dimension for transcoder
    hook_point='normalized_0_ln2',    # Activation hook point
    hook_point_layer=0,               # Layer number
    l1_coefficient=1e-3,              # Sparsity coefficient
    is_post_ln=True,                  # Whether to operate on post-LN activations
    downstream_layers=[1, 2, 3]       # Downstream layers to affect
)

# Circuit tracing options
paths = tracer.trace_circuit(
    feature_vector,
    num_iters=3,                # Number of iterations for path expansion
    num_branches=5,             # Number of branches to explore per iteration
    ignore_bos=True,            # Ignore beginning-of-sequence token
    do_raw_attribution=False,   # Use raw attribution instead of zero ablation
    filter=None                 # Optional filter for path components
)

# Visualization options
fig = tracer.visualize_circuit(
    edges, 
    nodes,
    y_mult=1.0,                 # Vertical spacing multiplier
    width=1000,                 # Figure width in pixels
    height=800,                 # Figure height in pixels
    arrow_width_multiplier=3.0, # Width scaling for arrows
    node_size_multiplier=20.0,  # Size scaling for nodes
    show_edge_labels=True,      # Show contribution values on edges
    title="Circuit Visualization"
)
```

## 📊 Interpreting Results

The circuit visualization provides insight into how information flows through the model:

- **Node colors** indicate component types (MLPs, attention heads)
- **Edge colors** show contribution sign (green for positive, red for negative)
- **Edge thickness** represents contribution magnitude
- **Node position** corresponds to layer and component type
- **Error nodes** represent reconstruction errors in transcoders

Use these visualizations to:
1. Identify key components responsible for specific predictions
2. Understand inter-layer interactions and dependencies
3. Find potential interventions for model steering
4. Debug unexpected model behaviors

## 📖 References and Further Reading

For more information on the techniques used in this toolkit:

- [Transformer Circuits](https://transformer-circuits.pub/) - Research on understanding transformer computations
- [Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2309.08600) - Background on using sparse autoencoders for feature decomposition
- [Mechanistic Interpretability](https://distill.pub/2020/circuits/) - Overview of circuit analysis in neural networks

## 🏷️ License

This project is licensed under the MIT License - see the LICENSE file for details. 