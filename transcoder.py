import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union, Set
from tqdm import tqdm

@dataclass
class TranscoderConfig:
    """Configuration for a Transcoder model."""
    d_model: int  # Model dimension
    d_hidden: int  # Hidden dimension for the transcoder
    hook_point: str  # Name of activation hook point (e.g. 'normalized_0_ln2')
    hook_point_layer: int  # Layer number for the hook point
    l1_coefficient: float = 1e-3  # L1 sparsity coefficient
    
    # Whether transcoder operates on the normalized activations (post-LayerNorm)
    is_post_ln: bool = True
    
    # For multi-layer transcoders, the list of downstream layers to affect
    # Default to an empty list which will be populated with all subsequent layers
    # unless excluded_layers is specified
    downstream_layers: List[int] = field(default_factory=list)
    
    # Layers that should not be affected by the transcoder
    excluded_layers: List[int] = field(default_factory=list)
    
    # Whether to propagate effects to downstream layers
    affect_downstream: bool = True

class SparseAutoencoder(nn.Module):
    """
    A simple sparse autoencoder implementation for transformer feature interpretation.
    This is the base for transcoders.
    """
    def __init__(self, cfg: TranscoderConfig):
        super().__init__()
        self.cfg = cfg
        
        # Encoder (d_model -> d_hidden)
        self.W_enc = nn.Parameter(torch.empty(cfg.d_model, cfg.d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden))
        
        # Decoder (d_hidden -> d_model)
        self.W_dec = nn.Parameter(torch.empty(cfg.d_hidden, cfg.d_model))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse feature space.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Encoded features of shape [batch_size, seq_len, d_hidden]
        """
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_hidden]
        h = F.linear(x, self.W_enc, self.b_enc)
        return h
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode from sparse feature space back to activation space.
        
        Args:
            h: Encoded features of shape [batch_size, seq_len, d_hidden]
            
        Returns:
            Reconstructed activations of shape [batch_size, seq_len, d_model]
        """
        # [batch_size, seq_len, d_hidden] -> [batch_size, seq_len, d_model]
        x_recon = F.linear(h, self.W_dec.t(), self.b_dec)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the sparse autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of (reconstructed_x, pre_activation_features)
        """
        # Reshape if needed
        original_shape = x.shape
        if len(original_shape) > 3:
            x = x.reshape(-1, original_shape[-1])
        
        # Encode
        h_pre = self.encode(x)
        
        # Apply activation (ReLU for sparsity)
        h = F.relu(h_pre)
        
        # Decode
        x_recon = self.decode(h)
        
        # Reshape back if needed
        if len(original_shape) > 3:
            x_recon = x_recon.reshape(original_shape)
            h = h.reshape(*original_shape[:-1], self.cfg.d_hidden)
        
        return x_recon, h
    
    def loss_fn(self, x: torch.Tensor, x_recon: torch.Tensor, h: torch.Tensor, beta_sparsity=None) -> torch.Tensor:
        """
        Compute loss for the sparse autoencoder.
        
        Args:
            x: Original input tensor
            x_recon: Reconstructed output tensor
            h: Encoded features (post-activation)
            beta_sparsity: Optional override for the sparsity coefficient
            
        Returns:
            Total loss
        """
        # Use provided beta_sparsity if given, otherwise use the config value
        l1_coef = beta_sparsity if beta_sparsity is not None else self.cfg.l1_coefficient
        
        # MSE reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # L1 sparsity loss
        l1_loss = h.abs().mean()
        
        # Total loss
        total_loss = recon_loss + l1_coef * l1_loss
        
        return total_loss

class Transcoder(SparseAutoencoder):
    """
    Transcoder for circuit analysis - extends SparseAutoencoder with methods for 
    analyzing feature importance and interpretability.
    """
    def __init__(self, cfg: TranscoderConfig):
        super().__init__(cfg)
        self.feature_labels = None  # Optional feature labels for interpretability
        self.feature_weights = None  # Optional weights to modify specific features
        self.feature_acts = None    # Store feature activations during training
        
    def get_feature_activations(self, model, tokens, token_idx=-1):
        """
        Get feature activations for a specific token in the input.
        
        Args:
            model: Transformer model
            tokens: Input token IDs
            token_idx: Index of the token to analyze (-1 means the last token)
            
        Returns:
            Feature activations for the specified token
        """
        with torch.no_grad():
            # Forward pass through the model and capture activations
            outputs = model(tokens)
            
            # If feature_acts was captured during forward pass, use it
            if self.feature_acts is not None:
                h = self.encode(self.feature_acts)
                h = F.relu(h)  # Apply activation
                
                # Extract activations for the specified token
                if token_idx == -1:
                    token_idx = tokens.shape[-1] - 1
                
                # Handle both batched and unbatched inputs
                if len(h.shape) > 2:
                    token_acts = h[0, token_idx]
                else:
                    token_acts = h[token_idx]
                
                return token_acts
            
            # If no activations were captured, return None
            return None
    
    def top_features(self, activations, k=10):
        """
        Get the top-k active features.
        
        Args:
            activations: Feature activations tensor
            k: Number of top features to return
            
        Returns:
            Tuple of (top feature indices, top feature values)
        """
        # Get top-k feature indices and values
        values, indices = torch.topk(activations, k=k)
        return indices, values
    
    def set_feature_labels(self, labels):
        """Set human-readable labels for features."""
        assert len(labels) == self.cfg.d_hidden, "Number of labels must match d_hidden"
        self.feature_labels = labels
    
    def feature_to_vector(self, feature_idx):
        """Convert a feature index to its corresponding latent space vector."""
        return self.W_dec[:, feature_idx]
    
    def set_feature_weights(self, weights_dict):
        """
        Set weights for specific features to modify their influence.
        
        Args:
            weights_dict: Dictionary mapping feature indices to weight multipliers
        """
        if self.feature_weights is None:
            self.feature_weights = torch.ones(self.cfg.d_hidden, device=self.W_dec.device)
        
        for idx, weight in weights_dict.items():
            if 0 <= idx < self.cfg.d_hidden:
                self.feature_weights[idx] = weight
    
    def reset_feature_weights(self):
        """Reset all feature weights to their default values."""
        self.feature_weights = None
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode from sparse feature space back to activation space.
        Override to apply feature weights during decoding.
        
        Args:
            h: Encoded features of shape [batch_size, seq_len, d_hidden]
            
        Returns:
            Reconstructed activations of shape [batch_size, seq_len, d_model]
        """
        # Apply feature weights if set
        if self.feature_weights is not None:
            # Reshape weights to match h's batch dimension
            if len(h.shape) > 2:
                weight_shape = [1] * (len(h.shape) - 1) + [self.cfg.d_hidden]
                weights = self.feature_weights.view(*weight_shape)
                h = h * weights
            else:
                h = h * self.feature_weights
                
        # [batch_size, seq_len, d_hidden] -> [batch_size, seq_len, d_model]
        x_recon = F.linear(h, self.W_dec.t(), self.b_dec)
        return x_recon
    
    def get_feature_influence(self, feature_idx, model_output):
        """
        Compute the influence of a specific feature on the model output.
        
        Args:
            feature_idx: Index of the feature to analyze
            model_output: Output logits from the model
            
        Returns:
            Influence vector for the feature
        """
        # Get the feature's decoder weights
        feature_vector = self.feature_to_vector(feature_idx)
        
        # Project to model output space (d_model -> vocab_size)
        with torch.no_grad():
            influence = model_output.unembed.weight @ feature_vector
            
        return influence

class MultiLayerTranscoder(nn.Module):
    """
    A transcoder that propagates through multiple layers of the model.
    This allows for more accurate circuit analysis across layers.
    """
    def __init__(self, model, transcoders):
        super().__init__()
        self.model = model
        self.transcoders = transcoders
        self.layer_mapping = {t.cfg.hook_point_layer: t for t in transcoders}
        
    def forward(self, x, start_layer, end_layer=None, token_idx=None, excluded_layers=None):
        """
        Forward pass through multiple layers using transcoders.
        
        Args:
            x: Input tensor
            start_layer: Layer to start from
            end_layer: Layer to end at (or None for all remaining layers)
            token_idx: Optional token index to focus on
            excluded_layers: Optional set of layer indices to exclude from transcoder application
            
        Returns:
            Processed tensor after passing through transcoders
        """
        batch_size, seq_len, d_model = x.shape
        end_layer = end_layer or self.model.n_layers - 1
        excluded_layers = excluded_layers or set()
        
        # Focus on a specific token if requested
        if token_idx is not None:
            x = x[:, token_idx:token_idx+1]
        
        # Cache intermediate results for proper propagation
        layer_inputs = {}
        layer_outputs = {}
        
        # Store initial input
        current_x = x
        
        # First pass: collect all layer inputs and outputs using original model blocks
        with torch.no_grad():
            for layer in range(start_layer, end_layer + 1):
                # Cache layer input
                layer_inputs[layer] = current_x
                
                # Apply layer normalization
                if hasattr(self.model.blocks[layer], 'ln2'):
                    normalized = self.model.blocks[layer].ln2(current_x)
                else:
                    normalized = current_x
                
                # Apply MLP
                mlp_out, _ = self.model.blocks[layer].mlp(normalized, layer_idx=layer)
                
                # Apply residual connection
                current_x = current_x + mlp_out
                
                # Cache layer output
                layer_outputs[layer] = current_x
        
        # Second pass: apply transcoders and propagate effects
        x = layer_inputs[start_layer]
        
        for layer in range(start_layer, end_layer + 1):
            # Apply normalization as in the original model
            if hasattr(self.model.blocks[layer], 'ln2'):
                normalized = self.model.blocks[layer].ln2(x)
            else:
                normalized = x
            
            # Apply transcoder if this layer has one and is not excluded
            if layer in self.layer_mapping and layer not in excluded_layers:
                transcoder = self.layer_mapping[layer]
                x_reconstructed, _ = transcoder(normalized)
                
                # Update current layer output
                x = x + x_reconstructed
                
                # Propagate changes to downstream layers
                if layer < end_layer:
                    delta = x - layer_outputs[layer]
                    
                    # Apply delta to all downstream layers
                    for next_layer in range(layer + 1, end_layer + 1):
                        if next_layer not in excluded_layers:
                            layer_inputs[next_layer] = layer_inputs[next_layer] + delta
            else:
                # Use original output
                x = layer_outputs[layer]
        
        # Restore original sequence length if needed
        if token_idx is not None:
            result = torch.zeros(batch_size, seq_len, d_model, device=x.device)
            result[:, token_idx:token_idx+1] = x
            return result
        return x
        
    def decode_with_weights(self, weights_by_layer_feature):
        """
        Apply specific weights to features across layers.
        
        Args:
            weights_by_layer_feature: Dictionary mapping (layer, feature_idx) to weight value
            
        Returns:
            Self (for method chaining)
        """
        for (layer, feature_idx), weight in weights_by_layer_feature.items():
            if layer in self.layer_mapping:
                transcoder = self.layer_mapping[layer]
                
                # Initialize weights if needed
                if transcoder.feature_weights is None:
                    transcoder.feature_weights = torch.ones(
                        transcoder.cfg.d_hidden, 
                        device=transcoder.W_dec.device
                    )
                
                # Apply weight
                transcoder.feature_weights[feature_idx] = weight
        
        return self
    
    def reset_all_weights(self):
        """Reset all feature weights across all transcoders."""
        for transcoder in self.transcoders:
            transcoder.reset_feature_weights()
        return self

class TranscoderRetrievalHook:
    """Hook for retrieving activations from a module."""
    
    def __init__(self, transcoder, shared_activations):
        self.transcoder = transcoder
        self.shared_activations = shared_activations
        
    def __call__(self, module, input_tensor, output_tensor):
        # Store the activations directly
        self.transcoder.last_activations = output_tensor
        # Also store in the shared dictionary for propagation
        self.shared_activations["activations"] = output_tensor
        return output_tensor

class MultiLayerTranscoderWrapper(torch.nn.Module):
    """
    A wrapper for modules that applies a transcoder to the output and handles downstream effects.
    
    This class wraps a module and applies a transcoder to its output. It can be configured to
    be a source layer (with a transcoder) or a downstream layer that receives effects from
    upstream transcoders.
    """
    
    def __init__(
        self,
        original_module,
        transcoder=None,
        is_source=True,
        propagate_downstream=False,
        shared_activations=None,
    ):
        """
        Initialize the wrapper.
        
        Args:
            original_module: The original module to wrap
            transcoder: The transcoder to apply to the output (can be None for downstream layers)
            is_source: Whether this layer is a source layer (with a transcoder)
            propagate_downstream: Whether to propagate effects to downstream layers
            shared_activations: Dictionary for sharing activations across layers
        """
        super().__init__()
        self.original_module = original_module
        self.transcoder = transcoder
        self.is_source = is_source
        self.propagate_downstream = propagate_downstream
        self.shared_activations = shared_activations or {"activations": None}
        
    def forward(self, *args, **kwargs):
        """
        Forward pass through the wrapper.
        
        For source layers:
            1. Get the original output
            2. Apply the transcoder
            3. Calculate the delta between original and transcoded output
            4. Store the delta for downstream layers
            5. Return the transcoded output
            
        For downstream layers:
            1. Get the original output
            2. Apply the upstream delta to the output
            3. Return the modified output
        """
        # Get the original output
        original_output = self.original_module(*args, **kwargs)
        
        # For source layers with transcoders
        if self.is_source and self.transcoder is not None:
            # Apply the transcoder to the output
            with torch.set_grad_enabled(self.training):
                # Record the original activations for the transcoder
                if self.training:
                    self.transcoder.feature_acts = original_output.detach()
                
                # Apply the transcoder
                encoded = self.transcoder.encode(original_output)
                transcoded_output = self.transcoder.decode(encoded)
                
                # Calculate the delta between original and transcoded output
                delta = transcoded_output - original_output
                
                # Store the delta for downstream layers if propagation is enabled
                if self.propagate_downstream:
                    self.shared_activations["activations"] = delta
                
                # Return the transcoded output
                return transcoded_output
        
        # For downstream layers
        elif not self.is_source and self.propagate_downstream and self.shared_activations["activations"] is not None:
            # Apply the upstream delta to the output
            delta = self.shared_activations["activations"]
            return original_output + delta
        
        # For layers without transcoder or propagation
        return original_output

class TranscoderReplacementContext:
    """Context manager for replacing MLPs in the model with transcoders."""

    def __init__(
        self,
        gpt2_model,
        transcoder,
        hook_point="mlp",
        hook_point_layer=11,
        propagate_downstream=False,
        excluded_layers=None,
    ):
        """
        Args:
            gpt2_model: The GPT-2 model to replace MLPs in.
            transcoder: The transcoder to use for replacement.
            hook_point: The name of the module to replace, e.g. "mlp".
            hook_point_layer: The layer number to replace, e.g. 11.
            propagate_downstream: Whether to propagate transcoder effects to downstream layers.
            excluded_layers: List of layer indices to exclude from transcoder effects.
        """
        self.gpt2_model = gpt2_model
        self.transcoder = transcoder
        self.hook_point = hook_point
        self.hook_point_layer = hook_point_layer
        self.propagate_downstream = propagate_downstream
        self.excluded_layers = excluded_layers or []
        
        self.original_modules = {}
        self.hooks = []
        self.shared_activations = {"activations": None}
        
    def __enter__(self):
        """Replace MLPs with transcoders."""
        # Reset shared activations
        self.shared_activations = {"activations": None}
        
        # Check if we're targeting a single layer or all layers
        target_layer = self.hook_point_layer
        is_single_layer = target_layer is not None
        
        # Get the number of layers in the model
        num_layers = len(self.gpt2_model.transformer.h)
        
        for layer_idx in range(num_layers):
            # Skip if this layer is in the excluded layers
            if layer_idx in self.excluded_layers:
                continue
                
            # Skip if we're targeting a specific layer and this isn't it
            if is_single_layer and layer_idx != target_layer:
                # However, if we're propagating downstream, we need to process layers after the target
                if not (self.propagate_downstream and layer_idx > target_layer):
                    continue
            
            # Get the module at this layer
            if self.hook_point == "mlp":
                module = self.gpt2_model.transformer.h[layer_idx].mlp
            else:
                raise ValueError(f"Hook point {self.hook_point} not supported")
            
            # Determine if this is the source layer
            is_source = not is_single_layer or layer_idx == target_layer
            
            # Store the original module
            self.original_modules[layer_idx] = module
            
            # Create the wrapper with the appropriate settings
            wrapped_module = MultiLayerTranscoderWrapper(
                original_module=module,
                transcoder=self.transcoder,
                is_source=is_source,
                propagate_downstream=self.propagate_downstream,
                shared_activations=self.shared_activations
            )
            
            # Replace the module
            if self.hook_point == "mlp":
                self.gpt2_model.transformer.h[layer_idx].mlp = wrapped_module
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original MLPs."""
        for layer_idx, original_module in self.original_modules.items():
            if self.hook_point == "mlp":
                self.gpt2_model.transformer.h[layer_idx].mlp = original_module
        
        # Clear the hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Clear shared activations
        self.shared_activations = {"activations": None}
        
        return False  # Don't suppress exceptions

def setup_downstream_hooks(model, transcoders_dict, excluded_layers=None):
    """
    Set up hooks on downstream layers to apply activation deltas from transcoders.
    
    Args:
        model: The transformer model
        transcoders_dict: Dictionary mapping layer indices to transcoders
        excluded_layers: List of layer indices that should not be affected by transcoders
        
    Returns:
        List of hook handles for cleanup
    """
    excluded_layers = excluded_layers or []
    hook_handles = []
    
    # Register hooks on all MLP modules (excluding specified layers)
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx in excluded_layers:
            continue
            
        # Skip layers that have been replaced with transcoders
        if layer_idx in transcoders_dict:
            continue
            
        # Define the hook function that will apply deltas from upstream transcoders
        def apply_delta_hook(module, input_tensor, output_tensor, layer_idx=layer_idx):
            # Check if there are any upstream transcoders that affect this layer
            modified_output = output_tensor
            
            for transcoder_layer, transcoder in transcoders_dict.items():
                # Only apply deltas from upstream transcoders
                if transcoder_layer >= layer_idx:
                    continue
                    
                # Get the transcoder wrapper
                if isinstance(model.blocks[transcoder_layer].mlp, MultiLayerTranscoderWrapper):
                    wrapper = model.blocks[transcoder_layer].mlp
                    
                    # Check if this layer should be affected by the upstream transcoder
                    if layer_idx in wrapper.downstream_layers:
                        # Get the activation delta and apply it
                        delta = wrapper.get_activation_delta()
                        if delta is not None:
                            modified_output = modified_output + delta
            
            return modified_output
        
        # Register the hook on the MLP module
        handle = block.mlp.register_forward_hook(
            lambda module, input_tensor, output_tensor, layer_idx=layer_idx: 
            apply_delta_hook(module, input_tensor, output_tensor, layer_idx)
        )
        hook_handles.append(handle)
    
    return hook_handles

def train_transcoder(
    gpt2_small,
    tokenizer,
    batch_size=16,
    num_batches=100,
    lr=1e-3,
    beta_sparsity=1e-3,
    d_hidden=None,
    hook_point="mlp",
    hook_point_layer=11,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=0,
    propagate_downstream=False,
    excluded_layers=None,
):
    """Train a transcoder on a dataset of tokens.

    Args:
        gpt2_small: The GPT-2 model to obtain activations from.
        tokenizer: The tokenizer for GPT-2.
        batch_size: The batch size for training.
        num_batches: The number of batches to train for.
        lr: The learning rate for the optimizer.
        beta_sparsity: The coefficient for the sparsity loss.
        d_hidden: The dimensionality of the hidden layer.
        hook_point: The name of the module to hook into.
        hook_point_layer: The layer to hook into.
        device: The device to use for training.
        seed: The random seed for reproducibility.
        propagate_downstream: Whether to propagate transcoder effects to downstream layers.
        excluded_layers: List of layer indices to exclude from transcoder effects.

    Returns:
        transcoder: The trained transcoder.
        loss_curve: The loss curve during training.
    """
    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get the model to use for activations
    gpt2_model = gpt2_small

    # Create the transcoder
    d_model = gpt2_model.config.n_embd
    if hook_point == "mlp":
        d_model = gpt2_model.config.n_embd * 4
    
    config = TranscoderConfig(
        d_model=d_model,
        d_hidden=d_hidden,
        hook_point=hook_point,
        hook_point_layer=hook_point_layer,
    )
    transcoder = Transcoder(config).to(device)

    # Generate activation dataset
    activations = []
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate a batch of random tokens
            tokens = torch.randint(
                0, tokenizer.vocab_size, (batch_size, 1024), device=device
            )
            with TranscoderReplacementContext(
                gpt2_model, 
                transcoder, 
                hook_point, 
                hook_point_layer,
                propagate_downstream=propagate_downstream,
                excluded_layers=excluded_layers,
            ) as ctx:
                # Forward pass
                gpt2_model(tokens)
                # Get activations from the transcoder
                if transcoder.feature_acts is not None:
                    activations.append(transcoder.feature_acts.detach())

    # Concatenate all activations
    activations = torch.cat(activations, dim=0)

    # Train the transcoder on the activations
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=lr)
    loss_curve = []

    # Train loop
    for _ in tqdm(range(num_batches)):
        # Get a random batch of activations
        idx = torch.randperm(activations.shape[0])[:batch_size]
        batch = activations[idx].to(device)

        # Forward pass
        encoded = transcoder.encode(batch)
        decoded = transcoder.decode(encoded)
        loss = transcoder.loss_fn(batch, decoded, encoded, beta_sparsity)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        loss_curve.append(loss.item())

    return transcoder, loss_curve

def train_multi_layer_transcoder(model, transcoders, dataset, batch_size=32, num_epochs=5,
                                learning_rate=0.001, excluded_layers=None):
    """
    Train multiple transcoders jointly, considering their interactions.
    
    Args:
        model: Transformer model
        transcoders: List of transcoders to train
        dataset: Dataset of token sequences
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        excluded_layers: Optional set of layer indices to exclude from training
        
    Returns:
        Dictionary mapping layer indices to lists of losses
    """
    # Group transcoders by layer
    if isinstance(transcoders, dict):
        layer_to_transcoder = transcoders
    else:
        layer_to_transcoder = {t.cfg.hook_point_layer: t for t in transcoders}
    
    # Apply excluded layers
    excluded_layers = set(excluded_layers or [])
    layer_to_transcoder = {l: t for l, t in layer_to_transcoder.items() if l not in excluded_layers}
    
    # Setup downstream layers for each transcoder
    for layer, transcoder in layer_to_transcoder.items():
        # Set downstream layers to all layers after the current one
        downstream = [l for l in layer_to_transcoder.keys() if l > layer]
        transcoder.cfg.downstream_layers = downstream
    
    # Create optimizers for each transcoder
    optimizers = {layer: torch.optim.Adam(tc.parameters(), lr=learning_rate) 
                 for layer, tc in layer_to_transcoder.items()}
    
    # Track losses for each layer
    layer_losses = {layer: [] for layer in layer_to_transcoder.keys()}
    
    # Determine all hook points we need
    all_hook_points = []
    for layer in layer_to_transcoder.keys():
        if layer_to_transcoder[layer].cfg.is_post_ln:
            all_hook_points.append(f"normalized_{layer}_ln2")
        else:
            all_hook_points.append(f"resid_mid_{layer}")
    
    for epoch in range(num_epochs):
        epoch_losses = {layer: [] for layer in layer_to_transcoder.keys()}
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Run model and cache all needed activations
            with torch.no_grad():
                _, cache = model.run_with_cache(batch, names_filter=all_hook_points)
            
            # Train each transcoder in layer order (starting from earliest layers)
            for layer in sorted(layer_to_transcoder.keys()):
                transcoder = layer_to_transcoder[layer]
                optimizer = optimizers[layer]
                
                # Get activations for this layer
                if transcoder.cfg.is_post_ln:
                    act_name = f"normalized_{layer}_ln2"
                else:
                    act_name = f"resid_mid_{layer}"
                activations = cache[act_name]
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, features = transcoder(activations)
                
                # Compute standard reconstruction loss
                loss_dict = transcoder.loss_fn(activations, reconstructed, features)
                total_loss = loss_dict['total']
                
                # Compute downstream effects if applicable
                downstream_layers = transcoder.cfg.downstream_layers
                for layer_idx, downstream_layer in enumerate(downstream_layers):
                    # Get target activations for this downstream layer
                    if transcoder.cfg.is_post_ln:
                        downstream_act_name = f"normalized_{downstream_layer}_ln2"
                    else:
                        downstream_act_name = f"resid_mid_{downstream_layer}"
                    
                    target_activations = cache[downstream_act_name]
                    
                    # Propagate the effect (simplified propagation)
                    # For a more accurate propagation, we would need to implement
                    # the full forward pass through intermediate layers
                    current_acts = reconstructed
                    
                    # Compute downstream loss
                    downstream_loss = F.mse_loss(current_acts, target_activations)
                    
                    # Add to total loss with a weight that decreases with layer distance
                    layer_weight = 0.5 ** (layer_idx + 1)  # Exponential decay
                    total_loss += layer_weight * downstream_loss
                
                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()
                
                # Record loss
                epoch_losses[layer].append(total_loss.item())
        
        # Average losses for this epoch
        for layer, losses in epoch_losses.items():
            avg_loss = sum(losses) / len(losses)
            layer_losses[layer].append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Layer {layer}, Loss: {avg_loss:.6f}")
    
    return layer_losses

def visualize_feature_importance(model, transcoders, dataset, top_k=5, excluded_layers=None):
    """
    Visualize the importance of features across different layers.
    
    Args:
        model: Transformer model
        transcoders: Dictionary mapping layers to transcoders
        dataset: Dataset of token sequences
        top_k: Number of top features to display per layer
        excluded_layers: Optional set of layer indices to exclude
        
    Returns:
        Dictionary mapping (layer, feature_idx) to importance score
    """
    # Handle transcoders as list or dict
    if not isinstance(transcoders, dict):
        transcoders = {t.cfg.hook_point_layer: t for t in transcoders}
    
    # Apply excluded layers
    excluded_layers = set(excluded_layers or [])
    layer_to_transcoder = {l: t for l, t in transcoders.items() if l not in excluded_layers}
    
    # Run model on a sample batch
    sample_idx = 0
    sample_batch = dataset[sample_idx:sample_idx+1]
    
    # Determine hook points
    hook_points = []
    for layer, transcoder in layer_to_transcoder.items():
        if transcoder.cfg.is_post_ln:
            hook_points.append(f"normalized_{layer}_ln2")
        else:
            hook_points.append(f"resid_mid_{layer}")
    
    # Get activations
    with torch.no_grad():
        _, cache = model.run_with_cache(
            sample_batch, 
            names_filter=hook_points
        )
    
    # Calculate feature importance for each layer
    importance_scores = {}
    
    for layer, transcoder in layer_to_transcoder.items():
        # Get activations for this layer
        if transcoder.cfg.is_post_ln:
            act_name = f"normalized_{layer}_ln2"
        else:
            act_name = f"resid_mid_{layer}"
        activations = cache[act_name]
        
        # Get feature activations
        _, feature_activations = transcoder(activations)
        
        # Sum over batch and sequence dimensions to get overall feature importance
        feature_importance = feature_activations.sum(dim=(0, 1))
        
        # Get top-k features
        values, indices = torch.topk(feature_importance, min(top_k, len(feature_importance)))
        
        # Store in result dictionary
        for feature_idx, importance in zip(indices.tolist(), values.tolist()):
            importance_scores[(layer, feature_idx)] = importance
    
    return importance_scores

def ablate_features(model, transcoders, dataset, features_to_ablate, excluded_layers=None):
    """
    Ablate specific features and measure their impact.
    
    Args:
        model: Transformer model
        transcoders: Dictionary mapping layers to transcoders
        dataset: Dataset of token sequences
        features_to_ablate: List of (layer, feature_idx) tuples to ablate
        excluded_layers: Optional set of layer indices to exclude
        
    Returns:
        Dictionary with metrics for evaluating the impact of ablation
    """
    # Handle transcoders as list or dict
    if not isinstance(transcoders, dict):
        transcoders = {t.cfg.hook_point_layer: t for t in transcoders}
    
    # Apply excluded layers
    excluded_layers = set(excluded_layers or [])
    
    # Run model on a sample batch
    sample_idx = 0
    sample_batch = dataset[sample_idx:sample_idx+1]
    
    # Run original model
    with torch.no_grad():
        original_logits, _ = model(sample_batch)
    
    # Create a weights dictionary for ablation (set ablated features to 0)
    ablation_weights = {}
    for layer, feature_idx in features_to_ablate:
        if layer not in excluded_layers:
            ablation_weights[(layer, feature_idx)] = 0.0
    
    # Run model with ablated features
    with torch.no_grad(), TranscoderReplacementContext(
        model, transcoders, excluded_layers, apply_feature_weights=ablation_weights
    ):
        ablated_logits, _ = model(sample_batch)
    
    # Calculate metrics
    logit_diff = ablated_logits - original_logits
    l2_norm = torch.norm(logit_diff).item()
    
    # Get top affected tokens
    token_impact = torch.norm(logit_diff, dim=2)[0, -1]  # Impact on last token
    _, top_indices = torch.topk(token_impact, min(5, token_impact.size(0)))
    
    top_affected = {
        idx.item(): token_impact[idx].item() 
        for idx in top_indices
    }
    
    return {
        'l2_norm': l2_norm,
        'top_affected_tokens': top_affected
    }

def create_transcoder_pipeline(model, layer_hidden_dims=None, downstream_propagation=True):
    """
    Create a full set of transcoders for all model layers with proper configuration.
    
    Args:
        model: Transformer model
        layer_hidden_dims: Optional dictionary mapping layer indices to hidden dimension sizes
                         If None, uses model's MLP dimension for all layers
        downstream_propagation: Whether to configure transcoders to propagate to downstream layers
        
    Returns:
        Dictionary mapping layer indices to transcoders
    """
    transcoders = {}
    num_layers = model.n_layers
    
    # Default to model's MLP dimension if not specified
    if layer_hidden_dims is None:
        layer_hidden_dims = {i: model.d_mlp for i in range(num_layers)}
    
    # Create transcoders for each layer
    for layer in range(num_layers):
        # Get hidden dimension for this layer
        d_hidden = layer_hidden_dims.get(layer, model.d_mlp)
        
        # Create configuration
        cfg = TranscoderConfig(
            d_model=model.d_model,
            d_hidden=d_hidden,
            hook_point=f'normalized_{layer}_ln2',
            hook_point_layer=layer,
            l1_coefficient=1e-3,
            is_post_ln=True
        )
        
        # Configure downstream propagation
        if downstream_propagation:
            cfg.downstream_layers = list(range(layer + 1, num_layers))
        
        # Create transcoder
        transcoders[layer] = Transcoder(cfg)
    
    return transcoders 