import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union
import matplotlib.pyplot as plt
import networkx as nx

# Define enums for component types and feature types
class ComponentType(enum.Enum):
    MLP = 'mlp'
    ATTN = 'attn'
    EMBED = 'embed'
    
    # Error terms
    TC_ERROR = 'tc_error'  # Error due to inaccurate transcoders
    PRUNE_ERROR = 'prune_error'  # Error due to only looking at top paths in graph
    BIAS_ERROR = 'bias_error'  # Account for bias terms in transcoders

class FeatureType(enum.Enum):
    NONE = 'none'
    TRANSCODER = 'tc'

class ContribType(enum.Enum):
    RAW = 'raw'
    ZERO_ABLATION = 'zero_ablation'

@dataclass
class Component:
    """An individual component in the model (e.g. an attention head or a transcoder feature)"""
    layer: int
    component_type: ComponentType
    token: Optional[int] = None
    attn_head: Optional[int] = None
    feature_type: Optional[FeatureType] = None
    feature_idx: Optional[int] = None
    
    def __str__(self, show_token=True):
        """String representation of the component"""
        base_str = f'{self.component_type.value}{self.layer}'
        
        attn_str = ''
        if self.component_type == ComponentType.ATTN:
            attn_str = f'[{self.attn_head}]'
            
        feature_str = ''
        if self.feature_type is not None and self.feature_idx is not None:
            feature_str = f"{self.feature_type.value}[{self.feature_idx}]"
            
        token_str = ''
        if self.token is not None and show_token:
            token_str = f'@{self.token}'
            
        return ''.join([base_str, attn_str, feature_str, token_str])
    
    def __repr__(self):
        return f'<Component {str(self)}>'

@dataclass
class FeatureVector:
    """
    A unique feature vector potentially associated with a path of components,
    along with a contribution value.
    """
    # A list of components that can be used to uniquely specify the direction of the feature vector
    component_path: List[Component]
    
    # The actual feature vector
    vector: torch.Tensor
    
    # Metadata about where the feature vector lives
    # sublayer can be 'mlp_out', 'resid_post', 'resid_mid', 'resid_pre'
    layer: int
    sublayer: str
    token: Optional[int] = None
    
    # Contribution information
    contrib: Optional[float] = None
    contrib_type: Optional[ContribType] = None
    
    # Error accounting
    error: float = 0.0
    
    def __post_init__(self):
        if self.token is None and len(self.component_path) > 0:
            self.token = self.component_path[-1].token
        if self.layer is None and len(self.component_path) > 0:
            self.layer = self.component_path[-1].layer
    
    def __str__(self, show_full=True, show_contrib=True, show_last_token=True):
        """String representation of the feature vector."""
        retstr = ''
        token_str = '' if self.token is None or not show_last_token else f'@{self.token}'
        
        if len(self.component_path) > 0:
            if show_full:
                retstr = ''.join(x.__str__(show_token=False) for x in self.component_path[:-1])
            retstr = ''.join([retstr, self.component_path[-1].__str__(show_token=False), token_str])
        else:
            retstr = f'*{self.sublayer}{self.layer}{token_str}'
            
        if show_contrib and self.contrib is not None:
            retstr = ''.join([retstr, f': {self.contrib:.2f}'])
            
        return retstr
    
    def __repr__(self):
        contrib_type_str = '' if self.contrib_type is None else f' contrib_type={self.contrib_type.value}'
        return f'<FeatureVector {str(self)}, sublayer={self.sublayer}{contrib_type_str}>'

@torch.no_grad()
def get_attn_head_contribs(model, cache, layer, range_normal):
    """
    Calculate contributions from attention heads to a specific direction.
    
    Args:
        model: Transformer model
        cache: Activation cache from a forward pass
        layer: Layer index
        range_normal: Direction to project onto
        
    Returns:
        Tensor of contributions from each attention head
    """
    split_vals = cache[f'v_{layer}']
    attn_pattern = cache[f'pattern_{layer}']
    
    # 'batch head dst src, batch src head d_head -> batch head dst src d_head'
    weighted_vals = torch.einsum('bhds,bshf->bhdst', attn_pattern, split_vals)
    
    # 'batch head dst src d_head, head d_model d_head -> batch head dst src d_model'
    weighted_outs = torch.einsum('bhdst,hdt->bhdsd', weighted_vals, model.blocks[layer].attention.W_O)
    
    # 'batch head dst src d_model, d_model -> batch head dst src'
    contribs = torch.einsum('bhdsd,d->bhds', weighted_outs, range_normal)
    
    return contribs

@torch.no_grad()
def get_transcoder_ixg(transcoder, cache, range_normal, input_layer, input_token_idx, 
                      return_numpy=True, is_transcoder_post_ln=True, return_feature_activs=True):
    """
    Get input-times-gradient for transcoder features.
    
    Args:
        transcoder: Transcoder model
        cache: Activation cache
        range_normal: Direction to project onto
        input_layer: Layer index for input
        input_token_idx: Token index for input
        return_numpy: Whether to return numpy arrays
        is_transcoder_post_ln: Whether transcoder operates on post-LN activations
        return_feature_activs: Whether to return feature activations
        
    Returns:
        Pulled-back feature and feature activations
    """
    pulledback_feature = transcoder.W_dec @ range_normal
    
    if is_transcoder_post_ln:
        act_name = f'normalized_{input_layer}_ln2'
    else:
        act_name = f'resid_mid_{input_layer}'
        
    feature_activs = transcoder(cache[act_name])[1][0, input_token_idx]
    pulledback_feature = pulledback_feature * feature_activs
    
    if return_numpy:
        pulledback_feature = pulledback_feature.cpu().numpy()
        feature_activs = feature_activs.cpu().numpy()
        
    if not return_feature_activs:
        return pulledback_feature
    else:
        return pulledback_feature, feature_activs

@torch.no_grad()
def get_ln_constant(model, cache, vector, layer, token, is_ln2=False, recip=False):
    """
    Approximate LayerNorm as a constant when propagating feature vectors backward.
    
    Args:
        model: Transformer model
        cache: Activation cache
        vector: Vector to propagate
        layer: Layer index
        token: Token index
        is_ln2: Whether this is the second LayerNorm in the block
        recip: Whether to return the reciprocal
        
    Returns:
        Scaling constant for the LayerNorm
    """
    x_act_name = f'resid_mid_{layer}' if is_ln2 else f'resid_pre_{layer}'
    x = cache[x_act_name][0, token]
    
    y_act_name = f'normalized_{layer}_ln2' if is_ln2 else f'normalized_{layer}_ln1'
    y = cache[y_act_name][0, token]
    
    if torch.dot(vector, x) == 0:
        return torch.tensor(0.)
        
    return torch.dot(vector, y) / torch.dot(vector, x) if not recip else torch.dot(vector, x) / torch.dot(vector, y)

@dataclass
class FilterType(enum.Enum):
    """Types of filter operations for feature filtering."""
    EQ = enum.auto()  # equals
    NE = enum.auto()  # not equal to
    GT = enum.auto()  # greater than
    GE = enum.auto()  # greater than or equal to
    LT = enum.auto()  # less than 
    LE = enum.auto()  # less than or equal to

@dataclass
class FeatureFilter:
    """Filter for selecting specific features based on their attributes."""
    # Feature-level filters
    layer: Optional[int] = field(default=None, metadata={'filter_level': 'feature'})
    layer_filter_type: FilterType = FilterType.EQ
    sublayer: Optional[str] = field(default=None, metadata={'filter_level': 'feature'})
    sublayer_filter_type: FilterType = FilterType.EQ
    token: Optional[int] = field(default=None, metadata={'filter_level': 'feature'})
    token_filter_type: FilterType = FilterType.EQ
    
    # Filters on last component in component_path
    component_type: Optional[ComponentType] = field(default=None, metadata={'filter_level': 'component'})
    component_type_filter_type: FilterType = FilterType.EQ
    attn_head: Optional[int] = field(default=None, metadata={'filter_level': 'component'})
    attn_head_filter_type: FilterType = FilterType.EQ
    feature_type: Optional[FeatureType] = field(default=None, metadata={'filter_level': 'component'})
    feature_type_filter_type: FilterType = FilterType.EQ
    feature_idx: Optional[int] = field(default=None, metadata={'filter_level': 'component'})
    feature_idx_filter_type: FilterType = FilterType.EQ
    
    def match(self, feature):
        """Check if a feature matches this filter."""
        component = None
        
        for field in field(self.__class__):
            name = field.name
            val = getattr(self, name)
            if val is None:
                continue
                
            try:
                filter_level = field.metadata['filter_level']
            except KeyError:
                continue  # Not a filter
                
            if filter_level == 'feature':
                if val is not None:
                    filter_type = getattr(self, f'{name}_filter_type')
                    feat_val = getattr(feature, name)
                    
                    if filter_type == FilterType.EQ and val != feat_val:
                        return False
                    if filter_type == FilterType.NE and val == feat_val:
                        return False
                    if filter_type == FilterType.GT and feat_val <= val:
                        return False
                    if filter_type == FilterType.GE and feat_val < val:
                        return False
                    if filter_type == FilterType.LT and feat_val >= val:
                        return False
                    if filter_type == FilterType.LE and feat_val > val:
                        return False
                        
            elif filter_level == 'component':
                if component is None:
                    if len(feature.component_path) <= 0:
                        return False
                    component = feature.component_path[-1]
                    
                if val is not None:
                    filter_type = getattr(self, f'{name}_filter_type')
                    comp_val = getattr(component, name)
                    
                    if filter_type == FilterType.EQ and val != comp_val:
                        return False
                    if filter_type == FilterType.NE and val == comp_val:
                        return False
                    if filter_type == FilterType.GT and comp_val <= val:
                        return False
                    if filter_type == FilterType.GE and comp_val < val:
                        return False
                    if filter_type == FilterType.LT and comp_val >= val:
                        return False
                    if filter_type == FilterType.LE and comp_val > val:
                        return False
                        
        return True 

@torch.no_grad()
def get_top_transcoder_features(model, transcoder, cache, feature_vector, layer, k=5):
    """
    Get the top transcoder features that contribute to a feature vector.
    
    Args:
        model: Transformer model
        transcoder: Transcoder model
        cache: Activation cache
        feature_vector: Feature vector to analyze
        layer: Layer index
        k: Number of top features to return
        
    Returns:
        List of (feature_idx, contribution) tuples
    """
    # Get the direction to project onto
    range_normal = feature_vector.vector
    
    # Get token index
    token_idx = feature_vector.token if feature_vector.token is not None else -1
    
    # Get input-times-gradient for transcoder features
    pulledback_features, feature_activs = get_transcoder_ixg(
        transcoder, cache, range_normal, layer, token_idx,
        is_transcoder_post_ln=transcoder.cfg.is_post_ln
    )
    
    # Sort features by contribution (absolute value)
    contributions = np.abs(pulledback_features)
    indices = np.argsort(-contributions)
    
    # Return top-k features and their contributions
    return [(idx, pulledback_features[idx]) for idx in indices[:k]]

@torch.no_grad()
def get_top_contribs(model, transcoders, cache, feature_vector, k=5, ignore_bos=True, 
                    only_return_all_scores=False, cap=None, filter=None):
    """
    Get the top contributors to a feature vector.
    
    Args:
        model: Transformer model
        transcoders: Dictionary mapping layer indices to transcoder models
        cache: Activation cache
        feature_vector: Feature vector to analyze
        k: Number of top contributors to return
        ignore_bos: Whether to ignore the beginning-of-sequence token
        only_return_all_scores: Whether to only return all scores without filtering
        cap: Maximum number of scores to return
        filter: Optional filter for features
        
    Returns:
        List of (feature_vector, contribution) tuples sorted by contribution magnitude
    """
    range_normal = feature_vector.vector
    token = feature_vector.token
    layer = feature_vector.layer
    
    # Skip BOS token if needed
    if ignore_bos and token == 0:
        return []
    
    new_paths = []
    
    # Check which sublayer this is in
    if feature_vector.sublayer == 'resid_post':
        # Look at MLP contribution
        prev_sublayer = 'resid_mid'
        
        # Get normalized activations
        ln2_out = cache[f'normalized_{layer}_ln2']
        
        # Get MLP output
        mlp_out, _ = model.blocks[layer].mlp(ln2_out[:, token:token+1], layer_idx=layer)
        mlp_out = mlp_out.squeeze(1)
        
        # Calculate contribution
        contrib = torch.dot(mlp_out, range_normal).item()
        
        # Create new feature vector for MLP's contribution
        component = Component(
            layer=layer,
            component_type=ComponentType.MLP,
            token=token
        )
        
        prev_fv = FeatureVector(
            component_path=feature_vector.component_path + [component],
            vector=mlp_out,
            layer=layer,
            sublayer=prev_sublayer,
            token=token,
            contrib=contrib,
            contrib_type=ContribType.RAW,
            error=feature_vector.error
        )
        
        # Add to list of paths
        new_paths.append((prev_fv, contrib))
        
        # If we have a transcoder for this layer, decompose MLP output into features
        if layer in transcoders:
            transcoder = transcoders[layer]
            
            # Get top transcoder features
            top_features = get_top_transcoder_features(model, transcoder, cache, feature_vector, layer, k=k)
            
            for feat_idx, feat_contrib in top_features:
                # Create component for this feature
                tc_component = Component(
                    layer=layer,
                    component_type=ComponentType.MLP,
                    token=token,
                    feature_type=FeatureType.TRANSCODER,
                    feature_idx=feat_idx
                )
                
                # Get the feature's decoder vector
                tc_vector = transcoder.W_dec[:, feat_idx]
                
                # Create feature vector for this transcoder feature
                tc_fv = FeatureVector(
                    component_path=feature_vector.component_path + [tc_component],
                    vector=tc_vector,
                    layer=layer,
                    sublayer=prev_sublayer,
                    token=token,
                    contrib=feat_contrib,
                    contrib_type=ContribType.RAW,
                    error=feature_vector.error
                )
                
                new_paths.append((tc_fv, feat_contrib))
    
    # In residual stream after attention, add attention contribution
    if feature_vector.sublayer in ['resid_mid', 'resid_post']:
        attn_sublayer = 'resid_pre' if feature_vector.sublayer == 'resid_mid' else 'resid_mid'
        
        # Get attention head contributions
        attn_contribs = get_attn_head_contribs(model, cache, layer, range_normal)
        
        # Add a path for each attention head
        for head_idx in range(model.n_heads):
            head_contrib = attn_contribs[0, head_idx, token].item()
            
            # Create component for this attention head
            attn_component = Component(
                layer=layer,
                component_type=ComponentType.ATTN,
                token=token,
                attn_head=head_idx
            )
            
            # Get the attention output vector for this head
            attn_pattern = cache[f'pattern_{layer}'][0, head_idx, token]
            v_values = cache[f'v_{layer}'][0, :, head_idx]
            
            # Calculate weighted value vectors
            weighted_values = torch.einsum('s,sd->d', attn_pattern, v_values)
            
            # Project through output weights
            attn_output = torch.einsum('d,dh->h', weighted_values, model.blocks[layer].attention.W_O[head_idx])
            
            # Create feature vector for this attention head
            attn_fv = FeatureVector(
                component_path=feature_vector.component_path + [attn_component],
                vector=attn_output,
                layer=layer,
                sublayer=attn_sublayer,
                token=token,
                contrib=head_contrib,
                contrib_type=ContribType.RAW,
                error=feature_vector.error
            )
            
            new_paths.append((attn_fv, head_contrib))
    
    # Apply filter if provided
    if filter is not None:
        new_paths = [(fv, contrib) for fv, contrib in new_paths if filter.match(fv)]
    
    # Sort by absolute contribution
    new_paths.sort(key=lambda x: -abs(x[1]))
    
    # Apply cap
    if cap is not None:
        new_paths = new_paths[:cap]
    
    if only_return_all_scores:
        return new_paths
    
    # Return top-k paths
    return new_paths[:k]

@torch.no_grad()
def greedy_get_top_paths(model, transcoders, cache, feature_vector, num_iters=2, num_branches=5, 
                         ignore_bos=True, do_raw_attribution=False, filter=None):
    """
    Greedily find the top computational paths contributing to a feature vector.
    
    Args:
        model: Transformer model
        transcoders: Dictionary mapping layer indices to transcoder models
        cache: Activation cache
        feature_vector: Feature vector to analyze
        num_iters: Number of iterations to trace back
        num_branches: Number of branches to explore at each step
        ignore_bos: Whether to ignore the beginning-of-sequence token
        do_raw_attribution: Whether to use raw attribution
        filter: Optional filter for features
        
    Returns:
        List of all paths found
    """
    all_paths = [feature_vector]
    
    # Start with the initial feature vector
    frontier = [feature_vector]
    
    # Perform iterative backward tracing
    for i in range(num_iters):
        new_frontier = []
        
        # For each feature vector in the frontier
        for fv in frontier:
            # Get top contributors to this feature vector
            top_contribs = get_top_contribs(
                model, transcoders, cache, fv, 
                k=num_branches, 
                ignore_bos=ignore_bos,
                filter=filter
            )
            
            # Add contributors to all paths and new frontier
            for new_fv, contrib in top_contribs:
                all_paths.append(new_fv)
                new_frontier.append(new_fv)
        
        # Update frontier for next iteration
        frontier = new_frontier
        
        # Break if no new paths were found
        if len(frontier) == 0:
            break
    
    return all_paths

def print_all_paths(paths):
    """Print all paths in a readable format."""
    for path in paths:
        print(path)

@torch.no_grad()
def get_raw_top_features_among_paths(all_paths, use_tokens=True, top_k=5, filter_layers=None, filter_sublayers=None):
    """
    Get the top features across all paths.
    
    Args:
        all_paths: List of feature vectors (paths)
        use_tokens: Whether to use tokens in feature key
        top_k: Number of top features to return
        filter_layers: Optional list of layers to filter by
        filter_sublayers: Optional list of sublayers to filter by
        
    Returns:
        Dictionary mapping feature keys to their total contributions
    """
    feature_contribs = {}
    
    for path in all_paths:
        # Skip if no component path
        if len(path.component_path) == 0:
            continue
        
        # Skip if layer filter is applied and this layer doesn't match
        if filter_layers is not None and path.layer not in filter_layers:
            continue
        
        # Skip if sublayer filter is applied and this sublayer doesn't match
        if filter_sublayers is not None and path.sublayer not in filter_sublayers:
            continue
        
        # Get the last component in the path
        last_comp = path.component_path[-1]
        
        # Create key based on component type and other metadata
        if last_comp.component_type == ComponentType.ATTN:
            key = f"attn{last_comp.layer}[{last_comp.attn_head}]"
            if use_tokens:
                key += f"@{last_comp.token}"
        elif last_comp.component_type == ComponentType.MLP:
            key = f"mlp{last_comp.layer}"
            if last_comp.feature_type == FeatureType.TRANSCODER:
                key += f".tc[{last_comp.feature_idx}]"
            if use_tokens:
                key += f"@{last_comp.token}"
        else:
            # Skip other component types
            continue
        
        # Add contribution to dictionary
        if key in feature_contribs:
            feature_contribs[key] += path.contrib if path.contrib is not None else 0
        else:
            feature_contribs[key] = path.contrib if path.contrib is not None else 0
    
    # Sort by absolute contribution
    sorted_features = sorted(feature_contribs.items(), key=lambda x: -abs(x[1]))
    
    # Return top-k features
    return dict(sorted_features[:top_k])

def path_to_str(path, show_contrib=False, show_last_token=False):
    """Convert a path to a string representation."""
    return path.__str__(show_contrib=show_contrib, show_last_token=show_last_token)

@torch.no_grad()
def paths_to_graph(all_paths, model=None, cache=None, use_jacobian=True, prune=True, threshold=0.01, max_edges=100):
    """
    Convert a list of paths to a graph representation.
    
    Args:
        all_paths: List of feature vectors (paths)
        model: Optional transformer model (required if use_jacobian=True)
        cache: Optional activation cache (required if use_jacobian=True)
        use_jacobian: Whether to use Jacobian-based weight calculation
        prune: Whether to prune the graph
        threshold: Minimum absolute weight for an edge to be kept
        max_edges: Maximum number of edges to keep
        
    Returns:
        Tuple of (edges, nodes) for the graph, and optional pruning metrics
    """
    # Dictionary of edges: (src, dst) -> weight
    edges = {}
    
    # Dictionary of nodes: key -> {layer, sublayer, type, token}
    nodes = {}
    
    # Calculate path weights
    for path in all_paths:
        # Skip incomplete paths
        if len(path.component_path) == 0:
            continue
        
        # Create node for this path
        path_key = path.__str__(show_contrib=False)
        
        last_comp = path.component_path[-1]
        nodes[path_key] = {
            'layer': path.layer,
            'sublayer': path.sublayer,
            'type': last_comp.component_type.value,
            'token': last_comp.token
        }
        
        # Get path's contribution
        contrib = path.contrib if path.contrib is not None else 0.0
        
        # Add edges for paths of length > 1
        if len(path.component_path) > 1:
            # Convert subpaths to strings
            prev_path = FeatureVector(
                component_path=path.component_path[:-1],
                vector=None,  # Not needed for string conversion
                layer=None,   # Will be filled in __post_init__
                sublayer=None,
                contrib=None
            )
            prev_key = prev_path.__str__(show_contrib=False)
            
            # Add edge
            edge_key = (prev_key, path_key)
            
            # If using Jacobian-based weights and we have model and cache
            if use_jacobian and model is not None and cache is not None:
                # Find the source feature vector in all_paths
                src_feature = None
                for p in all_paths:
                    if p.__str__(show_contrib=False) == prev_key:
                        src_feature = p
                        break
                
                if src_feature is not None and src_feature.vector is not None and path.vector is not None:
                    # Calculate Jacobian-based weight
                    weight = calculate_jacobian_weights(model, cache, src_feature, path)
                else:
                    # Fallback to contribution
                    weight = contrib
            else:
                # Use contribution as weight
                weight = contrib
            
            # Add to edges
            if edge_key in edges:
                edges[edge_key] += weight
            else:
                edges[edge_key] = weight
    
    # Create singleton nodes for paths of length 1
    for path in all_paths:
        if len(path.component_path) <= 0:
            continue
            
        path_key = path.__str__(show_contrib=False)
        
        if path_key not in nodes:
            last_comp = path.component_path[-1]
            nodes[path_key] = {
                'layer': path.layer,
                'sublayer': path.sublayer,
                'type': last_comp.component_type.value,
                'token': last_comp.token
            }
    
    # Apply pruning if requested
    pruning_metrics = None
    if prune:
        edges, nodes, pruning_metrics = prune_circuit_graph(edges, nodes, threshold, max_edges)
        
    if pruning_metrics:
        return edges, nodes, pruning_metrics
    else:
        return edges, nodes

@torch.no_grad()
def add_error_nodes_to_graph(model, cache, transcoders, edges, nodes, do_bias=True):
    """
    Add error nodes to the graph representing error in transcoders and pruned paths.
    
    Args:
        model: Transformer model
        cache: Activation cache
        transcoders: Dictionary mapping layer indices to transcoder models
        edges: Dictionary of edges in the graph
        nodes: Dictionary of nodes in the graph
        do_bias: Whether to add bias error nodes
        
    Returns:
        Updated (edges, nodes) dictionaries
    """
    # First, deal with transcoder error
    for layer, transcoder in transcoders.items():
        # Get the normalized activations
        ln2_out = cache[f'normalized_{layer}_ln2']
        
        # For each token position
        for token in range(ln2_out.size(1)):
            # Get MLP output
            mlp_out, _ = model.blocks[layer].mlp(ln2_out[:, token:token+1], layer_idx=layer)
            mlp_out = mlp_out.squeeze(1)
            
            # Get transcoder reconstruction
            tc_out, _ = transcoder(ln2_out[:, token:token+1])
            tc_out = tc_out.squeeze(1)
            
            # Calculate reconstruction error
            error = mlp_out - tc_out
            error_norm = torch.norm(error).item()
            
            # Add error node for transcoder reconstruction error
            error_node_key = f"tc_error{layer}@{token}"
            
            # Add node
            nodes[error_node_key] = {
                'layer': layer,
                'sublayer': 'resid_mid',
                'type': ComponentType.TC_ERROR.value,
                'token': token
            }
            
            # Look for all nodes of this layer at this token
            for node_key, node_data in nodes.items():
                if (node_data['layer'] == layer and 
                    node_data['token'] == token and 
                    node_data['type'] == ComponentType.MLP.value):
                    
                    # Add edge from error node to this node
                    edge_key = (error_node_key, node_key)
                    edges[edge_key] = error_norm
            
    # Add bias error nodes if requested
    if do_bias:
        for layer in range(model.n_layers):
            # For MLP bias
            if hasattr(model.blocks[layer].mlp, 'b_out'):
                mlp_bias = model.blocks[layer].mlp.b_out
                bias_norm = torch.norm(mlp_bias).item()
                
                # Add bias error node
                bias_node_key = f"bias_error{layer}_mlp"
                
                # Add node
                nodes[bias_node_key] = {
                    'layer': layer,
                    'sublayer': 'resid_mid',
                    'type': ComponentType.BIAS_ERROR.value,
                    'token': None
                }
                
                # Look for all MLP nodes in this layer
                for node_key, node_data in nodes.items():
                    if (node_data['layer'] == layer and 
                        node_data['type'] == ComponentType.MLP.value):
                        
                        # Add edge from bias node to this node
                        edge_key = (bias_node_key, node_key)
                        edges[edge_key] = bias_norm
            
            # For attention bias
            if hasattr(model.blocks[layer].attention, 'b_O'):
                attn_bias = model.blocks[layer].attention.b_O
                bias_norm = torch.norm(attn_bias).item()
                
                # Add bias error node
                bias_node_key = f"bias_error{layer}_attn"
                
                # Add node
                nodes[bias_node_key] = {
                    'layer': layer,
                    'sublayer': 'resid_pre',
                    'type': ComponentType.BIAS_ERROR.value,
                    'token': None
                }
                
                # Look for all attention nodes in this layer
                for node_key, node_data in nodes.items():
                    if (node_data['layer'] == layer and 
                        node_data['type'] == ComponentType.ATTN.value):
                        
                        # Add edge from bias node to this node
                        edge_key = (bias_node_key, node_key)
                        edges[edge_key] = bias_norm
    
    return edges, nodes

def sum_over_tokens(edges, nodes):
    """
    Sum edge weights over tokens to get token-independent circuit diagram.
    
    Args:
        edges: Dictionary of edges in the graph
        nodes: Dictionary of nodes in the graph
        
    Returns:
        Tuple of (new_edges, new_nodes) with token-independent graph
    """
    new_edges = {}
    new_nodes = {}
    
    # Create mapping from node keys to token-independent keys
    node_key_map = {}
    for node_key, node_data in nodes.items():
        # Create token-independent key
        token_indep_key = node_key.split('@')[0] if '@' in node_key else node_key
        node_key_map[node_key] = token_indep_key
        
        # Add node if not already present
        if token_indep_key not in new_nodes:
            new_node_data = node_data.copy()
            new_node_data['token'] = None
            new_nodes[token_indep_key] = new_node_data
    
    # Sum edge weights for token-independent edges
    for (src, dst), weight in edges.items():
        src_new = node_key_map[src]
        dst_new = node_key_map[dst]
        
        edge_key = (src_new, dst_new)
        if edge_key in new_edges:
            new_edges[edge_key] += weight
        else:
            new_edges[edge_key] = weight
    
    return new_edges, new_nodes

def layer_to_float(feature):
    """Convert layer to float for positioning in the graph visualization."""
    layer = feature['layer'] if feature['layer'] is not None else 0
    
    sublayer_offset = 0.0
    if feature['sublayer'] == 'resid_pre':
        sublayer_offset = 0.0
    elif feature['sublayer'] == 'resid_mid':
        sublayer_offset = 0.3
    elif feature['sublayer'] == 'resid_post':
        sublayer_offset = 0.6
    
    return layer + sublayer_offset

def nodes_to_coords(nodes, y_jitter=0.3, y_mult=1.0):
    """
    Convert nodes to coordinates for graph visualization.
    
    Args:
        nodes: Dictionary of nodes in the graph
        y_jitter: Amount of vertical jitter to add
        y_mult: Multiplier for y-coordinates
        
    Returns:
        Dictionary mapping node keys to (x, y) coordinates
    """
    # Group nodes by layer and sublayer
    layer_counts = {}
    for node_key, node_data in nodes.items():
        layer_float = layer_to_float(node_data)
        if layer_float not in layer_counts:
            layer_counts[layer_float] = 0
        layer_counts[layer_float] += 1
    
    # Assign coordinates
    coords = {}
    layer_indices = {}
    for node_key, node_data in nodes.items():
        layer_float = layer_to_float(node_data)
        
        if layer_float not in layer_indices:
            layer_indices[layer_float] = 0
        
        # Calculate y position with jitter
        y_pos = layer_indices[layer_float] * y_jitter
        y_pos = (y_pos - (layer_counts[layer_float] - 1) * y_jitter / 2) * y_mult
        
        coords[node_key] = (layer_float, y_pos)
        layer_indices[layer_float] += 1
    
    return coords

def get_contribs_in_graph(edges, nodes):
    """Calculate total contribution magnitude in the graph."""
    total_contrib = 0.0
    for (src, dst), weight in edges.items():
        total_contrib += abs(weight)
    return total_contrib

def plot_graph(edges, nodes, y_mult=1.0, width=800, height=600, arrow_width_multiplier=3.0, 
              node_size_multiplier=20.0, show_edge_labels=False, title=None):
    """
    Plot the circuit graph.
    
    Args:
        edges: Dictionary of edges in the graph
        nodes: Dictionary of nodes in the graph
        y_mult: Multiplier for y-coordinates
        width: Width of the plot
        height: Height of the plot
        arrow_width_multiplier: Multiplier for arrow widths
        node_size_multiplier: Multiplier for node sizes
        show_edge_labels: Whether to show edge weight labels
        title: Optional title for the plot
        
    Returns:
        Matplotlib figure
    """
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_key in nodes.keys():
        G.add_node(node_key)
    
    # Add edges
    for (src, dst), weight in edges.items():
        G.add_edge(src, dst, weight=weight)
    
    # Get node coordinates
    coords = nodes_to_coords(nodes, y_mult=y_mult)
    
    # Set node positions
    pos = {node: (coords[node][0], coords[node][1]) for node in G.nodes()}
    
    # Create figure
    plt.figure(figsize=(width/100, height/100))
    
    # Node colors based on type
    node_colors = []
    for node in G.nodes():
        node_type = nodes[node]['type'] if nodes[node]['type'] is not None else 'none'
        
        if node_type == ComponentType.MLP.value:
            node_colors.append('lightblue')
        elif node_type == ComponentType.ATTN.value:
            node_colors.append('lightgreen')
        elif node_type == ComponentType.EMBED.value:
            node_colors.append('yellow')
        elif node_type == ComponentType.TC_ERROR.value:
            node_colors.append('red')
        elif node_type == ComponentType.PRUNE_ERROR.value:
            node_colors.append('orange')
        elif node_type == ComponentType.BIAS_ERROR.value:
            node_colors.append('purple')
        else:
            node_colors.append('gray')
    
    # Node sizes based on sum of incoming edge weights
    node_sizes = []
    for node in G.nodes():
        size = 0.0
        for u, v, data in G.in_edges(node, data=True):
            size += abs(data['weight'])
        node_sizes.append(size * node_size_multiplier + 50)  # Add base size
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Edge colors based on weight sign
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data['weight'] > 0:
            edge_colors.append('green')
        else:
            edge_colors.append('red')
    
    # Edge widths based on weight magnitude
    edge_widths = []
    max_weight = max(abs(data['weight']) for u, v, data in G.edges(data=True))
    for u, v, data in G.edges(data=True):
        width = abs(data['weight']) / max_weight * arrow_width_multiplier
        edge_widths.append(width)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                          arrowstyle='->', arrowsize=10, alpha=0.6)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Optionally draw edge labels
    if show_edge_labels:
        edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    # Set title if provided
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

class CircuitTracer:
    """Main class for circuit tracing."""
    
    def __init__(self, model, transcoders=None):
        """
        Initialize the circuit tracer.
        
        Args:
            model: Transformer model
            transcoders: Optional dictionary mapping layer indices to transcoder models
        """
        self.model = model
        self.transcoders = transcoders or {}
        self.cache = None
        
    def run_model(self, tokens):
        """
        Run the model on input tokens and cache activations.
        
        Args:
            tokens: Input token IDs
            
        Returns:
            Model output logits
        """
        logits, self.cache = self.model.run_with_cache(tokens)
        return logits
    
    def trace_circuit(self, feature_vector, num_iters=2, num_branches=5, ignore_bos=True, 
                     do_raw_attribution=False, filter=None):
        """
        Trace a circuit from a starting feature vector.
        
        Args:
            feature_vector: Feature vector to start from
            num_iters: Number of iterations to trace back
            num_branches: Number of branches to explore at each step
            ignore_bos: Whether to ignore the beginning-of-sequence token
            do_raw_attribution: Whether to use raw attribution
            filter: Optional filter for features
            
        Returns:
            List of all paths found
        """
        if self.cache is None:
            raise ValueError("Run the model first with run_model() method")
            
        return greedy_get_top_paths(
            self.model, self.transcoders, self.cache, feature_vector,
            num_iters=num_iters, num_branches=num_branches,
            ignore_bos=ignore_bos, do_raw_attribution=do_raw_attribution,
            filter=filter
        )
    
    def get_circuit_graph(self, all_paths, add_error_nodes=True, sum_over_tokens=False, 
                        use_jacobian=True, prune=True, threshold=0.01, max_edges=100):
        """
        Convert traced paths to a circuit graph.
        
        Args:
            all_paths: List of feature vectors (paths)
            add_error_nodes: Whether to add error nodes to the graph
            sum_over_tokens: Whether to sum edge weights over tokens
            use_jacobian: Whether to use Jacobian-based weight calculation
            prune: Whether to prune the graph
            threshold: Minimum absolute weight for an edge to be kept
            max_edges: Maximum number of edges to keep
            
        Returns:
            Tuple of (edges, nodes) for the graph, and optional pruning metrics
        """
        # Convert paths to graph
        result = paths_to_graph(
            all_paths, 
            model=self.model if use_jacobian else None,
            cache=self.cache if use_jacobian else None,
            use_jacobian=use_jacobian,
            prune=prune,
            threshold=threshold,
            max_edges=max_edges
        )
        
        if len(result) == 3:
            edges, nodes, pruning_metrics = result
        else:
            edges, nodes = result
            pruning_metrics = None
        
        # Add error nodes if requested
        if add_error_nodes:
            edges, nodes = add_error_nodes_to_graph(
                self.model, self.cache, self.transcoders, edges, nodes
            )
        
        # Sum over tokens if requested
        if sum_over_tokens:
            edges, nodes = sum_over_tokens(edges, nodes)
        
        if pruning_metrics:
            return edges, nodes, pruning_metrics
        else:
            return edges, nodes
    
    def visualize_circuit(self, edges, nodes, **kwargs):
        """
        Visualize the circuit graph.
        
        Args:
            edges: Dictionary of edges in the graph
            nodes: Dictionary of nodes in the graph
            **kwargs: Additional arguments to pass to plot_graph
            
        Returns:
            Matplotlib figure
        """
        return plot_graph(edges, nodes, **kwargs)
    
    def analyze_feature(self, vector, token_idx, layer, sublayer, num_iters=2, num_branches=5, 
                       add_error_nodes=True, sum_over_tokens=False, use_jacobian=True, 
                       prune=True, threshold=0.01, max_edges=100, **kwargs):
        """
        Analyze a feature vector and visualize its circuit.
        
        Args:
            vector: Feature vector to analyze
            token_idx: Token index
            layer: Layer index
            sublayer: Sublayer name
            num_iters: Number of iterations to trace back
            num_branches: Number of branches to explore at each step
            add_error_nodes: Whether to add error nodes to the graph
            sum_over_tokens: Whether to sum edge weights over tokens
            use_jacobian: Whether to use Jacobian-based weight calculation
            prune: Whether to prune the graph
            threshold: Minimum absolute weight for an edge to be kept
            max_edges: Maximum number of edges to keep
            **kwargs: Additional arguments to pass to plot_graph
            
        Returns:
            Tuple of (edges, nodes, figure) for the circuit and optional pruning metrics
        """
        # Create feature vector
        feature_vector = FeatureVector(
            component_path=[],
            vector=vector,
            layer=layer,
            sublayer=sublayer,
            token=token_idx
        )
        
        # Trace circuit
        all_paths = self.trace_circuit(
            feature_vector, 
            num_iters=num_iters, 
            num_branches=num_branches
        )
        
        # Get circuit graph
        result = self.get_circuit_graph(
            all_paths, 
            add_error_nodes=add_error_nodes, 
            sum_over_tokens=sum_over_tokens,
            use_jacobian=use_jacobian,
            prune=prune,
            threshold=threshold,
            max_edges=max_edges
        )
        
        if len(result) == 3:
            edges, nodes, pruning_metrics = result
        else:
            edges, nodes = result
            pruning_metrics = None
        
        # Visualize circuit
        fig = self.visualize_circuit(edges, nodes, **kwargs)
        
        if pruning_metrics:
            return edges, nodes, fig, pruning_metrics
        else:
            return edges, nodes, fig

@torch.no_grad()
def calculate_jacobian_weights(model, cache, src_feature, dst_feature):
    """
    Calculate connection weights between features using Jacobian-based method.
    
    Args:
        model: Transformer model
        cache: Activation cache
        src_feature: Source feature vector (earlier in the network)
        dst_feature: Destination feature vector (later in the network)
        
    Returns:
        Connection weight based on Jacobian calculation
    """
    # Extract key information
    src_layer = src_feature.layer
    src_token = src_feature.token
    dst_layer = dst_feature.layer
    dst_token = dst_feature.token
    
    # Only process if the destination is later than the source
    if dst_layer < src_layer or (dst_layer == src_layer and 
                               sublayer_order(dst_feature.sublayer) <= sublayer_order(src_feature.sublayer)):
        return 0.0
    
    # Get source and destination vectors
    src_vector = src_feature.vector
    dst_vector = dst_feature.vector
    
    # For direct connections within the same layer
    if dst_layer == src_layer:
        # For MLP to residual connections
        if src_feature.sublayer == 'resid_mid' and dst_feature.sublayer == 'resid_post':
            # Direct connection with weight 1.0 (residual connection)
            dot_product = torch.dot(src_vector, dst_vector).item()
            src_norm = torch.norm(src_vector).item()
            dst_norm = torch.norm(dst_vector).item()
            
            if src_norm > 0 and dst_norm > 0:
                return dot_product / (src_norm * dst_norm)
            return 0.0
    
    # For connections across layers through attention
    if src_feature.sublayer in ['resid_pre', 'resid_mid'] and dst_feature.sublayer == 'resid_pre':
        # Calculate weight through attention mechanism
        if hasattr(model.blocks[dst_layer], 'attention'):
            attn = model.blocks[dst_layer].attention
            
            # Get attention pattern
            pattern = cache[f'pattern_{dst_layer}'][0, :, dst_token]  # [head, seq_len]
            
            # Calculate Jacobian influence through attention
            weight = 0.0
            for head_idx in range(model.n_heads):
                head_pattern = pattern[head_idx, src_token].item()
                
                # Only include if there's significant attention
                if head_pattern > 0.01:
                    # Calculate influence through this attention head
                    q_vector = cache[f'q_{dst_layer}'][0, dst_token, head_idx]  # [d_head]
                    k_vector = cache[f'k_{dst_layer}'][0, src_token, head_idx]  # [d_head]
                    
                    # Approximate Jacobian element for attention
                    scale_factor = 1.0 / torch.sqrt(torch.tensor(attn.d_head, dtype=torch.float))
                    attn_jacobian = scale_factor * head_pattern * (1 - head_pattern) * torch.outer(q_vector, k_vector)
                    
                    # Project source and destination vectors through this Jacobian
                    influence = torch.dot(dst_vector, attn_jacobian @ src_vector).item()
                    weight += influence
            
            return weight
    
    # For connections through MLP
    if src_feature.sublayer in ['resid_mid', 'resid_post'] and dst_layer > src_layer:
        # Get the MLP input at the destination layer
        ln2_out = cache[f'normalized_{dst_layer}_ln2'][0, dst_token]
        
        # If we have a transcoder for this layer, use it to calculate the Jacobian
        if hasattr(model, 'transcoders') and dst_layer in model.transcoders:
            transcoder = model.transcoders[dst_layer]
            
            # Calculate Jacobian for the transcoder
            with torch.enable_grad():
                ln2_out_tensor = ln2_out.clone().detach().requires_grad_(True)
                encoded = transcoder.encode(ln2_out_tensor.unsqueeze(0))
                encoded_activated = F.relu(encoded)
                decoded = transcoder.decode(encoded_activated)
                
                # Calculate Jacobian of MLP output with respect to MLP input
                jac = torch.autograd.functional.jacobian(
                    lambda x: transcoder.decode(F.relu(transcoder.encode(x.unsqueeze(0)))).squeeze(0),
                    ln2_out
                )
                
                # Project source and destination vectors through this Jacobian
                influence = torch.dot(dst_vector, jac @ src_vector).item()
                return influence
    
    # Default: simple cosine similarity as fallback
    dot_product = torch.dot(src_vector, dst_vector).item()
    src_norm = torch.norm(src_vector).item()
    dst_norm = torch.norm(dst_vector).item()
    
    if src_norm > 0 and dst_norm > 0:
        return dot_product / (src_norm * dst_norm)
    return 0.0

def sublayer_order(sublayer):
    """Helper function to determine the order of sublayers within a layer."""
    if sublayer == 'resid_pre':
        return 0
    elif sublayer == 'resid_mid':
        return 1
    elif sublayer == 'resid_post':
        return 2
    else:
        return -1

@torch.no_grad()
def prune_circuit_graph(edges, nodes, threshold=0.01, max_edges=100):
    """
    Prune the circuit graph to limit complexity.
    
    Args:
        edges: Dictionary of edges in the graph
        nodes: Dictionary of nodes in the graph
        threshold: Minimum absolute weight for an edge to be kept
        max_edges: Maximum number of edges to keep
        
    Returns:
        Pruned (edges, nodes) dictionaries and pruning metrics
    """
    # Copy edges and sort by absolute weight
    edges_items = list(edges.items())
    edges_items.sort(key=lambda x: -abs(x[1]))
    
    # Keep track of pruned weight
    total_weight = sum(abs(w) for _, w in edges_items)
    kept_weight = 0.0
    pruned_weight = 0.0
    
    # Apply threshold and max_edges constraint
    kept_edges = {}
    for i, ((src, dst), weight) in enumerate(edges_items):
        if abs(weight) >= threshold and i < max_edges:
            kept_edges[(src, dst)] = weight
            kept_weight += abs(weight)
        else:
            pruned_weight += abs(weight)
    
    # Only keep nodes that are connected by the kept edges
    kept_nodes = set()
    for src, dst in kept_edges.keys():
        kept_nodes.add(src)
        kept_nodes.add(dst)
    
    # Filter nodes dictionary
    kept_nodes_dict = {node: data for node, data in nodes.items() if node in kept_nodes}
    
    # Add prune error node if significant weight was pruned
    if pruned_weight > 0.001 * total_weight:
        prune_error_key = "prune_error"
        kept_nodes_dict[prune_error_key] = {
            'layer': max([nodes[n]['layer'] for n in kept_nodes if nodes[n]['layer'] is not None], default=0),
            'sublayer': 'resid_post',
            'type': ComponentType.PRUNE_ERROR.value,
            'token': None
        }
        
        # Add edge from prune error to output node(s)
        output_nodes = [node for node in kept_nodes if 
                        nodes[node]['layer'] == max([nodes[n]['layer'] for n in kept_nodes if nodes[n]['layer'] is not None], default=0)]
        
        for output_node in output_nodes:
            kept_edges[(prune_error_key, output_node)] = pruned_weight
    
    # Calculate pruning metrics
    metrics = {
        'total_weight': total_weight,
        'kept_weight': kept_weight,
        'pruned_weight': pruned_weight,
        'kept_edges': len(kept_edges),
        'original_edges': len(edges),
        'kept_nodes': len(kept_nodes_dict),
        'original_nodes': len(nodes),
        'pruned_weight_ratio': pruned_weight / total_weight if total_weight > 0 else 0.0
    }
    
    return kept_edges, kept_nodes_dict, metrics 