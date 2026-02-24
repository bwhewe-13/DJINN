###############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).
#
# LLNL-CODE-754815
#
# All rights reserved.
#
# This file is part of DJINN.
#
# For details, see github.com/LLNL/djinn.
#
# For details about use and distribution, please read DJINN/LICENSE .
###############################################################################


import numpy as np


def xavier_init(nin, nout):
    """Return a Xavier-initialized scalar weight.

    Parameters
    ----------
    nin : int
        Input dimension of the layer.
    nout : int
        Output dimension of the layer.

    Returns
    -------
    float
        Randomly initialized weight value.
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(3.0 / (nin + nout)))


def get_dimensions(X, Y, regression):
    """Extract input and output dimensions from training data.

    Parameters
    ----------
    X : numpy.ndarray
        Input features of shape ``(n_samples, n_features)``.
    Y : numpy.ndarray
        Target values for regression or class labels for classification.
    regression : bool
        Whether the task is regression.

    Returns
    -------
    tuple[int, int]
        Input and output dimensions ``(nin, nout)``.
    """
    nin = X.shape[1]

    if regression:
        if Y.size > Y.shape[0]:
            nout = Y.shape[1]
        else:
            nout = 1
    else:
        nout = len(np.unique(Y))

    return nin, nout


def extract_tree_structure(tree_):
    """Extract structural arrays and metadata from a fitted sklearn tree.

    Parameters
    ----------
    tree_ : sklearn.tree._tree.Tree
        Internal tree object from a fitted sklearn estimator.

    Returns
    -------
    dict
        Dictionary containing split features, node depths, child indices,
        thresholds, leaf flags, and node count.
    """
    features = tree_.feature
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    threshold = tree_.threshold

    # Calculate node depths
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # root node and parent depth

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return {
        "features": features,
        "node_depth": node_depth,
        "children_left": children_left,
        "children_right": children_right,
        "threshold": threshold,
        "is_leaves": is_leaves,
        "n_nodes": n_nodes,
    }


def build_node_dict(tree_structure):
    """Build a node-indexed dictionary representation of tree structure.

    Parameters
    ----------
    tree_structure : dict
        Output of :func:`extract_tree_structure`.

    Returns
    -------
    dict
        Mapping from node index to a dictionary with depth, feature, and child
        split information.
    """
    features = tree_structure["features"]
    node_depth = tree_structure["node_depth"]
    children_left = tree_structure["children_left"]
    children_right = tree_structure["children_right"]

    node = {}
    for i in range(len(features)):
        node[i] = {
            "depth": node_depth[i],
            "feature": features[i] if features[i] >= 0 else -2,
            "child_left": features[children_left[i]],
            "child_right": features[children_right[i]],
        }

    return node


def compute_layer_metadata(node_depth, features, nin):
    """Compute depth-wise tree metadata used by DJINN mapping.

    Parameters
    ----------
    node_depth : numpy.ndarray
        Depth value for each node.
    features : numpy.ndarray
        Feature index per node, negative values for leaves.
    nin : int
        Number of input features.

    Returns
    -------
    dict
        Metadata dictionary with ``num_layers``, ``nodes_per_depth``,
        ``leaves_per_depth``, and ``max_depth_feature``.
    """
    num_layers = len(np.unique(node_depth))
    nodes_per_depth = np.zeros(num_layers)
    leaves_per_depth = np.zeros(num_layers)

    for i in range(num_layers):
        ind = np.where(node_depth == i)[0]
        nodes_per_depth[i] = len(np.where(features[ind] >= 0)[0])
        leaves_per_depth[i] = len(np.where(features[ind] < 0)[0])

    # Max depth at which each feature appears
    max_depth_feature = np.zeros(nin)
    for i in range(nin):
        ind = np.where(features == i)[0]
        if len(ind) > 0:
            max_depth_feature[i] = np.max(node_depth[ind])

    return {
        "num_layers": num_layers,
        "nodes_per_depth": nodes_per_depth,
        "leaves_per_depth": leaves_per_depth,
        "max_depth_feature": max_depth_feature,
    }


def build_djinn_architecture(num_layers, nodes_per_depth, nin, nout):
    """Construct the DJINN layer-width vector.

    Parameters
    ----------
    num_layers : int
        Number of network layers.
    nodes_per_depth : numpy.ndarray
        Number of split nodes observed at each tree depth.
    nin : int
        Input feature dimension.
    nout : int
        Output dimension.

    Returns
    -------
    numpy.ndarray
        Network architecture array (neurons per layer).
    """
    djinn_arch = np.zeros(num_layers, dtype=int)
    djinn_arch[0] = nin

    for i in range(1, num_layers):
        djinn_arch[i] = djinn_arch[i - 1] + int(nodes_per_depth[i])

    djinn_arch[-1] = nout
    return djinn_arch


def initialize_weight_arrays(djinn_arch, num_layers):
    """Initialize DJINN weight matrices and new-neuron index lists.

    Parameters
    ----------
    djinn_arch : numpy.ndarray
        Network architecture array.
    num_layers : int
        Number of layers.

    Returns
    -------
    tuple[dict, list]
        Tuple ``(djinn_weights, neurons_layer)`` with empty layer matrices and
        indices of neurons newly introduced at each layer.
    """
    djinn_weights = {}
    for i in range(num_layers - 1):
        djinn_weights[i] = np.zeros((djinn_arch[i + 1], djinn_arch[i]))

    # Indices for new neurons in each layer
    neurons_layer = []
    for i in range(num_layers - 1):
        neurons_layer.append(np.arange(djinn_arch[i], djinn_arch[i + 1]))

    return djinn_weights, neurons_layer


def process_split_node(
    djinn_weights, node, nodes, i, k, kk, neurons_layer, nn_in, nn_out, num_layers
):
    """Process one split node and update mapped weight connections.

    Parameters
    ----------
    djinn_weights : dict
        Layer weight matrices to modify in-place.
    node : dict
        Node metadata dictionary.
    nodes : int
        Current node index.
    i : int
        Current network layer index.
    k : int
        Outgoing neuron counter.
    kk : int
        Incoming neuron counter.
    neurons_layer : list
        New neuron indices for each layer.
    nn_in : int
        Layer output width for initialization scaling.
    nn_out : int
        Layer input width for initialization scaling.
    num_layers : int
        Total number of network layers.

    Returns
    -------
    tuple[int, int]
        Updated counters ``(k, kk)``.
    """
    feature = node[nodes]["feature"]
    left = node[nodes]["child_left"]
    right = node[nodes]["child_right"]

    # Handle leaf at first split
    if nodes == 0 and (left < 0 or right < 0):
        for j in range(i, num_layers - 2):
            djinn_weights[j][feature, feature] = 1.0
        djinn_weights[num_layers - 2][:, feature] = 1.0

    # Handle left child
    if left >= 0:
        if i == 0:
            djinn_weights[i][neurons_layer[i][k], feature] = xavier_init(nn_in, nn_out)
        else:
            djinn_weights[i][neurons_layer[i][k], neurons_layer[i - 1][kk]] = (
                xavier_init(nn_in, nn_out)
            )
        djinn_weights[i][neurons_layer[i][k], left] = xavier_init(nn_in, nn_out)
        k += 1
        if kk >= len(neurons_layer[i - 1]):
            kk = 0

    # Handle left leaf
    if left < 0 and nodes != 0:
        lind = neurons_layer[i - 1][kk]
        for j in range(i, num_layers - 2):
            djinn_weights[j][lind, lind] = 1.0
        djinn_weights[num_layers - 2][:, lind] = 1.0

    # Handle right child
    if right >= 0:
        if i == 0:
            djinn_weights[i][neurons_layer[i][k], feature] = xavier_init(nn_in, nn_out)
        else:
            djinn_weights[i][neurons_layer[i][k], neurons_layer[i - 1][kk]] = (
                xavier_init(nn_in, nn_out)
            )
        djinn_weights[i][neurons_layer[i][k], right] = xavier_init(nn_in, nn_out)
        k += 1
        if kk >= len(neurons_layer[i - 1]):
            kk = 0

    # Handle right leaf
    if right < 0 and nodes != 0:
        lind = neurons_layer[i - 1][kk]
        for j in range(i, num_layers - 2):
            djinn_weights[j][lind, lind] = 1.0
        djinn_weights[num_layers - 2][:, lind] = 1.0

    kk += 1
    return k, kk


def fill_weight_connections(
    djinn_weights, node, num_layers, nin, neurons_layer, max_depth_feature
):
    """Populate all DJINN layer connections from tree metadata.

    Parameters
    ----------
    djinn_weights : dict
        Layer weight matrices to modify in-place.
    node : dict
        Node metadata dictionary.
    num_layers : int
        Number of network layers.
    nin : int
        Number of input features.
    neurons_layer : list
        New neuron indices for each layer.
    max_depth_feature : numpy.ndarray
        Maximum depth where each feature appears.
    """
    # add_diagonal_connections(djinn_weights, max_depth_feature, nin, num_layers)

    for n_layer in range(num_layers - 1):
        nn_in = djinn_weights[n_layer].shape[0]
        nn_out = djinn_weights[n_layer].shape[1]

        # Add diagonal pass-through connections for active features.
        for ff in range(nin):
            if n_layer < max_depth_feature[ff] - 1:
                djinn_weights[n_layer][ff, ff] = 1.0

        djinn_weights = off_diagonal_weights(
            n_layer, nn_in, nn_out, num_layers, node, djinn_weights, neurons_layer
        )


def connect_split_child(
    n_layer, nn_in, nn_out, k, kk, feature, child_feature, djinn_weights, neurons_layer
):
    """Connect a split child branch into the current NN layer.

    Parameters
    ----------
    n_layer : int
        Current layer index.
    nn_in : int
        Output width of the layer matrix.
    nn_out : int
        Input width of the layer matrix.
    k : int
        Index over newly created neurons at this layer.
    kk : int
        Index over active parent neurons from previous layer.
    feature : int
        Parent split feature index.
    child_feature : int
        Child split feature index (non-negative for split nodes).
    djinn_weights : dict
        Layer weight matrices.
    neurons_layer : list[numpy.ndarray]
        New neuron indices per layer.

    Returns
    -------
    tuple
        Updated ``(djinn_weights, k, kk)``.
    """
    target_neuron = neurons_layer[n_layer][k]
    if n_layer == 0:
        djinn_weights[n_layer][target_neuron, feature] = xavier_init(nn_in, nn_out)
    else:
        source_neuron = neurons_layer[n_layer - 1][kk]
        djinn_weights[n_layer][target_neuron, source_neuron] = xavier_init(
            nn_in, nn_out
        )
    djinn_weights[n_layer][target_neuron, child_feature] = xavier_init(nn_in, nn_out)
    k += 1
    if kk >= len(neurons_layer[n_layer - 1]):
        kk = 0
    return djinn_weights, k, kk


def connect_leaf_path(n_layer, num_layers, kk, neurons_layer, djinn_weights):
    """Propagate a leaf decision path directly through to output.

    Parameters
    ----------
    n_layer : int
        Layer where the leaf was encountered.
    num_layers : int
        Total number of layers.
    kk : int
        Parent neuron index for the leaf path.
    neurons_layer : list[numpy.ndarray]
        New neuron indices per layer.
    djinn_weights : dict
        Layer weight matrices.

    Returns
    -------
    dict
        Updated layer weight matrices.
    """
    leaf_input = neurons_layer[n_layer - 1][kk]
    for j in range(n_layer, num_layers - 2):
        djinn_weights[j][leaf_input, leaf_input] = 1.0
    djinn_weights[num_layers - 2][:, leaf_input] = 1.0
    return djinn_weights


def off_diagonal_weights(
    n_layer, nn_in, nn_out, num_layers, node, djinn_weights, neurons_layer
):
    """Populate off-diagonal connections that encode tree branching.

    Parameters
    ----------
    n_layer : int
        Current layer index.
    nn_in : int
        Layer output width.
    nn_out : int
        Layer input width.
    num_layers : int
        Number of network layers.
    node : dict
        Per-node tree metadata.
    djinn_weights : dict
        Layer weight matrices.
    neurons_layer : list[numpy.ndarray]
        New neuron indices per layer.

    Returns
    -------
    dict
        Updated layer weight matrices.
    """
    k = 0
    kk = 0
    # k tracks current layer's new-neuron cursor; kk tracks incoming neurons.
    for node_id, node_data in node.items():
        if node_data["depth"] != n_layer:
            continue

        feature = node_data["feature"]
        # if node is a leaf
        if feature < 0:
            continue

        left = node_data["child_left"]
        right = node_data["child_right"]

        if node_id == 0 and (left < 0 or right < 0):
            # Root immediately reaches a leaf: carry that feature to output.
            for jj in range(n_layer, num_layers - 2):
                djinn_weights[jj][feature, feature] = 1.0
            djinn_weights[num_layers - 2][:, feature] = 1.0

        if left >= 0:
            djinn_weights, k, kk = connect_split_child(
                n_layer,
                nn_in,
                nn_out,
                k,
                kk,
                feature,
                left,
                djinn_weights,
                neurons_layer,
            )
        elif node_id != 0:
            djinn_weights = connect_leaf_path(
                n_layer, num_layers, kk, neurons_layer, djinn_weights
            )
        if right >= 0:
            djinn_weights, k, kk = connect_split_child(
                n_layer,
                nn_in,
                nn_out,
                k,
                kk,
                feature,
                right,
                djinn_weights,
                neurons_layer,
            )
        elif node_id != 0:
            djinn_weights = connect_leaf_path(
                n_layer, num_layers, kk, neurons_layer, djinn_weights
            )
        kk += 1
    return djinn_weights


def connect_output_layer(djinn_weights, neurons_layer, num_layers):
    """Connect active hidden neurons to the output layer.

    Parameters
    ----------
    djinn_weights : dict
        Layer weight matrices to modify in-place.
    neurons_layer : list
        New neuron indices for each layer.
    num_layers : int
        Number of network layers.
    """
    m = len(neurons_layer[-2])
    ind = np.where(abs(djinn_weights[num_layers - 3][:, -m:]) > 0)[0]

    nn_in = djinn_weights[num_layers - 2].shape[0]
    nn_out = djinn_weights[num_layers - 2].shape[1]

    for inds in range(len(djinn_weights[num_layers - 2][:, ind])):
        djinn_weights[num_layers - 2][inds, ind] = xavier_init(nn_in, nn_out)


def map_single_tree_to_network(tree, nin, nout):
    """Map one fitted tree into DJINN architecture and initial weights.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        Internal tree object from a fitted estimator.
    nin : int
        Input dimension.
    nout : int
        Output dimension.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Tuple ``(djinn_arch, djinn_weights)``.
    """
    # Extract tree structure
    tree_structure = extract_tree_structure(tree)
    node = build_node_dict(tree_structure)

    # Compute layer metadata
    metadata = compute_layer_metadata(
        tree_structure["node_depth"], tree_structure["features"], nin
    )

    # Build architecture
    djinn_arch = build_djinn_architecture(
        metadata["num_layers"], metadata["nodes_per_depth"], nin, nout
    )

    # Initialize weights
    djinn_weights, neurons_layer = initialize_weight_arrays(
        djinn_arch, metadata["num_layers"]
    )

    # Fill in connections
    fill_weight_connections(
        djinn_weights,
        node,
        metadata["num_layers"],
        nin,
        neurons_layer,
        metadata["max_depth_feature"],
    )

    # Connect output layer
    connect_output_layer(djinn_weights, neurons_layer, metadata["num_layers"])

    return djinn_arch, djinn_weights


def tree_to_nn_weights(regression, X, Y, num_trees, rfr, seed=False):
    """Map every tree in an ensemble to DJINN architecture and weights.

    Parameters
    ----------
    regression : bool
        Whether the task is regression.
    X : numpy.ndarray
        Input features.
    Y : numpy.ndarray
        Target values.
    num_trees : int
        Number of trees to map.
    rfr : object
        Fitted sklearn forest-like estimator containing ``estimators_``.
    seed : int or bool, default=False
        Random seed used for deterministic initialization.

    Returns
    -------
    dict
        Dictionary with global dimensions and per-tree network shape, weights,
        and bias placeholders.
    """
    # Set random seed
    if seed:
        np.random.seed(seed)

    # Get dimensions
    nin, nout = get_dimensions(X, Y, regression)

    # Initialize output structure
    tree_to_network = {
        "n_in": nin,
        "n_out": nout,
        "network_shape": {},
        "weights": {},
        "biases": {},
    }

    # Map each tree in random forest
    for tt in range(num_trees):
        tree = rfr.estimators_[tt].tree_
        djinn_arch, djinn_weights = map_single_tree_to_network(tree, nin, nout)

        tree_to_network["network_shape"][f"tree_{tt}"] = djinn_arch
        tree_to_network["weights"][f"tree_{tt}"] = djinn_weights
        tree_to_network["biases"][f"tree_{tt}"] = []

    return tree_to_network
