import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import djinn.random_forest as rfns


def make_simple_tree():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1, 2, 2])
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    return clf, X, y


def test_xavier_init_reproducible():
    np.random.seed(0)
    vals = np.array([rfns.xavier_init(3, 5) for _ in range(1000)])
    expected_std = np.sqrt(3.0 / (3 + 5))
    assert abs(vals.mean()) < expected_std * 0.2
    assert abs(vals.std() - expected_std) < expected_std * 0.3


def test_get_dimensions_regression_and_classification():
    X = np.random.rand(10, 4)

    Y1 = np.random.rand(10)
    nin, nout = rfns.get_dimensions(X, Y1, regression=True)
    assert nin == 4
    assert nout == 1

    Y2 = np.random.rand(10, 3)
    nin, nout = rfns.get_dimensions(X, Y2, regression=True)
    assert nin == 4
    assert nout == 3

    Yc = np.array([0, 1, 1, 0, 2, 2, 1, 0, 2, 1])
    nin, nout = rfns.get_dimensions(X, Yc, regression=False)
    assert nin == 4
    assert nout == len(np.unique(Yc))


def test_extract_tree_structure_and_build_node_dict():
    clf, X, _ = make_simple_tree()
    tree = clf.tree_

    ts = rfns.extract_tree_structure(tree)
    assert ts["n_nodes"] == tree.node_count
    assert ts["features"].shape[0] == ts["n_nodes"]

    children_eq = np.sum(tree.children_left == tree.children_right)
    assert np.sum(ts["is_leaves"]) == children_eq

    node = rfns.build_node_dict(ts)
    assert len(node) == ts["n_nodes"]
    assert node[0]["depth"] == ts["node_depth"][0]

    metadata = rfns.compute_layer_metadata(ts["node_depth"], ts["features"], X.shape[1])
    assert metadata["num_layers"] == len(np.unique(ts["node_depth"]))
    assert (
        int(metadata["nodes_per_depth"].sum() + metadata["leaves_per_depth"].sum())
        == ts["n_nodes"]
    )


def test_build_architecture_and_initialize_arrays():
    num_layers = 4
    nodes_per_depth = np.array([1, 2, 1, 0])
    nin, nout = 3, 2

    arch = rfns.build_djinn_architecture(num_layers, nodes_per_depth, nin, nout)
    assert len(arch) == num_layers
    assert arch[0] == nin
    assert arch[-1] == nout

    weights, new_n_ind = rfns.initialize_weight_arrays(arch, num_layers)
    for i in range(num_layers - 1):
        assert weights[i].shape == (arch[i + 1], arch[i])
        assert isinstance(new_n_ind[i], np.ndarray)


def test_fill_weight_connections_adds_diagonal_pass_through():
    nin = 3
    num_layers = 3
    djinn_weights = {0: np.zeros((4, 4)), 1: np.zeros((4, 4))}
    max_depth_feature = np.array([2, 1, 3])
    neurons_layer = [np.array([3]), np.array([])]
    node = {}

    rfns.fill_weight_connections(
        djinn_weights,
        node,
        num_layers,
        nin,
        neurons_layer,
        max_depth_feature,
    )

    assert djinn_weights[0][0, 0] == 1.0
    assert djinn_weights[0][1, 1] == 0.0
    assert djinn_weights[0][2, 2] == 1.0
    assert djinn_weights[1][0, 0] == 0.0
    assert djinn_weights[1][2, 2] == 1.0


def test_process_split_node():
    num_layers = 3
    djinn_weights = {0: np.zeros((4, 2)), 1: np.zeros((3, 4))}
    new_n_ind = [np.array([2, 3]), np.array([])]
    node = {0: {"feature": 1, "child_left": 0, "child_right": 1}}

    k, kk = rfns.process_split_node(
        djinn_weights,
        node,
        0,
        0,
        0,
        0,
        new_n_ind,
        djinn_weights[0].shape[0],
        djinn_weights[0].shape[1],
        num_layers,
    )

    assert k == 2
    assert kk == 1
    assert djinn_weights[0][new_n_ind[0][0], 1] != 0.0
    assert djinn_weights[0][new_n_ind[0][0], 0] != 0.0
    assert djinn_weights[0][new_n_ind[0][1], 1] != 0.0


def test_connect_split_child_layer0_and_layer1():
    # layer 0 case
    djinn_weights = {0: np.zeros((4, 2)), 1: np.zeros((3, 4))}
    neurons_layer = [np.array([2, 3]), np.array([])]

    djinn_weights, k, kk = rfns.connect_split_child(
        0,
        djinn_weights[0].shape[0],
        djinn_weights[0].shape[1],
        0,
        0,
        1,
        0,
        djinn_weights,
        neurons_layer,
    )

    assert k == 1
    assert kk == 0
    assert djinn_weights[0][2, 1] != 0.0
    assert djinn_weights[0][2, 0] != 0.0

    # layer 1 case
    djinn_weights = {0: np.zeros((4, 2)), 1: np.zeros((5, 4))}
    neurons_layer = [np.array([2, 3]), np.array([4])]

    djinn_weights, k, kk = rfns.connect_split_child(
        1,
        djinn_weights[1].shape[0],
        djinn_weights[1].shape[1],
        0,
        0,
        1,
        2,
        djinn_weights,
        neurons_layer,
    )

    assert k == 1
    assert kk == 0
    assert djinn_weights[1][4, 2] != 0.0


def test_connect_leaf_path_sets_identity_and_output_links():
    num_layers = 4
    djinn_weights = {
        0: np.zeros((4, 2)),
        1: np.zeros((4, 4)),
        2: np.zeros((2, 4)),
    }
    neurons_layer = [np.array([2, 3]), np.array([4]), np.array([])]

    rfns.connect_leaf_path(1, num_layers, 0, neurons_layer, djinn_weights)

    assert djinn_weights[1][2, 2] == 1.0
    assert np.all(djinn_weights[2][:, 2] == 1.0)


def test_off_diagonal_weights_populates_branch_connections():
    num_layers = 3
    djinn_weights = {0: np.zeros((4, 2)), 1: np.zeros((3, 4))}
    neurons_layer = [np.array([2, 3]), np.array([])]
    node = {0: {"depth": 0, "feature": 1, "child_left": 0, "child_right": 1}}

    out = rfns.off_diagonal_weights(
        0,
        djinn_weights[0].shape[0],
        djinn_weights[0].shape[1],
        num_layers,
        node,
        djinn_weights,
        neurons_layer,
    )

    assert out is djinn_weights
    assert np.any(djinn_weights[0] != 0)


def test_fill_weight_connections_and_connect_output_layer():
    num_layers = 3
    nin = 2
    djinn_arch = np.array([2, 4, 3])

    djinn_weights, new_n_ind = rfns.initialize_weight_arrays(djinn_arch, num_layers)
    node = {0: {"depth": 0, "feature": 1, "child_left": 0, "child_right": 1}}
    max_depth_feature = np.array([1, 2])

    rfns.fill_weight_connections(
        djinn_weights, node, num_layers, nin, new_n_ind, max_depth_feature
    )
    assert np.any(djinn_weights[0] != 0)

    m = len(new_n_ind[-2])
    if m == 0:
        return

    djinn_weights[0][:, -m:] = 0.0
    djinn_weights[0][0, -m] = 1.0
    djinn_weights[0][2, -m] = 1.0

    rfns.connect_output_layer(djinn_weights, new_n_ind, num_layers)

    ind = np.where(abs(djinn_weights[num_layers - 3][:, -m:]) > 0)[0]
    assert ind.size > 0
    assert np.any(djinn_weights[num_layers - 2][:, ind] != 0)


def test_map_single_tree_to_network():
    clf, X, y = make_simple_tree()
    tree = clf.tree_

    arch, weights = rfns.map_single_tree_to_network(tree, X.shape[1], len(np.unique(y)))

    assert arch[0] == X.shape[1]
    assert arch[-1] == len(np.unique(y))
    assert isinstance(weights, dict)
    assert any(np.any(w != 0) for w in weights.values())


def test_tree_to_nn_weights_end_to_end():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
            [2, 3],
            [3, 3],
            [3, 4],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2, 2, 1])

    n_trees = 2
    rfr = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=3,
        bootstrap=False,
        random_state=2,
    )
    rfr.fit(X, y)

    out = rfns.tree_to_nn_weights(False, X, y, n_trees, rfr, seed=2)

    assert out["n_in"] == X.shape[1]
    assert out["n_out"] == len(np.unique(y))
    assert len(out["network_shape"]) == n_trees
    assert len(out["weights"]) == n_trees
    assert len(out["biases"]) == n_trees
    assert all(out["biases"][f"tree_{i}"] == [] for i in range(n_trees))
