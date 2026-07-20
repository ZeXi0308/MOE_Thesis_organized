import numpy as np

from run_route_fidelity_p0 import (
    RouteMatrix,
    _degree_preserving_shuffle,
    lower_metrics,
)


def test_degree_shuffle_preserves_simple_graph_degrees() -> None:
    routes = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
            [1, 3],
        ],
        dtype=np.int16,
    )
    shuffled = _degree_preserving_shuffle(routes, np.random.default_rng(7))
    assert np.array_equal(
        np.bincount(routes.reshape(-1), minlength=4),
        np.bincount(shuffled.reshape(-1), minlength=4),
    )
    assert all(len(set(row.tolist())) == 2 for row in shuffled)


def test_rank_deduplicated_lowering() -> None:
    route_matrix = RouteMatrix(
        experts=np.array([[0, 1], [2, 3]], dtype=np.int16),
        sample=np.array([0, 0], dtype=np.int32),
        layer=np.array([0, 0], dtype=np.int16),
        position=np.array([0, 1], dtype=np.int32),
        num_experts=4,
        top_k=2,
    )
    mapping = np.array([0, 0, 0, 1], dtype=np.int16)
    metrics, receiver_max = lower_metrics(
        route_matrix.experts, route_matrix, mapping, ep_size=2, batch_tokens=2
    )
    assert metrics["physical_records"] == 3.0
    assert metrics["mean_fanout"] == 1.5
    assert receiver_max.tolist() == [2]
