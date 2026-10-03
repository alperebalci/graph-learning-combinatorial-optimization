import numpy as np

from gnn_solver.dataset import build_exact_dataset
from gnn_solver.decoder import greedy_edge_decoder
from gnn_solver.instance import random_euclidean_instance
from gnn_solver.train import (
    TrainConfig,
    fit_edge_gnn,
    fit_edge_model,
    predict_edge_scores,
)


def test_neural_training_and_decoding_smoke():
    examples = build_exact_dataset([0, 1, 2, 3], n_nodes=6)
    model, history = fit_edge_gnn(
        examples,
        TrainConfig(hidden_dim=16, layers=1, epochs=2, seed=0),
    )
    assert len(history) == 2
    assert np.isfinite(history[-1])

    scores = predict_edge_scores(model, examples[0])
    instance = random_euclidean_instance(0, n_nodes=6)
    tour = greedy_edge_decoder(instance, scores)
    assert set(tour) == set(range(6))


def test_graphsage_edge_baseline_smoke():
    examples = build_exact_dataset([0, 1, 2, 3], n_nodes=6)
    model, history = fit_edge_model(
        examples,
        TrainConfig(
            hidden_dim=16,
            layers=2,
            epochs=2,
            seed=0,
            architecture="graphsage",
            k_neighbors=3,
        ),
    )

    assert len(history) == 2
    assert np.isfinite(history[-1])

    scores = predict_edge_scores(model, examples[0])
    assert scores.shape == (6, 6)
    assert np.allclose(scores, scores.T, atol=1e-6)

    instance = random_euclidean_instance(0, n_nodes=6)
    tour = greedy_edge_decoder(instance, scores)
    assert set(tour) == set(range(6))


def _assert_node_only_baseline_smoke(architecture: str):
    examples = build_exact_dataset([0, 1, 2, 3], n_nodes=6)
    model, history = fit_edge_model(
        examples,
        TrainConfig(
            hidden_dim=16,
            layers=2,
            epochs=2,
            seed=0,
            architecture=architecture,
            k_neighbors=3,
        ),
    )

    assert len(history) == 2
    assert np.isfinite(history[-1])

    scores = predict_edge_scores(model, examples[0])
    assert scores.shape == (6, 6)
    assert np.allclose(scores, scores.T, atol=1e-6)
    assert np.isfinite(scores).all()

    instance = random_euclidean_instance(0, n_nodes=6)
    tour = greedy_edge_decoder(instance, scores)
    assert set(tour) == set(range(6))


def test_gin_edge_baseline_smoke():
    _assert_node_only_baseline_smoke("gin")


def test_pna_edge_baseline_smoke():
    _assert_node_only_baseline_smoke("pna")


def test_unknown_architecture_rejected():
    examples = build_exact_dataset([0], n_nodes=5)
    config = TrainConfig(architecture="not_a_model", epochs=1)
    try:
        fit_edge_model(examples, config)
    except ValueError as exc:
        assert "unknown architecture" in str(exc)
    else:
        raise AssertionError("unknown architecture should be rejected")
