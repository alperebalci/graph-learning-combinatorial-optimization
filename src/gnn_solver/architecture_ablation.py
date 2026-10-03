from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .beam_decoder import multistart_beam_decoder
from .dataset import TSPExample, build_exact_dataset
from .exact import held_karp
from .heuristics import nearest_neighbor, two_opt
from .instance import random_euclidean_instance
from .train import TrainConfig, fit_edge_model, predict_edge_scores


@dataclass(frozen=True)
class ArchitectureResult:
    architecture: str
    split: str
    seed: int
    n_nodes: int
    gap_pct: float
    latency_ms: float
    feasible: bool


def _gap(instance, tour, optimal_cost: float) -> tuple[float, bool]:
    try:
        cost = instance.tour_cost(tour)
    except ValueError:
        return float("inf"), False
    return 100.0 * (cost - optimal_cost) / optimal_cost, True


def _decision_gap(model, example: TSPExample, beam_width: int) -> float:
    instance = random_euclidean_instance(
        example.seed,
        n_nodes=example.node_features.shape[0],
    )
    optimal_cost = float(held_karp(instance)["cost"])
    scores = predict_edge_scores(model, example)
    tour = multistart_beam_decoder(instance, scores, beam_width=beam_width, refine=True)
    gap, feasible = _gap(instance, tour, optimal_cost)
    return gap if feasible else float("inf")


def select_model(
    architecture: str,
    train_examples: list[TSPExample],
    validation_examples: list[TSPExample],
    *,
    model_seeds: tuple[int, ...] = (0, 1, 2),
    hidden_dim: int = 32,
    layers: int = 2,
    epochs: int = 12,
    beam_width: int = 6,
    k_neighbors: int = 3,
):
    candidates = []
    for seed in model_seeds:
        model, history = fit_edge_model(
            train_examples,
            TrainConfig(
                hidden_dim=hidden_dim,
                layers=layers,
                epochs=epochs,
                seed=seed,
                architecture=architecture,
                k_neighbors=k_neighbors,
            ),
            validation_examples=validation_examples,
        )
        validation_gap = float(
            np.mean(
                [
                    _decision_gap(model, example, beam_width)
                    for example in validation_examples
                ]
            )
        )
        candidates.append((validation_gap, seed, model, history[-1]))

    candidates.sort(key=lambda row: (row[0], row[1]))
    return {
        "architecture": architecture,
        "model": candidates[0][2],
        "selected_seed": candidates[0][1],
        "validation_gap_pct": candidates[0][0],
        "final_training_loss": candidates[0][3],
        "parameter_count": sum(p.numel() for p in candidates[0][2].parameters()),
        "all_validation_gaps": {seed: gap for gap, seed, _, _ in candidates},
    }


def evaluate_model(
    model,
    architecture: str,
    *,
    split: str,
    seeds: list[int],
    n_nodes: int,
    beam_width: int = 6,
) -> list[ArchitectureResult]:
    examples = {example.seed: example for example in build_exact_dataset(seeds, n_nodes=n_nodes)}
    rows = []
    for seed in seeds:
        instance = random_euclidean_instance(seed, n_nodes=n_nodes)
        optimal_cost = float(held_karp(instance)["cost"])

        start = perf_counter()
        scores = predict_edge_scores(model, examples[seed])
        tour = multistart_beam_decoder(instance, scores, beam_width=beam_width, refine=True)
        latency_ms = (perf_counter() - start) * 1000.0
        gap, feasible = _gap(instance, tour, optimal_cost)
        rows.append(
            ArchitectureResult(
                architecture=architecture,
                split=split,
                seed=seed,
                n_nodes=n_nodes,
                gap_pct=gap,
                latency_ms=latency_ms,
                feasible=feasible,
            )
        )
    return rows


def evaluate_classical_reference(
    *,
    split: str,
    seeds: list[int],
    n_nodes: int,
) -> list[ArchitectureResult]:
    rows = []
    for seed in seeds:
        instance = random_euclidean_instance(seed, n_nodes=n_nodes)
        optimal_cost = float(held_karp(instance)["cost"])
        start = perf_counter()
        tour = two_opt(instance, nearest_neighbor(instance))
        latency_ms = (perf_counter() - start) * 1000.0
        gap, feasible = _gap(instance, tour, optimal_cost)
        rows.append(
            ArchitectureResult(
                architecture="nearest_neighbor_2opt",
                split=split,
                seed=seed,
                n_nodes=n_nodes,
                gap_pct=gap,
                latency_ms=latency_ms,
                feasible=feasible,
            )
        )
    return rows


def summarize(rows: list[ArchitectureResult]) -> list[dict[str, float | str]]:
    result = []
    keys = sorted({(row.split, row.architecture) for row in rows})
    for split, architecture in keys:
        selected = [
            row
            for row in rows
            if row.split == split and row.architecture == architecture
        ]
        result.append(
            {
                "split": split,
                "architecture": architecture,
                "mean_gap_pct": float(np.mean([row.gap_pct for row in selected])),
                "median_gap_pct": float(np.median([row.gap_pct for row in selected])),
                "mean_latency_ms": float(np.mean([row.latency_ms for row in selected])),
                "feasibility_rate": float(np.mean([row.feasible for row in selected])),
            }
        )
    return result


def main() -> None:
    train_examples = build_exact_dataset(list(range(20)), n_nodes=8)
    validation_examples = build_exact_dataset([50, 51, 52, 53, 54], n_nodes=8)

    selections = {
        architecture: select_model(
            architecture,
            train_examples,
            validation_examples,
            model_seeds=(0, 1, 2),
            epochs=12,
            beam_width=6,
        )
        for architecture in ("edge_gnn", "graphsage", "gin", "pna")
    }

    for name, selection in selections.items():
        print(
            f"{name}: selected_seed={selection['selected_seed']}, "
            f"validation_gap={selection['validation_gap_pct']:.4f}%, "
            f"parameters={selection['parameter_count']}"
        )

    rows: list[ArchitectureResult] = []
    splits = [
        ("test", [100, 101, 102, 103], 8),
        ("ood_10", [200, 201, 202], 10),
        ("ood_12", [300, 301], 12),
    ]
    for split, seeds, n_nodes in splits:
        rows.extend(evaluate_classical_reference(split=split, seeds=seeds, n_nodes=n_nodes))
        for architecture, selection in selections.items():
            rows.extend(
                evaluate_model(
                    selection["model"],
                    architecture,
                    split=split,
                    seeds=seeds,
                    n_nodes=n_nodes,
                    beam_width=6,
                )
            )

    for row in summarize(rows):
        print(
            f"{row['split']},{row['architecture']},"
            f"gap={row['mean_gap_pct']:.3f}%,"
            f"latency_ms={row['mean_latency_ms']:.3f},"
            f"feasible={row['feasibility_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
