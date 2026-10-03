# Graph Learning for Combinatorial Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`gflownet-combinatorial-optimization`](projects/gflownet-combinatorial-optimization/)
- [`gnn-minimum-vertex-cover`](projects/gnn-minimum-vertex-cover/)
- [`graph-attention-network-cora`](projects/graph-attention-network-cora/)
- [`graphsage-cora-node-classification`](projects/graphsage-cora-node-classification/)
- [`neural-algorithmic-reasoning-combinatorial-optimization`](projects/neural-algorithmic-reasoning-combinatorial-optimization/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

Research-oriented benchmark for **learning-augmented combinatorial optimization** with graph neural networks and exact/heuristic OR baselines.

## Research question

Can a graph neural model learn structural information from exact small-instance solutions and use it to guide a fast combinatorial decoder without sacrificing feasibility, and when does that guidance outperform classical heuristics after accounting for optimality gap, latency and distribution shift?

## Current status

**Phase 3 implemented: multi-seed statistical evaluation + deeper OOD size shift + latency decomposition.**

A controlled **GNN architecture ablation** is also included: the primary edge-aware model is compared with GraphSAGE-, GIN- and PNA-style node encoders under the same TSP training, decoding and OOD protocol.

The repository now includes:

- deterministic Euclidean TSP generation;
- exact Held-Karp oracle;
- nearest-neighbor and 2-opt baselines;
- exact-tour edge supervision;
- permutation-equivariant PyTorch message-passing GNN;
- GraphSAGE-style k-NN node-aggregation baseline for architecture ablation;
- greedy, beam and multi-start feasible decoding;
- 2-opt post-refinement;
- validation checkpoint selection;
- independent model seeds `0/1/2`;
- instance-level model-seed aggregation;
- frozen test and OOD node-count blocks;
- paired bootstrap 95% confidence intervals;
- exact sign tests and paired win rates;
- inference / decoder / local-search / total latency decomposition;
- unit tests and GitHub Actions CI.

## Problem

The initial benchmark is symmetric Euclidean TSP. The learning model predicts edge utility; a separate combinatorial decoder constructs only Hamiltonian tours. The GNN is therefore guidance, not a replacement for feasibility logic.

## Evaluation protocol

Training uses 8-node exact instances. Validation is used only for checkpoint/model selection. Final evaluation uses disjoint blocks:

- `test`: 8 nodes, seeds `100-103`;
- `ood_10`: 10 nodes, seeds `200-202`;
- `ood_12`: 12 nodes, seeds `300-301`.

The principal learned method is `gnn_beam_2opt`. Its reference is **nearest-neighbor + 2-opt**.

Three independently trained GNN seeds are evaluated on every instance. Their results are averaged **within each instance first**; model seeds are not treated as independent test samples. Paired inference is then performed over instance seeds.

## Statistical reporting

For GNN beam + 2-opt versus nearest-neighbor + 2-opt, Phase 3 reports:

- mean paired optimality-gap difference;
- paired 95% bootstrap confidence interval;
- paired win rate;
- exact two-sided sign-test p-value;
- number of paired instances.

A negative paired gap difference favors the GNN method. Statistical significance alone is not considered sufficient for promotion; latency and feasibility remain part of the decision.

## Architecture ablation

`python -m gnn_solver.architecture_ablation` compares the primary edge-aware `EdgeGNN` with GraphSAGE-, GIN- and PNA-style baselines under matched training data, exact labels, model seeds, validation selection, decoder, 2-opt refinement and test/OOD blocks.

The node-only baselines use sparse k-nearest-neighbor structure with different aggregation biases while pairwise distance remains available for neighborhood construction and final edge scoring. The experiment also reports parameter counts so PNA's extra capacity is visible. GraphSAINT is deliberately excluded here because these tiny exact-label graphs do not have a sampling bottleneck.

See [`docs/graphsage_ablation.md`](./docs/graphsage_ablation.md) for the controlled experimental contract and the GraphSAINT scope rationale.

## Latency decomposition

The neural pipeline records separately:

1. GNN edge-score inference;
2. multi-start beam decoding;
3. 2-opt refinement;
4. total end-to-end decision latency.

This prevents local-search time from being hidden inside a generic neural inference number.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,neural]'
ruff check src tests
pytest -q
python -m gnn_solver.evaluate
python -m gnn_solver.phase2_experiment
python -m gnn_solver.phase3_experiment
python -m gnn_solver.architecture_ablation
```

## Repository layout

```text
src/gnn_solver/
  instance.py
  exact.py
  heuristics.py
  features.py
  dataset.py
  model.py
  decoder.py
  beam_decoder.py
  train.py
  statistics.py
  evaluate.py
  phase2_experiment.py
  phase3_experiment.py
  architecture_ablation.py
tests/
  test_core.py
  test_neural.py
  test_phase2.py
  test_phase3.py
configs/
  experiment.json
docs/
  experimental_protocol.md
  graphsage_ablation.md
.github/workflows/
  ci.yml
```

## Research contract

- feasibility must remain 100%;
- exact optimality gaps are used where tractable;
- test/OOD blocks are never used for model selection;
- model-seed replication is not confused with test-sample replication;
- nearest-neighbor + 2-opt is the main classical reference;
- architecture comparisons must hold decoder and evaluation protocol fixed;
- quality gains must be interpreted together with latency;
- negative or null GNN results are retained.

## Next research stage

### Phase 4 — CVRP extension

Reuse the graph-learning stack on capacitated vehicle routing with explicit demand and remaining-vehicle-capacity state, exact/small-instance reference solutions, strong OR heuristics and a feasibility-preserving neural decoder.

## Scope boundary

The TSP benchmark is now complete enough to serve as the controlled methodology layer. GraphSAGE, GIN and PNA are retained only as controlled architecture ablations; generic node-classification and embedding-visualization tutorials are intentionally out of scope. Large-scale VRP, learned branching, neural cut selection and scheduling graphs should be separate extensions rather than silent additions to the TSP benchmark.

## License

MIT
