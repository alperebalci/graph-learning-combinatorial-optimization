# Controlled GNN Architecture Ablation

> The filename is retained for link stability. The experiment now covers
> GraphSAGE, GIN and PNA-style baselines in addition to the primary EdgeGNN.

## Why this exists

The repository's primary `EdgeGNN` injects pairwise edge features into every
message-passing layer. That is a strong inductive bias for TSP because distances
are part of the combinatorial objective.

A collection of disconnected GNN tutorials would not test whether an
architecture improves the actual optimization pipeline. This ablation therefore
keeps the exact labels, model-selection protocol, decoder, local search and
test/OOD blocks fixed while changing only the neural encoder.

The controlled question is:

> Which message-passing inductive biases help downstream TSP decisions when the
> optimization pipeline is held fixed?

## Compared architectures

### `edge_gnn`

The repository's primary model. Pairwise edge features enter the learned message
function at every layer and the final edge scorer.

### `graphsage`

A GraphSAGE-style node encoder. It builds a k-nearest-neighbor graph, averages
neighbor states, combines self and neighborhood representations, and exposes
pairwise distance again only in the final edge scorer.

### `gin`

A GIN-style encoder using trainable epsilon and sum aggregation on a symmetrized
k-nearest-neighbor graph. The message-passing layers remain node-only; Euclidean
distance is used for neighborhood construction and final edge scoring.

This is useful because GIN is a high-expressivity MPNN baseline, but the
experiment does not pretend that a generic GIN layer automatically captures TSP
edge costs.

### `pna`

A compact PNA-style encoder using mean, max, min and standard-deviation
aggregators together with degree-based amplification and attenuation. The
symmetrized k-nearest-neighbor graph gives non-uniform degrees, so the degree
scalers remain meaningful.

The implementation is pure PyTorch and intentionally transparent. It follows the
core PNA design ideas rather than importing `torch_geometric.nn.PNAConv`.

## Why GraphSAINT is not included here

GraphSAINT is a subgraph-sampling/training method for large graphs, not a
drop-in message-passing architecture. This benchmark uses tiny exact-label TSP
instances (8-12 nodes) and evaluates complete candidate edge sets. Sampling
subgraphs would add a different training regime without solving a real
scalability bottleneck, so it would confound the architecture comparison.

GraphSAINT becomes appropriate when this research area contains genuinely large
graphs where full-batch message passing is a measured memory or throughput
problem.

## Experimental controls

All neural models use:

- the same exact Held-Karp training labels;
- the same training, validation, test and OOD instance seeds;
- the same hidden dimension, depth, optimizer and epoch budget;
- the same validation-based checkpointing;
- the same independent model seeds;
- the same multi-start beam decoder;
- the same 2-opt post-refinement;
- the same exact optimality-gap computation.

Model seed selection is performed on validation **decision gap**, not test or OOD
performance. The script also reports parameter count because PNA-style
aggregation has a larger update map and is not parameter-matched to the simpler
baselines.

The classical `nearest_neighbor + 2-opt` method remains in the output as a
non-neural reference.

## Run

```bash
pip install -e '.[dev,neural]'
python -m gnn_solver.architecture_ablation
```

The script reports, for `test`, `ood_10` and `ood_12`:

- selected model seed and validation gap;
- parameter count;
- mean and median exact optimality gap;
- end-to-end neural-score + decoder + 2-opt latency;
- feasibility rate.

## Interpretation

A lower edge-prediction loss is not sufficient evidence that an architecture is
better. Promotion should be based on downstream optimality gap under the same
feasibility-preserving decoder, together with latency, model size and OOD
behavior.

GIN and PNA are included because they add distinct aggregation biases that are
important in the GNN literature. They should remain baselines unless the
end-to-end optimization metrics justify a stronger role.
