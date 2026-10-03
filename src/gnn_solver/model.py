from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without neural extra
    raise ImportError("Install neural dependencies with pip install -e '.[neural]'") from exc


def _symmetric_knn_mask(distance, k_neighbors: int):
    n = distance.shape[0]
    if n <= 1:
        return torch.zeros((n, n), dtype=torch.bool, device=distance.device)
    k = min(k_neighbors, n - 1)
    penalty = torch.eye(n, dtype=distance.dtype, device=distance.device) * 1e9
    neighbor_index = torch.topk(distance + penalty, k=k, largest=False, dim=1).indices
    mask = torch.zeros((n, n), dtype=torch.bool, device=distance.device)
    mask.scatter_(1, neighbor_index, True)
    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask


class EdgeGNN(nn.Module):
    """Small permutation-equivariant GNN that scores undirected TSP edges."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_features):
        """Return symmetric edge logits for one complete graph.

        Shapes: node_features ``[n,2]`` and edge_features ``[n,n,3]``.
        """
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)
        n = h.shape[0]
        mask = 1.0 - torch.eye(n, dtype=h.dtype, device=h.device)

        for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
            hi = h[:, None, :].expand(n, n, -1)
            hj = h[None, :, :].expand(n, n, -1)
            messages = message_mlp(torch.cat([hi, hj, e], dim=-1))
            aggregate = (messages * mask[..., None]).sum(dim=1) / max(n - 1, 1)
            h = h + update_mlp(torch.cat([h, aggregate], dim=-1))

        hi = h[:, None, :].expand(n, n, -1)
        hj = h[None, :, :].expand(n, n, -1)
        pair = torch.cat([hi + hj, torch.abs(hi - hj), e], dim=-1)
        logits = self.edge_head(pair).squeeze(-1)
        logits = 0.5 * (logits + logits.T)
        return logits


class GraphSAGEEdgeBaseline(nn.Module):
    """GraphSAGE-style node aggregation baseline for TSP edge scoring.

    The primary model injects edge features into every message-passing step.
    This baseline deliberately uses only node states during neighborhood
    aggregation. Pairwise distance is exposed only to the final edge scorer.

    A k-nearest-neighbor graph is derived from the TSP distance matrix so that
    GraphSAGE does not collapse into near-global averaging on the complete graph.
    """

    def __init__(self, hidden_dim: int = 64, layers: int = 3, k_neighbors: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.k_neighbors = k_neighbors
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sage_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _knn_neighbor_mean(self, h, distance):
        n = h.shape[0]
        if n <= 1:
            return h
        k = min(self.k_neighbors, n - 1)
        penalty = torch.eye(n, dtype=distance.dtype, device=distance.device) * 1e9
        neighbor_index = torch.topk(distance + penalty, k=k, largest=False, dim=1).indices
        return h[neighbor_index].mean(dim=1)

    def forward(self, node_features, edge_features):
        """Return symmetric TSP edge logits with GraphSAGE-style aggregation."""
        h = self.node_encoder(node_features)
        distance = edge_features[..., 0]

        for update in self.sage_updates:
            neighbor_mean = self._knn_neighbor_mean(h, distance)
            h = h + update(torch.cat([h, neighbor_mean], dim=-1))

        n = h.shape[0]
        hi = h[:, None, :].expand(n, n, -1)
        hj = h[None, :, :].expand(n, n, -1)
        pair = torch.cat(
            [hi + hj, torch.abs(hi - hj), distance[..., None]],
            dim=-1,
        )
        logits = self.edge_head(pair).squeeze(-1)
        logits = 0.5 * (logits + logits.T)
        return logits


class GINEdgeBaseline(nn.Module):
    """GIN-style k-NN node encoder adapted to symmetric TSP edge scoring.

    GIN sum aggregation is intentionally node-only. Euclidean distance defines
    the sparse neighborhood and is exposed again only in the final edge scorer.
    This keeps the comparison with the edge-aware primary model interpretable.
    """

    def __init__(self, hidden_dim: int = 64, layers: int = 3, k_neighbors: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.k_neighbors = k_neighbors
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(layers))
        self.gin_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_features):
        h = self.node_encoder(node_features)
        distance = edge_features[..., 0]
        mask = _symmetric_knn_mask(distance, self.k_neighbors)
        adjacency = mask.to(dtype=h.dtype)

        for eps, mlp in zip(self.eps, self.gin_mlps):
            neighbor_sum = adjacency @ h
            h = mlp((1.0 + eps) * h + neighbor_sum)

        n = h.shape[0]
        hi = h[:, None, :].expand(n, n, -1)
        hj = h[None, :, :].expand(n, n, -1)
        pair = torch.cat([hi + hj, torch.abs(hi - hj), distance[..., None]], dim=-1)
        logits = self.edge_head(pair).squeeze(-1)
        return 0.5 * (logits + logits.T)


class PNAEdgeBaseline(nn.Module):
    """PNA-style multi-aggregator k-NN encoder for TSP edge scoring.

    The block combines mean/max/min/std neighborhood statistics with degree-based
    amplification and attenuation before a residual node update. It is a compact
    pure-PyTorch adaptation of PNA principles rather than a dependency on PyG's
    ``PNAConv``.
    """

    def __init__(self, hidden_dim: int = 64, layers: int = 3, k_neighbors: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.k_neighbors = k_neighbors
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pna_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 13, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _aggregate(h, mask):
        adjacency = mask.to(dtype=h.dtype)
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = adjacency @ h / degree

        expanded = h.unsqueeze(0).expand(h.shape[0], -1, -1)
        max_values = expanded.masked_fill(~mask[..., None], float("-inf")).max(dim=1).values
        min_values = expanded.masked_fill(~mask[..., None], float("inf")).min(dim=1).values
        max_values = torch.where(torch.isfinite(max_values), max_values, h)
        min_values = torch.where(torch.isfinite(min_values), min_values, h)

        second_moment = adjacency @ (h * h) / degree
        std = torch.sqrt(torch.clamp(second_moment - mean * mean, min=0.0) + 1e-6)
        aggregate = torch.cat([mean, max_values, min_values, std], dim=-1)

        log_degree = torch.log1p(degree)
        average_log_degree = log_degree.mean().clamp_min(1e-6)
        amplification = log_degree / average_log_degree
        attenuation = average_log_degree / log_degree.clamp_min(1e-6)
        return torch.cat(
            [aggregate, aggregate * amplification, aggregate * attenuation],
            dim=-1,
        )

    def forward(self, node_features, edge_features):
        h = self.node_encoder(node_features)
        distance = edge_features[..., 0]
        mask = _symmetric_knn_mask(distance, self.k_neighbors)

        for update in self.pna_updates:
            aggregate = self._aggregate(h, mask)
            h = h + update(torch.cat([h, aggregate], dim=-1))

        n = h.shape[0]
        hi = h[:, None, :].expand(n, n, -1)
        hj = h[None, :, :].expand(n, n, -1)
        pair = torch.cat([hi + hj, torch.abs(hi - hj), distance[..., None]], dim=-1)
        logits = self.edge_head(pair).squeeze(-1)
        return 0.5 * (logits + logits.T)

