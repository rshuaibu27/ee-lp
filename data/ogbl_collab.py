import torch
import functools


def load_ogbl_collab(root="./data"):
    # Patch torch.load for PyTorch 2.6 compatibility
    _original_load = torch.load
    torch.load = functools.partial(_original_load, weights_only=False)

    try:
        from ogb.linkproppred import PygLinkPropPredDataset
        dataset = PygLinkPropPredDataset(name="ogbl-collab", root=root)
        data    = dataset[0]
        split   = dataset.get_edge_split()
    finally:
        # Restore original torch.load
        torch.load = _original_load

    train_edges = split["train"]["edge"].t()   # (2, E_train)
    val_pos     = split["valid"]["edge"]       # (V, 2)
    test_pos    = split["test"]["edge"]        # (T, 2)
    val_neg     = split["valid"]["edge_neg"]   # (V, 2)
    test_neg    = split["test"]["edge_neg"]    # (T, 2)

    # Node features (year embeddings, dim=128)
    x = data.x if data.x is not None else \
        torch.ones(data.num_nodes, 1, dtype=torch.float)
    x = x.float()
    x = x / (x.norm(dim=1, keepdim=True).clamp(min=1e-6))

    # Training negatives (random, as per standard practice)
    train_neg = torch.randint(0, data.num_nodes,
                              (train_edges.t().size(0), 2))

    return dict(
        x=x,
        edge_index=train_edges,
        train_pos=train_edges.t().contiguous(),
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        num_nodes=data.num_nodes,
        num_features=x.size(1),
    )