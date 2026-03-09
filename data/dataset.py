import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit


def sample_hard_negatives(pos_pairs, edge_index, num_nodes, num_neg=1, seed=42):
    rng = np.random.default_rng(seed)
    adj = {i: set() for i in range(num_nodes)}
    for s, d in zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()):
        adj[s].add(d)
        adj[d].add(s)

    neg_src, neg_dst = [], []
    for u, v in pos_pairs.cpu().numpy():
        u = int(u)
        candidates = [w for w in range(num_nodes)
                      if w != u and w not in adj[u]]
        if len(candidates) == 0:
            candidates = [w for w in range(num_nodes) if w != u]
        chosen = rng.choice(candidates,
                            size=min(num_neg, len(candidates)),
                            replace=False)
        for w in chosen:
            neg_src.append(u)
            neg_dst.append(w)

    return torch.tensor(list(zip(neg_src, neg_dst)), dtype=torch.long)


def load_dataset(name="Cora", root="./data", hard_negatives=True):
    assert name in ("Cora", "CiteSeer"), f"Expected Cora or CiteSeer, got {name}"

    dataset = Planetoid(root=root, name=name)
    data    = dataset[0]

    # Normalise features
    x      = data.x.float()
    x      = x / (x.sum(dim=1, keepdim=True).clamp(min=1))
    data.x = x

    splitter = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0,
    )
    train_data, val_data, test_data = splitter(data)

    train_edge_index = train_data.edge_index
    num_nodes        = data.num_nodes

    def pos_pairs(d):
        mask = d.edge_label == 1
        return d.edge_label_index[:, mask].t().contiguous()

    train_pos = pos_pairs(train_data)
    val_pos   = pos_pairs(val_data)
    test_pos  = pos_pairs(test_data)

    if hard_negatives:
        val_neg  = sample_hard_negatives(val_pos,  train_edge_index,
                                         num_nodes, seed=0)
        test_neg = sample_hard_negatives(test_pos, train_edge_index,
                                         num_nodes, seed=1)
    else:
        val_neg  = torch.randint(0, num_nodes, (val_pos.size(0), 2))
        test_neg = torch.randint(0, num_nodes, (test_pos.size(0), 2))

    train_neg = torch.randint(0, num_nodes, (train_pos.size(0), 2))

    return dict(
        x=data.x,
        edge_index=train_edge_index,
        train_pos=train_pos,
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        num_nodes=num_nodes,
        num_features=data.num_features,
    )


def hits_at_k(pos_scores, neg_scores, k=20):
    pos_scores = pos_scores.detach().cpu()
    neg_scores = neg_scores.detach().cpu()

    n = pos_scores.size(0)
    if neg_scores.size(0) >= n:
        neg_scores = neg_scores[:n]
    else:
        rep = (n // neg_scores.size(0)) + 1
        neg_scores = neg_scores.repeat(rep)[:n]

    threshold = torch.topk(neg_scores, min(k, n)).values[-1]
    return (pos_scores > threshold).float().mean().item()


def mrr(pos_scores, neg_scores):
    pos_scores = pos_scores.detach().cpu()
    neg_scores = neg_scores.detach().cpu()
    ranks = []
    for i in range(pos_scores.size(0)):
        rank = 1 + (neg_scores > pos_scores[i]).sum().item()
        ranks.append(1.0 / rank)
    return float(np.mean(ranks))