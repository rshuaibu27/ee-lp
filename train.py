import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import trange
import numpy as np

from data.dataset import hits_at_k, mrr


def binary_cross_entropy_loss(pos_scores, neg_scores):
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores)
    ])
    return F.binary_cross_entropy_with_logits(scores, labels)


def train_epoch(model, data, optimiser, device, batch_size=512):
    model.train()
    x          = data["x"].to(device)
    edge_index = data["edge_index"].to(device)
    train_pos  = data["train_pos"].to(device)
    train_neg  = data["train_neg"].to(device)

    # Shuffle pairs each epoch
    perm      = torch.randperm(train_pos.size(0), device=device)
    train_pos = train_pos[perm]
    train_neg = train_neg[perm[:train_neg.size(0)]]

    total_loss = 0.0
    n_batches  = 0

    for start in range(0, train_pos.size(0), batch_size):
        end     = min(start + batch_size, train_pos.size(0))
        pos_b   = train_pos[start:end]
        neg_b   = train_neg[start:end]
        pairs_b = torch.cat([pos_b, neg_b], dim=0)

        optimiser.zero_grad()
        scores, _ = model(x, edge_index, pairs_b)

        n_pos = pos_b.size(0)
        loss  = binary_cross_entropy_loss(scores[:n_pos], scores[n_pos:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data, split, device, k_list=(10, 20, 50)):
    model.eval()
    x         = data["x"].to(device)
    ei        = data["edge_index"].to(device)
    pos_pairs = data[f"{split}_pos"].to(device)
    neg_pairs = data[f"{split}_neg"].to(device)

    pos_scores, pos_times = model(x, ei, pos_pairs)
    neg_scores, _         = model(x, ei, neg_pairs)

    results = {}
    for k in k_list:
        results[f"Hits@{k}"] = hits_at_k(pos_scores, neg_scores, k=k)
    results["MRR"] = mrr(pos_scores, neg_scores)

    if pos_times is not None:
        results["mean_exit_layer"] = pos_times.mean().item()
        results["exit_std"]        = pos_times.std().item()

    return results


def run(model, data, cfg, device):
    model = model.to(device)
    optimiser = Adam(model.parameters(),
                     lr=cfg.get("lr", 1e-3),
                     weight_decay=cfg.get("weight_decay", 1e-5))

    best_val  = -1.0
    best_test = None
    patience_counter = 0
    patience  = cfg.get("patience", 50)
    epochs    = cfg.get("epochs", 500)
    log_every = cfg.get("log_every", 25)

    history = {"train_loss": [], "val_hits20": [], "mean_exit": []}

    for epoch in trange(1, epochs + 1, desc="Training"):
        loss = train_epoch(model, data, optimiser, device,
                           batch_size=cfg.get("batch_size", 512))
        history["train_loss"].append(loss)

        if epoch % log_every == 0 or epoch == epochs:
            val_res  = evaluate(model, data, "val",  device)
            test_res = evaluate(model, data, "test", device)

            val_h20 = val_res["Hits@20"]
            history["val_hits20"].append(val_h20)

            if "mean_exit_layer" in val_res:
                history["mean_exit"].append(val_res["mean_exit_layer"])

            if val_h20 > best_val:
                best_val  = val_h20
                best_test = test_res
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    return best_test, history