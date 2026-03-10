"""
Reproduces all results from the paper:
  "EE-LP: Pairwise Early-Exit Graph Neural Networks for Link Prediction"
"""

import argparse
import json
import os
import functools
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.utils import degree

from models.ee_lp import EELP, GCNLinkPredictor, SAGELinkPredictor, GATLinkPredictor
from data.dataset import load_dataset, hits_at_k, mrr
from data.ogbl_collab import load_ogbl_collab
from train import run


# Config
EELP_CFG = dict(hidden_dim=64, L=20, lf=1, nu0=1.0, lr=1e-3,
                weight_decay=1e-5, epochs=500, patience=50,
                batch_size=512, log_every=25)

BASELINE_CFG = dict(hidden_dim=64, L=3, lr=1e-3,
                    weight_decay=1e-5, epochs=500, patience=50,
                    batch_size=512, log_every=25)

COLLAB_EELP_CFG = dict(hidden_dim=64, L=20, lf=1, nu0=1.0, lr=1e-3,
                       weight_decay=1e-5, epochs=200, patience=30,
                       batch_size=1024, log_every=20)

COLLAB_BASELINE_CFG = dict(hidden_dim=64, L=3, lr=1e-3,
                           weight_decay=1e-5, epochs=200, patience=30,
                           batch_size=1024, log_every=20)

SEEDS = [0, 1, 2]


# Helpers
def agg(rlist, key):
    vals = [r[key] for r in rlist if key in r]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def build_model(model_name, in_dim, is_collab=False):
    if model_name == "EE-LP":
        cfg = COLLAB_EELP_CFG if is_collab else EELP_CFG
        return EELP(in_dim=in_dim,
                    hidden_dim=cfg["hidden_dim"],
                    L=cfg["L"],
                    lf=cfg["lf"],
                    nu0=cfg["nu0"]), cfg
    cfg = COLLAB_BASELINE_CFG if is_collab else BASELINE_CFG
    classes = {"GCN": GCNLinkPredictor,
               "SAGE": SAGELinkPredictor,
               "GAT": GATLinkPredictor}
    return classes[model_name](in_dim=in_dim,
                               hidden_dim=cfg["hidden_dim"],
                               L=cfg["L"]), cfg



# Plotting and analysis
def plot_comparison(all_results, datasets, out_dir):
    models = ["GCN", "SAGE", "GAT", "EE-LP"]
    colors = ["#4393c3", "#92c5de", "#d6604d", "#2ca02c"]
    x      = np.arange(len(datasets))
    width  = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 3), 4.5))
    for i, (m, c) in enumerate(zip(models, colors)):
        means, stds = [], []
        for ds in datasets:
            mu, sd = agg(all_results.get(ds, {}).get(m, []), "Hits@20")
            means.append(mu or 0.0)
            stds.append(sd or 0.0)
        ax.bar(x + i * width, means, width, yerr=stds,
               capsize=5, label=m, color=c,
               edgecolor="black", alpha=0.88)

    ax.set_ylabel("Hits@20", fontsize=13)
    ax.set_title("Link Prediction Performance — HeaRT Protocol", fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "hits_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_exit_distributions(all_results, datasets, out_dir):
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        rlist = all_results.get(ds, {}).get("EE-LP", [])
        exits_mean = [r["mean_exit_layer"] for r in rlist if "mean_exit_layer" in r]
        exits_std  = [r["exit_std"]        for r in rlist if "exit_std" in r]

        seeds = [f"Seed {i}" for i in range(len(exits_mean))]
        ax.bar(seeds, exits_mean, yerr=exits_std,
               capsize=6, color="steelblue",
               edgecolor="black", alpha=0.8)
        ax.axhline(y=20, color="grey", linestyle="--",
                   linewidth=1.5, label="Max L=20")
        ax.set_ylim(0, 22)
        ax.set_ylabel("Mean exit layer", fontsize=12)
        ax.set_title(ds, fontsize=13)
        ax.legend(fontsize=10)

    plt.suptitle("EE-LP Mean Exit Layer per Seed", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "exit_layers.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_degree_analysis(data, dataset_name, out_dir, device):
    """Degree-stratified exit analysis for a single dataset."""
    edge_index = data["edge_index"]
    num_nodes  = data["num_nodes"]
    test_pos   = data["test_pos"]
    deg        = degree(edge_index[0], num_nodes=num_nodes).cpu().numpy()

    all_exits, all_degrees = [], []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = EELP(in_dim=data["num_features"],
                     hidden_dim=64, L=20, lf=1, nu0=1.0).to(device)
        cfg = dict(epochs=500, patience=50, log_every=500,
                   batch_size=512, lr=1e-3, weight_decay=1e-5)
        run(model, data, cfg, device)
        model.eval()
        with torch.no_grad():
            _, exit_times = model(data["x"].to(device),
                                  edge_index.to(device),
                                  test_pos.to(device))
        src = test_pos[:, 0].numpy()
        dst = test_pos[:, 1].numpy()
        all_exits.append(exit_times.cpu().numpy())
        all_degrees.append((deg[src] + deg[dst]) / 2.0)

    all_exits   = np.concatenate(all_exits)
    all_degrees = np.concatenate(all_degrees)
    terciles    = np.percentile(all_degrees, [33, 66])

    means = [all_exits[all_degrees <= terciles[0]].mean(),
             all_exits[(all_degrees > terciles[0]) &
                       (all_degrees <= terciles[1])].mean(),
             all_exits[all_degrees > terciles[1]].mean()]
    stds  = [all_exits[all_degrees <= terciles[0]].std(),
             all_exits[(all_degrees > terciles[0]) &
                       (all_degrees <= terciles[1])].std(),
             all_exits[all_degrees > terciles[1]].std()]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Low", "Mid", "High"], means, yerr=stds, capsize=6,
           color=["#4393c3", "#92c5de", "#d6604d"],
           edgecolor="black", alpha=0.85)
    ax.axhline(y=20, color="grey", linestyle="--",
               linewidth=1.5, label="Max L=20")
    ax.set_ylim(0, 22)
    ax.set_xlabel("Node pair degree group", fontsize=12)
    ax.set_ylabel("Mean exit layer", fontsize=12)
    ax.set_title(f"Exit vs Degree — {dataset_name}", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"degree_exit_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["cora", "citeseer"],
                        choices=["cora", "citeseer", "collab", "all"])
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--device",  type=str, default="auto")
    args = parser.parse_args()

    if "all" in args.datasets:
        args.datasets = ["cora", "citeseer", "collab"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    model_names = ["GCN", "SAGE", "GAT", "EE-LP"]

    for ds_name in args.datasets:
        print(f"\n{'='*55}\n  {ds_name}\n{'='*55}")
        is_collab = ds_name == "collab"

        if is_collab:
            data = load_ogbl_collab(root="./data")
            ds_key = "ogbl-collab"
        else:
            data   = load_dataset(ds_name.capitalize(), hard_negatives=True)
            ds_key = ds_name.capitalize()

        all_results[ds_key] = {}

        for model_name in model_names:
            print(f"\n  -- {model_name} --")
            seed_results = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model, cfg = build_model(model_name,
                                         data["num_features"],
                                         is_collab=is_collab)
                res, _ = run(model, data, cfg, device)
                seed_results.append(res)
                print(f"  Seed {seed} | "
                      f"Hits@20: {res.get('Hits@20', 0):.4f} "
                      f"MRR: {res.get('MRR', 0):.4f}", end="")
                if "mean_exit_layer" in res:
                    print(f" | Exit: {res['mean_exit_layer']:.1f}"
                          f"±{res['exit_std']:.1f}")
                else:
                    print()

            all_results[ds_key][model_name] = seed_results

        # Degree analysis for small datasets only
        if not is_collab:
            plot_degree_analysis(data, ds_key, args.out_dir, device)

    # Summary
    print(f"\n\n{'='*65}\n  FINAL RESULTS\n{'='*65}")
    print(f"  {'Dataset':<14} {'Model':<8} "
          f"{'Hits@20':>9} {'Hits@50':>9} {'MRR':>8}")
    print(f"  {'-'*55}")
    for ds_key, mdict in all_results.items():
        for m in model_names:
            if m not in mdict:
                continue
            row = f"  {ds_key:<14} {m:<8}"
            for k in ["Hits@20", "Hits@50", "MRR"]:
                mu, sd = agg(mdict[m], k)
                row += f" {mu:.3f}±{sd:.3f}" if mu else f"{'N/A':>9}"
            if m == "EE-LP":
                mu, sd = agg(mdict[m], "mean_exit_layer")
                if mu:
                    row += f"  exit={mu:.1f}±{sd:.1f}"
            print(row)

    # Plots
    datasets_list = list(all_results.keys())
    plot_comparison(all_results, datasets_list, args.out_dir)
    plot_exit_distributions(all_results, datasets_list, args.out_dir)

    # Save JSON
    json_out = {}
    for ds_key, mdict in all_results.items():
        json_out[ds_key] = {}
        for m, rlist in mdict.items():
            json_out[ds_key][m] = [
                {k: float(v) if isinstance(v, (float, int, np.floating))
                 else v for k, v in r.items()}
                for r in rlist
            ]
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()