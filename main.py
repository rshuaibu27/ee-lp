import argparse
import json
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import load_dataset
from models.ee_lp import EELP, GCNLinkPredictor
from train import run


# Hyperparameters

EELP_CFG = dict(
    hidden_dim=64,
    L=20,
    lf=1,
    nu0=1.0,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=500,
    patience=50,
    batch_size=512,
    log_every=25,
)

GCN_CFG = dict(
    hidden_dim=64,
    L=3,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=500,
    patience=50,
    batch_size=512,
    log_every=25,
)

SEEDS = [0, 1, 2]


# Single run
def run_seed(model_name, dataset_name, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data   = load_dataset(dataset_name, hard_negatives=True)
    in_dim = data["num_features"]

    if model_name == "EE-LP":
        model = EELP(in_dim=in_dim,
                     hidden_dim=EELP_CFG["hidden_dim"],
                     L=EELP_CFG["L"],
                     lf=EELP_CFG["lf"],
                     nu0=EELP_CFG["nu0"])
        cfg = EELP_CFG
    else:
        model = GCNLinkPredictor(in_dim=in_dim,
                                 hidden_dim=GCN_CFG["hidden_dim"],
                                 L=GCN_CFG["L"])
        cfg = GCN_CFG

    test_results, history = run(model, data, cfg, device)
    return test_results, history


# Helpers

def aggregate(results_list, key):
    vals = [r[key] for r in results_list if key in r]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


# Plots
def plot_exit_dynamics(history_list, dataset_name, out_dir):
    """How mean exit layer changes over training."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, h in enumerate(history_list):
        exits = h.get("mean_exit", [])
        if exits:
            ax.plot(exits, alpha=0.7, label=f"seed {i}")
    ax.axhline(y=EELP_CFG["L"], color="grey", linestyle="--",
               label=f"Max L={EELP_CFG['L']}")
    ax.set_xlabel("Evaluation checkpoint")
    ax.set_ylabel("Mean exit layer")
    ax.set_title(f"EE-LP exit dynamics — {dataset_name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"exit_dynamics_{dataset_name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison(all_results, out_dir):
    """Bar chart: Hits@20 for GCN vs EE-LP on each dataset."""
    models   = ["GCN", "EE-LP"]
    datasets = list(all_results.keys())
    x        = np.arange(len(datasets))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(models):
        means, stds = [], []
        for ds in datasets:
            mu, sd = aggregate(all_results[ds].get(m, []), "Hits@20")
            means.append(mu or 0.0)
            stds.append(sd or 0.0)
        ax.bar(x + i * width, means, width,
               yerr=stds, capsize=4, label=m, alpha=0.85)

    ax.set_ylabel("Hits@20")
    ax.set_title("Link Prediction — HeaRT protocol")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "hits_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_exit_histogram(model, data, device, dataset_name, out_dir):
    """Distribution of per-pair exit depths on the test set."""
    model.eval()
    x  = data["x"].to(device)
    ei = data["edge_index"].to(device)
    tp = data["test_pos"].to(device)

    with torch.no_grad():
        _, exit_times = model(x, ei, tp)

    exit_times = exit_times.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(exit_times, bins=20, edgecolor="black",
            alpha=0.75, color="steelblue")
    ax.axvline(exit_times.mean(), color="red", linestyle="--",
               label=f"Mean = {exit_times.mean():.1f}")
    ax.set_xlabel("Continuous exit time (sum of τ values)")
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"EE-LP exit distribution — {dataset_name} test set")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"exit_histogram_{dataset_name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["Cora", "CiteSeer", "all"])
    parser.add_argument("--seeds",   type=int, default=3)
    parser.add_argument("--device",  type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    datasets = (["Cora", "CiteSeer"]
                if args.dataset == "all" else [args.dataset])
    seeds    = list(range(args.seeds))
    models   = ["GCN", "EE-LP"]

    all_results = {ds: {} for ds in datasets}

    for ds in datasets:
        print(f"\n{'='*55}\n Dataset: {ds}\n{'='*55}")

        for model_name in models:
            print(f"\n  -- {model_name} --")
            seed_results   = []
            seed_histories = []

            for seed in seeds:
                print(f"  Seed {seed}...")
                res, hist = run_seed(model_name, ds, seed, device)
                seed_results.append(res)
                seed_histories.append(hist)
                print(f"  Hits@20: {res.get('Hits@20', 0):.4f}  "
                      f"MRR: {res.get('MRR', 0):.4f}")
                if "mean_exit_layer" in res:
                    print(f"  Mean exit: {res['mean_exit_layer']:.2f} "
                          f"± {res['exit_std']:.2f} layers")

            all_results[ds][model_name] = seed_results

            if model_name == "EE-LP":
                plot_exit_dynamics(seed_histories, ds, args.out_dir)

        # Print summary table
        print(f"\n  Results for {ds}:")
        print(f"  {'Model':<10} {'Hits@10':>9} {'Hits@20':>9} "
              f"{'Hits@50':>9} {'MRR':>8}")
        print("  " + "-" * 50)
        for m in models:
            row = f"  {m:<10}"
            for k in ["Hits@10", "Hits@20", "Hits@50", "MRR"]:
                mu, sd = aggregate(all_results[ds][m], k)
                row += f" {mu:.3f}±{sd:.3f}" if mu is not None \
                       else f" {'N/A':>9}"
            print(row)

    plot_comparison(all_results, args.out_dir)

    # Save raw numbers
    json_out = {}
    for ds, mdict in all_results.items():
        json_out[ds] = {}
        for m, rlist in mdict.items():
            json_out[ds][m] = [
                {k: float(v) if isinstance(v, (float, int, np.floating))
                 else v for k, v in r.items()}
                for r in rlist
            ]
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()