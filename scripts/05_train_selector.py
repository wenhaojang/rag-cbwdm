from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml
from src.selector_dataset import build_training_examples, feature_names
from src.selector_model import FeatureMLPSelector, save_feature_selector_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feature MLP selector from CBWDM teacher data.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--posteriors", required=True, help="Training posterior JSONL path.")
    parser.add_argument("--teacher", required=True, help="Training teacher JSONL path.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="MLP hidden dimension.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--limit-states", type=int, default=None, help="Only use first N teacher states.")
    parser.add_argument(
        "--max-candidates-per-state",
        type=int,
        default=None,
        help="Maximum candidates per teacher state.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def selector_params(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    selector = config.get("selector", {})
    return {
        "epochs": args.epochs if args.epochs is not None else int(selector.get("epochs", 5)),
        "batch_size": args.batch_size if args.batch_size is not None else int(selector.get("batch_size", 64)),
        "lr": args.lr if args.lr is not None else float(selector.get("lr", 1e-4)),
        "hidden_dim": args.hidden_dim if args.hidden_dim is not None else int(selector.get("hidden_dim", 128)),
        "dropout": args.dropout if args.dropout is not None else float(selector.get("dropout", 0.1)),
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    params = selector_params(config, args)
    set_seed(params["seed"])

    X, y, metadata = build_training_examples(
        posterior_path=resolve_project_path(args.posteriors),
        teacher_path=resolve_project_path(args.teacher),
        max_candidates_per_state=args.max_candidates_per_state,
        limit_states=args.limit_states,
    )
    positives = int(np.sum(y == 1.0))
    negatives = int(np.sum(y == 0.0))
    if positives == 0 or negatives == 0:
        raise ValueError(f"Training requires both positive and negative examples, got pos={positives} neg={negatives}")
    print(f"[selector_train] examples={len(y)} feature_dim={X.shape[1]} positives={positives} negatives={negatives}")

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
    model = FeatureMLPSelector(input_dim=X.shape[1], hidden_dim=params["hidden_dim"], dropout=params["dropout"])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    model.train()
    for epoch in range(1, params["epochs"] + 1):
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x.float())
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_count += len(batch_y)
        avg_loss = total_loss / max(total_count, 1)
        print(f"[selector_train] epoch={epoch} avg_loss={avg_loss:.6f}")

    num_labels = (X.shape[1] - 10) // 5
    names = feature_names(num_labels) if num_labels > 0 and 5 * num_labels + 10 == X.shape[1] else None
    checkpoint_config = {
        "hidden_dim": params["hidden_dim"],
        "dropout": params["dropout"],
        "lr": params["lr"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "seed": params["seed"],
        "num_examples": int(len(y)),
        "num_metadata": int(len(metadata)),
    }
    save_feature_selector_checkpoint(
        resolve_project_path(args.output),
        model,
        feature_dim=X.shape[1],
        config_dict=checkpoint_config,
        feature_names=names,
    )
    print(f"[selector_train] saved={resolve_project_path(args.output)}")


if __name__ == "__main__":
    main()
