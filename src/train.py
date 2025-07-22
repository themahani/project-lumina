import os

import mlflow
import torch
import torch_geometric
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import our model classes dynamically
from models import GraphSAGE


# Helper functions for training and testing (same as before)
def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    # We only compute loss on the training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(data, model):
    model.eval()
    out = model(data)
    # Get predictions for validation and test sets
    y_pred_val = out[data.val_mask].argmax(dim=1)
    y_true_val = data.y[data.val_mask]

    y_pred_test = out[data.test_mask].argmax(dim=1)
    y_true_test = data.y[data.test_mask]

    # Calculate metrics
    val_f1 = f1_score(y_true_val.cpu(), y_pred_val.cpu(), average="binary")
    test_f1 = f1_score(y_true_test.cpu(), y_pred_test.cpu(), average="binary")

    val_auc = roc_auc_score(y_true_val.cpu(), out[data.val_mask][:, 1].cpu())
    test_auc = roc_auc_score(y_true_test.cpu(), out[data.test_mask][:, 1].cpu())

    return val_f1, test_f1, val_auc, test_auc


def main():
    # Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Data and Create Splits
    # Loading geometric data, setting weights_only to false, assuming dataset is safe
    print("Loading processed graph data...")
    data = torch.load(config["data"]["processed_path"], weights_only=False)

    labeled_indices = (data.y != -1).nonzero(as_tuple=False).view(-1)
    train_indices, test_indices = train_test_split(
        labeled_indices,
        test_size=config["training"]["test_split_ratio"],
        random_state=42,
        stratify=data.y[labeled_indices],
    )
    val_indices, test_indices = train_test_split(
        test_indices,
        test_size=config["training"]["val_split_ratio"],
        random_state=42,
        stratify=data.y[test_indices],
    )
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    # Setup MLflow Experiment
    mlflow.set_experiment(config["mlflow_experiment_name"])

    with mlflow.start_run() as run:
        print(f"Starting run {run.info.run_id}...")
        mlflow.log_params(config["model"]["params"])
        mlflow.log_params(config["training"])
        mlflow.log_param("model_name", config["model"]["name"])
        mlflow.log_dict(config, "config.yaml")

        # Dynamically Select and Initialize Model
        model_class = globals()[config["model"]["name"]]
        model = model_class(
            in_channels=data.num_node_features,
            **config["model"]["params"],  # Unpack params like hidden_channels
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        # Training Loop
        print("Starting training...")
        best_val_f1 = 0
        for epoch in range(1, config["training"]["epochs"] + 1):
            loss = train(data, model, optimizer, criterion)
            val_f1, test_f1, val_auc, test_auc = test(data, model)

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            mlflow.log_metric("test_f1", test_f1, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            mlflow.log_metric("test_auc", test_auc, step=epoch)

            # Save the best model based on validation F1-score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                mlflow.set_tag("best_epoch", epoch)
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=config["model_registry_name"],
                )

        print("Training finished.")
        print(f"Model logged and registered as '{config['model_registry_name']}'")


if __name__ == "__main__":
    main()
