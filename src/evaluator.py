import os
import torch
import torch.onnx
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate(model, loader, device, split="val"):
    """Calculates comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Move to CPU for Scikit-Learn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    metrics = {
        f"{split}_accuracy": accuracy,
        f"{split}_precision": precision,
        f"{split}_recall": recall,
        f"{split}_f1": f1
    }
    
    model.train()
    return metrics

def test_model(model, test_loader, config, device, run): # <--- Add 'run' argument
    print("🧪 Running final evaluation...")

    # ---- Evaluate ----
    metrics = evaluate(model, test_loader, device, split="test")
    
    # Use the passed 'run' object for logging
    run.log(metrics) 

    # Promote metrics to summary -> scalar values of metrics
    for k, v in metrics.items():
        run.summary[k] = v

    print(f"Final Test Metrics: {metrics}")

    # -------------------------
    # Export ONNX
    # -------------------------
    model.eval()
    os.makedirs("model", exist_ok=True)

    dummy_input = next(iter(test_loader))[0].to(device)
    
    # Fix: Use run.id from the passed object
    onnx_filename = f"mnist_{run.id}.onnx"
    onnx_path = os.path.join("model", onnx_filename)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
    )
    
    assert os.path.exists(onnx_path), f"File not found: {onnx_path}"
    
    #------------------------------
    #Build artifact metadata
    #------------------------------

    artifact_metadata = {
    "run_id": run.id,
    "architecture": config.architecture,
    }

    # Add test metrics
    artifact_metadata.update(metrics)
    print("📦 Artifact metadata test:", artifact_metadata)

    # Add config
    artifact_metadata.update(config)
    print("📦 Artifact metadata config:", artifact_metadata)

    #add validation metrics from run.summary
    artifact_metadata.update(run.summary)
    print("📦 Artifact metadata val:", artifact_metadata)

    # -------------------------
    # Candidate model artifact
    # -------------------------
    candidate_artifact = wandb.Artifact(
        name="mnist-cnn-candidate",
        type="model",
        metadata=artifact_metadata,
    )

    candidate_artifact.add_file(os.path.abspath(onnx_path))

    print("📦 Uploading candidate model artifact...")
    
    run.log_artifact(candidate_artifact)
    print("✅ Candidate model logged")

    # -------------------------
    # Best model promotion (by validation accuracy)
    # -------------------------
    # get the current best validation accuracy
    api = wandb.Api()

    best_val = None
    try:
        best_artifact = api.artifact(
            "kokrhui-ong-tng-digital/pytorch-sqlite-sweeps/mnist-cnn-best:best"
        )
        best_val = best_artifact.metadata.get("best_val_accuracy")
    except wandb.errors.CommError:
        print("ℹ️ No existing best model found")

    current_val = run.summary.get("val_accuracy")

    if best_val is None or current_val > best_val:
        print("🏆 New best model found")

        best_artifact = wandb.Artifact(
            name="mnist-cnn-best",
            type="model",
            description="Best model selected by validation accuracy",
            metadata={
                **artifact_metadata,
                "selection_metric": "val_accuracy",
                "best_val_accuracy": current_val,
                "source_run_id": run.id,
            },
        )

        best_artifact.add_file(os.path.abspath(onnx_path))
        run.log_artifact(best_artifact, aliases=["best"])

    else:
        print(
            f"ℹ️ Model did not beat best "
            f"({current_val:.4f} <= {best_val:.4f})"
        )
