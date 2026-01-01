# Hyperparameter configuration

PARAMS = {
    # Training
    "epochs": 1,
    "batch_size": 128,
    "learning_rate": 0.005,

    # Model
    "classes": 10,
    "kernels": [16, 32],
    "architecture": "CNN",

    # Data (LOGICAL reference, not path)
    "dataset_artifact": "mnist-sqlite-data:latest",

    # Metadata
    "dataset": "MNIST",
    "model_path": "model/mnist.onnx"
}
