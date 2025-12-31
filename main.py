import wandb
import torch.nn as nn
import torch.optim as torch_optim
from torch.utils.data import DataLoader

from config import PARAMS
from src.utils import set_seed, get_device
from src.dataset import MNISTDatabaseDataset
from src.model import ConvNet
from src.trainer import train_model
from src.evaluator import test_model

def make(config, device):
    # Initialize SQL Datasets
    train_dataset = MNISTDatabaseDataset(config.db_path, split='train')
    test_dataset = MNISTDatabaseDataset(config.db_path, split='test')

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch_optim.Adam(
        model.parameters(), lr=config.learning_rate
    )

    return model, train_loader, test_loader, criterion, optimizer

def model_pipeline(hyperparameters):
    device = get_device()
    set_seed()

    # Initialize W&B
    with wandb.init(project="pytorch-sqlite-demo", config=hyperparameters):
        config = wandb.config # logs hyperparameters
        
        # Build components
        model, train_loader, test_loader, criterion, optimizer = make(config, device)
        print(f"Model architecture:\n{model}")
        
        # Train
        train_model(model, train_loader, criterion, optimizer, config, device)
        
        # Test
        test_model(model, test_loader, device)
    
    return model

if __name__ == "__main__":
    model_pipeline(PARAMS)