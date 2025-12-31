import wandb
from tqdm.auto import tqdm

def train_batch(images, labels, model, optimizer, criterion, device):
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def train_log(loss, example_ct, epoch):
    loss_val = float(loss)
    # metric logging: it pushes the current loss and epoch to W&B.
    wandb.log({"epoch": epoch, "loss": loss_val}, step=example_ct)


    
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss_val:.3f}")

def train_model(model, loader, criterion, optimizer, config, device):
    """
    This tells W&B to "hook" into your model. Every 10 batches, 
    it will upload histograms of your gradients and weights. 
    This helps you see if your model is dying (gradients -> 0) or 
    exploding. Gradient watching
    """
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    example_ct = 0
    batch_ct = 0
    
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):
            loss = train_batch(images, labels, model, optimizer, criterion, device)
            example_ct += len(images)
            batch_ct += 1
            
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
