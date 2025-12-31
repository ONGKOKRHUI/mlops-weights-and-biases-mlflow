import torch
import torch.onnx
import wandb

def test_model(model, test_loader, device):
    model.eval()
    
    # Run inference
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"Accuracy of the model on the {total} test images: {accuracy:%}")
        wandb.log({"test_accuracy": accuracy})
        
    # Export to ONNX
    # We grab one batch to trace the graph
    images_batch = next(iter(test_loader))[0].to(device)
    torch.onnx.export(model, images_batch, "model.onnx")
    wandb.save("model.onnx")