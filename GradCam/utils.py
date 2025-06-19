from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from .hook import LayerHook
from matplotlib.colors import LinearSegmentedColormap

def generate_grad_cam(model, image, extractor, target_class):
    model.zero_grad()
    output = model(image)
    class_score = output[0, target_class]
    class_score.backward()
    gradients = extractor.gradient
    feature_map = extractor.feature_map
    weights = gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feature_map).sum(dim=1, keepdim=True)
    cam = F.relu(cam)  # ReLU to remove negative values
    cam = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
    cam -= cam.min()
    cam /= torch.max(cam) # Normalize between 0 and 1
    return cam.squeeze().cpu().numpy()

def overlay_heatmap(image, heatmap):
    heatmap = Image.fromarray(np.uint8(plt.cm.viridis(heatmap) * 255)).convert("RGB")
    heatmap = heatmap.resize((28, 28), Image.LANCZOS)
    blended = Image.blend(image.convert("RGB"), heatmap, alpha=0.5)
    return blended

def show_grad_cam(model, original_image, image, target_layer, classes, model_type='global_net'):
    extractor = LayerHook(model, target_layer, model_type)  # Targeting the last conv layer
    # Get top-3 predicted classes and least probable class
    output = model(image)
    n_classes = 2
    top_classes = output.topk(n_classes).indices.cpu().numpy()[0]
    least_class = output.argmin().cpu().numpy()

    original_image = to_pil_image(original_image)
    # Generate heatmaps
    fig, axes = plt.subplots(1, 1+n_classes+1, figsize=(20, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Input Image")
    
    for i, class_idx in enumerate(np.concatenate([top_classes, [least_class]])):
        heatmap = generate_grad_cam(model, image, extractor, class_idx)
        # Compute mean gradient magnitude in the feature map
        grad_mag = extractor.gradient.abs().mean().item() if extractor.gradient is not None else float('nan')
        overlay = overlay_heatmap(original_image, heatmap)
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"Class {classes[class_idx]}\nMean grad: {grad_mag:.4f}")
    plt.show()
