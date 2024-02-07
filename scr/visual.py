# visual.py
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()


def mask_to_rgba(mask, color="red", opacity=0.5):
    MASK_COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    assert color in MASK_COLORS
    assert mask.ndim == 3 or mask.ndim == 2

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones * opacity), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones * opacity), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones * opacity), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones * opacity), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones * opacity), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones * opacity), axis=-1)
    

def visualize_evaluation_results(image_tensor, pred1, pred2, gt1, gt2, results_text, threshold=0.5):
    """
    Visualizes the segmentation results by the model with thresholding and color overlay, alongside ground truth masks.
    
    Parameters:
    - image_tensor: A PyTorch tensor of the image to be visualized. 
    - pred1, pred2: Model predictions for breast area and density.
    - gt1, gt2: Ground truth masks for breast area and density.
    - threshold: The threshold for converting probability maps to binary masks.
    """
    
    # Apply threshold to convert predictions to binary masks
    pred1_binary = (pred1 > threshold).float().squeeze().cpu().numpy()
    pred2_binary = (pred2 > threshold).float().squeeze().cpu().numpy()

    # Convert ground truths to RGBA for visualization
    gt1_colored = mask_to_rgba(gt1.squeeze().cpu().numpy(), color="cyan")
    gt2_colored = mask_to_rgba(gt2.squeeze().cpu().numpy(), color="magenta")

    # Convert predictions to RGBA for visualization
    pred1_colored = mask_to_rgba(pred1_binary, color="green")
    pred2_colored = mask_to_rgba(pred2_binary, color="red")

    # Convert the original image tensor to PIL for consistent visualization
    input_pil = to_pil_image(image_tensor.cpu())

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    # Original Image
    axes[0, 0].imshow(input_pil, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Prediction Breast Area
    axes[0, 1].imshow(input_pil, cmap='gray')
    axes[0, 1].imshow(pred1_colored, interpolation='none')
    axes[0, 1].set_title('Prediction: Breast Area')
    axes[0, 1].axis('off')

    # Prediction Density
    axes[0, 2].imshow(input_pil, cmap='gray')
    axes[0, 2].imshow(pred2_colored, interpolation='none')
    axes[0, 2].set_title('Prediction: Density')
    axes[0, 2].axis('off')

    # Ground Truth Breast Area
    axes[1, 1].imshow(input_pil, cmap='gray')
    axes[1, 1].imshow(gt1_colored, interpolation='none')
    axes[1, 1].set_title('Ground Truth: Breast Area')
    axes[1, 1].axis('off')

    # Ground Truth Density
    axes[1, 2].imshow(input_pil, cmap='gray')
    axes[1, 2].imshow(gt2_colored, interpolation='none')
    axes[1, 2].set_title('Ground Truth: Density')
    axes[1, 2].axis('off')

    # Format and display results text
    axes[1, 0].clear()
    axes[1, 0].axis('off')
    results_lines = results_text.split('\t')  # Split the results text into separate lines

    if results_lines:
        results_lines = results_lines[1:]
    
    # Remove "dsc_plus_plus_" from the first three lines
    for i in range(min(3, len(results_lines))):
        results_lines[i] = results_lines[i].replace('dsc_plus_plus_', '')

    formatted_text = '\n'.join(results_lines)
    axes[1, 0].text(0.05, 0.75, formatted_text, ha='left', va='top', fontsize=12, wrap=True, transform=axes[1, 0].transAxes)

    plt.show()