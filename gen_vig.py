import torch
from PIL import Image
import random
from torchvision.transforms.functional import to_pil_image, to_tensor


def create_vignette_mask(image_shape, center=None, radius=None, strength=None):
    height, width = image_shape
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = random.randint(max(width, height) // 1.2, max(width, height))
    if strength is None:
        strength = random.uniform(0.8, 1.2)
    y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing='ij')
    y = y.float()
    x = x.float()
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = torch.clamp(1 - strength * (distance / radius), 0, 1)
    return mask


def apply_vignetting(inp, mask):
    return inp * mask.unsqueeze(0)


# Load an image and convert it to a PyTorch tensor
image_path = './input.png'
image = Image.open(image_path).convert('RGB')
image_tensor = to_tensor(image)

# Create a vignette mask and apply it to the image tensor
vignette_mask = create_vignette_mask(image_tensor.shape[1:])
vignetted_image_tensor = apply_vignetting(image_tensor, vignette_mask)

# Convert the vignetted image tensor back to a PIL Image and save it
vignetted_image = to_pil_image(vignetted_image_tensor.cpu())
vignetted_image.save('./output.png')
