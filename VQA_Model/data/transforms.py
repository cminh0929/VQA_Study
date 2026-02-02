"""
Image transformations for VQA
Preprocessing for both pretrained and from-scratch models
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, use_pretrained: bool = True):
    """
    Get training transforms
    
    Args:
        image_size: Target image size (224 for ResNet50/VGG16)
        use_pretrained: Whether using pretrained model (affects normalization)
    
    Returns:
        torchvision.transforms.Compose
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ]
    
    # Add normalization for pretrained models
    if use_pretrained:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    else:
        # Simple normalization for from-scratch models
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int = 224, use_pretrained: bool = True):
    """
    Get validation/test transforms (no augmentation)
    
    Args:
        image_size: Target image size
        use_pretrained: Whether using pretrained model
    
    Returns:
        torchvision.transforms.Compose
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    # Add normalization
    if use_pretrained:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    else:
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
    
    return transforms.Compose(transform_list)


def denormalize_image(tensor: torch.Tensor, use_pretrained: bool = True):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        use_pretrained: Whether pretrained normalization was used
    
    Returns:
        Denormalized tensor
    """
    if use_pretrained:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    else:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


class ImageTransform:
    """Wrapper class for image transformations"""
    
    def __init__(self, image_size: int = 224, use_pretrained: bool = True):
        """
        Args:
            image_size: Target image size
            use_pretrained: Whether using pretrained model
        """
        self.image_size = image_size
        self.use_pretrained = use_pretrained
        
        self.train_transform = get_train_transforms(image_size, use_pretrained)
        self.val_transform = get_val_transforms(image_size, use_pretrained)
    
    def __call__(self, image: Image.Image, is_training: bool = True):
        """
        Apply transformation to image
        
        Args:
            image: PIL Image
            is_training: Whether in training mode
        
        Returns:
            Transformed tensor
        """
        if is_training:
            return self.train_transform(image)
        else:
            return self.val_transform(image)


# Example usage
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load sample image
    img_path = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\images\000000000025.jpg"
    img = Image.open(img_path).convert('RGB')
    
    print(f"Original image size: {img.size}")
    
    # Test pretrained transforms
    print("\nTesting pretrained transforms...")
    transform_pretrained = ImageTransform(image_size=224, use_pretrained=True)
    img_tensor_pretrained = transform_pretrained(img, is_training=False)
    print(f"  Transformed shape: {img_tensor_pretrained.shape}")
    print(f"  Value range: [{img_tensor_pretrained.min():.3f}, {img_tensor_pretrained.max():.3f}]")
    
    # Test from-scratch transforms
    print("\nTesting from-scratch transforms...")
    transform_scratch = ImageTransform(image_size=224, use_pretrained=False)
    img_tensor_scratch = transform_scratch(img, is_training=False)
    print(f"  Transformed shape: {img_tensor_scratch.shape}")
    print(f"  Value range: [{img_tensor_scratch.min():.3f}, {img_tensor_scratch.max():.3f}]")
    
    # Test denormalization
    print("\nTesting denormalization...")
    img_denorm = denormalize_image(img_tensor_pretrained, use_pretrained=True)
    print(f"  Denormalized range: [{img_denorm.min():.3f}, {img_denorm.max():.3f}]")
    
    print("\n✓ All transforms working correctly!")
