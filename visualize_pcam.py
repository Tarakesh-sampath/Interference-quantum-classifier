import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from src.data.pcam_loader import get_pcam_dataset

def show_images():
    """Downloads (if needed) and displays a few samples using torchvision."""
    print("⏳ Initializing PCAM dataset (this may start a download)...")
    
    try:
        # We start with the 'test' split because it's usually smaller for a quick check, 
        # but you can change it to 'train'.
        dataset = get_pcam_dataset(split='test', download=True)
        print(f"✅ Dataset ready! Total samples: {len(dataset)}")

        plt.figure(figsize=(10, 5))
        for i in range(2):
            image, label = dataset[i]
            
            # Convert CHW back to HWC for plotting
            img_plot = image.permute(1, 2, 0).numpy()
            
            plt.subplot(1, 2, i + 1)
            plt.imshow(img_plot)
            plt.title(f"Label: {'Malignant' if label == 1 else 'Benign'}")
            plt.axis('off')
        
        plt.tight_layout()
        print("🖼️ Displaying images...")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_images()
