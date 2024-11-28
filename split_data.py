import os
import random
import shutil

# Path to your dataset folders
images_path = 'D:\Telegram Desktop\YOLO\dataset\images'

# List all image files
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle the image files randomly
random.shuffle(image_files)

# Split ratio (80% for training, 20% for validation)
train_size = int(0.8 * len(image_files))
val_size = len(image_files) - train_size

# Create directories for train and val sets (if not exist)
os.makedirs(os.path.join(images_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'val'), exist_ok=True)


# Move files to train and val directories
for i, file in enumerate(image_files):
    # Path to the image and label
    image_file = os.path.join(images_path, file)
    # label_file = os.path.join(labels_path, file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
    
    # Determine if it's for training or validation
    if i < train_size:
        shutil.move(image_file, os.path.join(images_path, 'train', file))
        # shutil.move(label_file, os.path.join(labels_path, 'train', file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')))
    else:
        shutil.move(image_file, os.path.join(images_path, 'val', file))
        # shutil.move(label_file, os.path.join(labels_path, 'val', file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')))

print(f"Dataset split complete. {train_size} images in training set, {val_size} in validation set.")
