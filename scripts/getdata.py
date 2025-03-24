import tensorflow as tf
import numpy as np
import os
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
import gc

#  Reduce Image Size & Use float16 to Save Memory
IMG_SIZE = (128, 128)  # Reduced from (224, 224)
BATCH_SIZE = 16
DATA_TYPE = np.float16  # Changed from float32

#  Function: Load images efficiently
def load_data(real_dir, deepfake_dir):

    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    deepfake_paths = [os.path.join(deepfake_dir, f) for f in os.listdir(deepfake_dir) if f.endswith(('.jpg', '.png'))]

    real_labels = np.zeros(len(real_paths), dtype=int)  # 0 = Real
    deepfake_labels = np.ones(len(deepfake_paths), dtype=int)  # 1 = Deepfake

    image_paths = np.array(real_paths + deepfake_paths)
    labels = np.concatenate([real_labels, deepfake_labels])

    # Shuffle Data for Better Class Mixing
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    image_paths, labels = image_paths[indices], labels[indices]

    def parse_image(image_path, label):
        """Load, decode, and resize images with optimized memory usage."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)  # Smaller image
        image = tf.cast(image, tf.float16) / 255.0  # Normalize & Convert to float16
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset

#  Function: Balance data with SMOTE (Batch Processing)
def balance_data_with_smote(dataset, max_pca_components=50):
    """Balances data using batch-wise SMOTE and saves data incrementally to disk to prevent RAM overflow."""
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    balanced_data_path = 'data/processed/balanced_data.npy'
    balanced_labels_path = 'data/processed/balanced_labels.npy'

    #  Use Incremental Saving to Avoid RAM Overflow
    data_saved = []
    labels_saved = []

    for i, (images, labels) in enumerate(dataset):
        try:
            data_np = images.numpy().astype(DATA_TYPE)  # Convert to float16
            labels_np = labels.numpy().astype(int)

            #  Ensure Each Batch Has Both Classes Before SMOTE
            unique_classes = np.unique(labels_np)
            if len(unique_classes) < 2:
                print(f" Skipping batch {i} - Only one class found: {unique_classes}")
                continue

            # Flatten Images for SMOTE
            data_reshaped = np.array([img.flatten() for img in data_np])

            # Apply PCA for Dimensionality Reduction
            pca = PCA(n_components=min(len(data_np), max_pca_components))
            data_pca = pca.fit_transform(data_reshaped)

            # Apply SMOTE
            balanced_data, balanced_labels = ros.fit_resample(data_pca, labels_np)

            # Inverse Transform PCA & Reshape
            balanced_data = pca.inverse_transform(balanced_data)
            balanced_data = balanced_data.reshape(-1, *IMG_SIZE, 3)  # Reshape back

            # Save Data Incrementally to Avoid Memory Overflow
            data_saved.extend(balanced_data)
            labels_saved.extend(balanced_labels)

            if len(data_saved) >= 2000:  # Save every 2000 samples to prevent RAM overflow
                np.save(balanced_data_path, np.array(data_saved, dtype=DATA_TYPE))
                np.save(balanced_labels_path, np.array(labels_saved, dtype=int))
                print(f" Saved {len(data_saved)} samples to disk.")
                data_saved, labels_saved = [], []  # Clear memory
            
            gc.collect()  # Free unused memory

        except Exception as e:
            print(f" Error processing batch {i}: {e}")

    # Final Save
    if data_saved:
        np.save(balanced_data_path, np.array(data_saved, dtype=DATA_TYPE))
        np.save(balanced_labels_path, np.array(labels_saved, dtype=int))
        print(f" Final save: {len(data_saved)} samples.")

#  Run Data Pipeline
if __name__ == "__main__":
    real_dir = 'data/processed/real'
    deepfake_dir = 'data/processed/deepfake'

    try:
        dataset = load_data(real_dir, deepfake_dir)
        balance_data_with_smote(dataset)
    except Exception as e:
        print(f"Critical Error: {e}")
