import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import RandomOverSampler

#  Load Data
X = np.load('data/processed/balanced_data.npy')  # (samples, 128, 128, 3)
y = np.load('data/processed/balanced_labels.npy')  # (samples,)

#  Normalize Data
X = X / 255.0  # Scale pixel values

#  Convert Image Depth to CV_8U for OpenCV Processing
X = (X * 255).astype(np.uint8)  # Normalize and convert to 8-bit

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#  Handle Class Imbalance with Oversampling
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train_resampled = X_train_resampled.reshape(-1, 128, 128, 3)

#  Compute Class Weights Dynamically
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Computed Class Weights:", class_weights_dict)

# Data Augmentation (More Variations)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],  # Adjust brightness
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train_resampled)

# Load Pretrained VGG16 Model (Feature Extractor)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers[:-4]:  # Unfreeze last 4 layers
    layer.trainable = False

#  Define Fine-Tuned Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

#Focal Loss Function (Improved)
def focal_loss(alpha=0.5, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        loss = alpha * (1 - pt) ** gamma * bce
        return tf.reduce_mean(loss)
    return loss

#  Learning Rate Scheduler
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

#  Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#  Compile Model with Optimized Learning Rate
model.compile(optimizer=optimizers.Adam(learning_rate=3e-5),  # Lower LR
              loss=focal_loss(alpha=0.5, gamma=2.0),
              metrics=['accuracy'])

#  Train Model with Callbacks
history = model.fit(datagen.flow(X_train_resampled, y_train_resampled, batch_size=16),
                    validation_data=(X_val, y_val),
                    epochs=30,
                    class_weight=class_weights_dict,
                    callbacks=[lr_schedule, early_stopping])

#  Predict Probabilities
y_pred_proba = model.predict(X_val)

#  Find Best Decision Threshold
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
best_threshold = thresholds[np.argmax(precisions * recalls)]
print("Optimal Decision Threshold:", best_threshold)

#  Adjust Predictions Using Best Threshold
y_pred = (y_pred_proba > best_threshold).astype(int)



#  Save Model
model.save("deepfake_detector.h5")
print("✅ Model saved successfully!")
