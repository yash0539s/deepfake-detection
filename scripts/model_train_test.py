#  Import Required Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV  # For Platt scaling
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.callbacks import EarlyStopping

#  Load Data
X = np.load("D:\\deep_fake_hacthon\\data\\processed\\augmented_data.npy")  # Shape: (samples, 128, 128, 3)
y = np.load("D:\\deep_fake_hacthon\\data\\processed\\augmented_labels.npy")  # Shape: (samples,)

#  Normalize Data
X = (X / 255.0).astype(np.float32)  # Use float32 for better precision

#  Train-Test Split (80-20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#  Apply Oversampling Only to Training Data
ros = RandomOverSampler()
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Convert to 2D for oversampling
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)
X_train_resampled = X_train_resampled.reshape(-1, 128, 128, 3)  # Convert back to 4D

#  Data Augmentation (Applied to Training Set)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode="nearest"
)
datagen.fit(X_train_resampled)

#  Define the Model Using VGG16 Backbone
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base layers
base_model.trainable = False

# Build the full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

#  Custom Focal Loss (Handles Class Imbalance)
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        loss = alpha * (1 - pt) ** gamma * bce
        return tf.reduce_mean(loss)
    return loss

#  Compile Model with Optimized Learning Rate and Adam Optimizer
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss=focal_loss(alpha=0.5, gamma=2.0),
              metrics=['accuracy'])

#  Early Stopping Callback to Prevent Overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#  Train Model with Data Augmentation and Early Stopping
history = model.fit(datagen.flow(X_train_resampled, y_train_resampled, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=35,
                    callbacks=[early_stopping])

#  Predict Probabilities
y_pred_proba = model.predict(X_val)

#  Calibrate Probabilities Using Platt Scaling

#  Find Best Decision Threshold
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
best_threshold = thresholds[np.argmax(precisions * recalls)]
print("Optimal Decision Threshold:", best_threshold)

#  Adjust Predictions Using Best Threshold
y_pred = (y_pred_proba > best_threshold).astype(int)

#  Evaluate Model
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

#  Save Model
model.save("deepfake_detector_v8.h5")
print(" Model saved successfully!")

