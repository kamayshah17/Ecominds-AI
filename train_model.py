import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
train_dir = "train"
test_dir = "test"

img_size = (224, 224)
batch_size = 32

# ----------------------------
# Data Generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  # ✅ binary classification
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# ----------------------------
# Transfer Learning - MobileNetV2
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # ✅ binary output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # ✅ binary loss
              metrics=['accuracy'])

# ----------------------------
# Training
# ----------------------------
history = model.fit(train_gen, validation_data=test_gen, epochs=5)

# ----------------------------
# Save Model
# ----------------------------
model.save("waste_classifier_binary.h5")

# ----------------------------
# Plot Accuracy/Loss
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training Accuracy")
plt.show()
