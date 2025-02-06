import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Paths
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VALID_DIR = os.path.join(BASE_DIR, "Valid")
MODEL_SAVE_PATH = "models/potato_leaf_cnn.keras"

# Prepare data
def prepare_data():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.9, 1.1),
        horizontal_flip=True,
        fill_mode="nearest"
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,  # Reduced batch size for stability
        class_mode='categorical',
        shuffle=True
    )

    valid_data = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    return train_data, valid_data

# Build CNN Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),  # Adjusted dropout to reduce overfitting
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 output classes
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Comprehensive model evaluation
def evaluate_model(model, valid_data):
    y_true = valid_data.classes
    y_pred_proba = model.predict(valid_data)
    y_pred = np.argmax(y_pred_proba, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=valid_data.class_indices.keys(),
                yticklabels=valid_data.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save confusion matrix
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(valid_data.class_indices.keys())))

# Main training script
def main():
    train_data, valid_data = prepare_data()

    # Compute class weights properly
    class_labels = np.unique(train_data.classes)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=train_data.classes)
    class_weight_dict = dict(zip(class_labels, class_weights))

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,  # Reduced patience to avoid long training times
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    history = model.fit(
        train_data,
        epochs=20,  # Adjusted epochs for better convergence
        validation_data=valid_data,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr]
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    evaluate_model(model, valid_data)

if __name__ == "__main__":
    main()
