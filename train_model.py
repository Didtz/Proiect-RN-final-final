"""
Train the plant identification model
Automatically detects all plants in training_data folder and trains on them
"""

import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 0.001,
    'validation_split': 0.2,
}

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('training_data', exist_ok=True)


def get_plant_classes_from_training_data(training_data_path='training_data'):
    """Dynamically get all plant classes from training_data folder"""
    try:
        path = Path(training_data_path)
        if path.exists():
            classes = sorted([d.name for d in path.iterdir() if d.is_dir()])
            logger.info(f"Found {len(classes)} plant classes: {classes}")
            return classes
        else:
            logger.error(f"Training data folder not found: {training_data_path}")
            return []
    except Exception as e:
        logger.error(f"Error reading training data: {e}")
        return []


class PlantModelTrainer:
    """Train plant identification model on all available plant classes"""

    def __init__(self):
        self.model = None
        self.class_names = get_plant_classes_from_training_data()
        self.num_classes = len(self.class_names)
        self.history = None
        
        if not self.class_names:
            raise ValueError("No plant classes found in training_data folder!")
        
        logger.info(f"Training on {self.num_classes} plant classes: {self.class_names}")

    def build_model(self):
        """Build transfer learning model using MobileNetV2"""
        logger.info("Building model...")
        
        base_model = MobileNetV2(
            input_shape=(*CONFIG['image_size'], 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze base model layers
        base_model.trainable = False

        # Build custom model
        model = keras.Sequential([
            layers.Input(shape=(*CONFIG['image_size'], 3)),
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        optimizer = Adam(learning_rate=CONFIG['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        logger.info("Model built successfully")
        return model

    def load_local_training_data(self):
        """Load images from local training_data folder using ImageDataGenerator"""
        logger.info("Loading images from local training_data folder...")
        
        # Image data generator for training with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=CONFIG['validation_split']
        )
        
        # Load training data
        self.train_data = train_datagen.flow_from_directory(
            'training_data',
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        self.validation_data = train_datagen.flow_from_directory(
            'training_data',
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        logger.info(f"Training samples: {self.train_data.samples}")
        logger.info(f"Validation samples: {self.validation_data.samples}")
        logger.info(f"Classes: {list(self.train_data.class_indices.keys())}")
        
        # Verify class names match
        loaded_classes = list(self.train_data.class_indices.keys())
        self.class_names = sorted(loaded_classes)  # Use actual loaded classes
        logger.info(f"Loaded {len(self.class_names)} classes: {self.class_names}")

    def train(self):
        """Train the model on local training data"""
        if self.model is None:
            self.build_model()

        logger.info("Loading training data from local folder...")
        self.load_local_training_data()

        logger.info(f"Starting training for {CONFIG['epochs']} epochs...")
        logger.info(f"Training on {len(self.class_names)} plant classes")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            self.train_data,
            steps_per_epoch=self.train_data.samples // CONFIG['batch_size'],
            validation_data=self.validation_data,
            validation_steps=self.validation_data.samples // CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")
        return self.history

    def evaluate(self):
        """Evaluate the model on validation set"""
        if self.model is None:
            logger.error("Model not trained. Please train first.")
            return None

        logger.info("Evaluating model on validation set...")
        
        # Evaluate
        loss, accuracy = self.model.evaluate(self.validation_data, verbose=1)
        logger.info(f"Validation Loss: {loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}")

        return {'loss': loss, 'accuracy': accuracy}

    def save_model(self, model_path='models/plant_model.h5'):
        """Save the trained model"""
        if self.model is None:
            logger.error("No model to save")
            return False

        try:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save model metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'classes': self.class_names,
                'image_size': CONFIG['image_size'],
                'config': CONFIG
            }
            
            metadata_path = model_path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            logger.error("No training history to plot")
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=100)
        logger.info("Training history plot saved to models/training_history.png")
        plt.show()


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Plant Identification Model Training")
    logger.info("Using local training_data folder")
    logger.info("=" * 60)

    # Initialize trainer
    try:
        trainer = PlantModelTrainer()
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error("Please add plant images to training_data folder first!")
        logger.error("Structure: training_data/plantname/image1.jpg, image2.jpg, ...")
        return

    # Build model
    logger.info("\nStep 1: Building model architecture...")
    trainer.build_model()

    # Train model
    logger.info("\nStep 2: Training the model...")
    logger.info(f"Classes to train on: {trainer.class_names}")
    trainer.train()

    # Evaluate model
    logger.info("\nStep 3: Evaluating model...")
    trainer.evaluate()

    # Save model
    logger.info("\nStep 4: Saving model...")
    trainer.save_model('models/plant_model.h5')

    # Plot training history
    logger.info("\nStep 5: Generating training plots...")
    try:
        trainer.plot_history()
    except Exception as e:
        logger.warning(f"Could not plot history: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Training completed successfully!")
    logger.info(f"Model saved to: models/plant_model.h5")
    logger.info(f"Trained on {len(trainer.class_names)} plant classes")
    logger.info(f"Classes: {', '.join(trainer.class_names)}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
