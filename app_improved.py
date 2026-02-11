"""
Plant Identification Web Application with Comparison Photos
Flask-based web interface for plant identification using deep learning
Includes comparison photos from the training dataset
"""

import os
import json
import random
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'models/plant_model.h5'
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
TRAINING_DATA_FOLDER = 'training_data'  # Folder with plant training images
HOUSE_PLANTS_JSON = 'plant_images/house_plants.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)


def load_house_plants_database():
    """Load plant information from house_plants.json"""
    try:
        if os.path.exists(HOUSE_PLANTS_JSON):
            with open(HOUSE_PLANTS_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load house plants database: {e}")
    return []


def get_all_plant_classes():
    """Dynamically get all plant classes from training_data folder"""
    try:
        training_path = Path(TRAINING_DATA_FOLDER)
        if training_path.exists():
            classes = sorted([d.name for d in training_path.iterdir() if d.is_dir()])
            logger.info(f"Found {len(classes)} plant classes: {classes}")
            return classes
    except Exception as e:
        logger.warning(f"Could not read training data folder: {e}")
    return []


def build_dynamic_plant_database():
    """Build plant database from house_plants.json and training data"""
    database = {}
    house_plants = load_house_plants_database()
    
    # Create a lookup by common name (case-insensitive)
    plant_lookup = {}
    for plant in house_plants:
        if plant.get('common'):
            for common_name in plant['common']:
                key = common_name.lower().strip()
                plant_lookup[key] = {
                    'scientific_name': plant.get('latin', 'Unknown'),
                    'family': plant.get('family', ''),
                    'description': plant.get('use', ['Unknown'])[0] if plant.get('use') else 'Unknown',
                    'watering': plant.get('watering', 'Regular watering'),
                    'light': plant.get('ideallight', 'Bright light'),
                    'difficulty': get_difficulty_from_watering(plant.get('watering', ''))
                }
    
    # Get all training classes
    class_names = get_all_plant_classes()
    
    for class_name in class_names:
        # Try to find plant info
        if class_name.lower() in plant_lookup:
            database[class_name] = plant_lookup[class_name.lower()]
            database[class_name]['common_name'] = class_name.capitalize()
        else:
            # Create basic entry for plants not in database
            database[class_name] = {
                'common_name': class_name.capitalize(),
                'scientific_name': f'{class_name.capitalize()} spp.',
                'description': f'{class_name.capitalize()} plant',
                'watering': 'Regular watering',
                'light': 'Bright light',
                'difficulty': 'Intermediate'
            }
    
    return database, class_names


def get_difficulty_from_watering(watering_text):
    """Estimate difficulty from watering instructions"""
    watering_lower = watering_text.lower()
    if 'moist' in watering_lower or 'frequently' in watering_lower or 'constantly' in watering_lower:
        return 'Advanced'
    elif 'dry' in watering_lower or 'sparse' in watering_lower or 'minimal' in watering_lower:
        return 'Easy'
    else:
        return 'Intermediate'


# Initialize dynamic plant database
PLANT_DATABASE, class_names = build_dynamic_plant_database()
logger.info(f"Loaded {len(PLANT_DATABASE)} plants for identification")



# Global model variable
model = None
image_size = (224, 224)
model_ready = False


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_comparison_image(plant_name):
    """
    Get a comparison image for the identified plant from training data
    Returns: base64 encoded image or None
    """
    try:
        plant_folder = Path(TRAINING_DATA_FOLDER) / plant_name
        if plant_folder.exists():
            images = list(plant_folder.glob('*.jpg')) + list(plant_folder.glob('*.png'))
            if images:
                # Pick a random image from the plant folder
                image_path = random.choice(images)
                with open(image_path, 'rb') as f:
                    import base64
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    # Determine image type
                    img_type = 'jpeg' if str(image_path).lower().endswith('.jpg') else 'png'
                    return f"data:image/{img_type};base64,{encoded}"
    except Exception as e:
        logger.warning(f"Could not get comparison image for {plant_name}: {e}")
    return None


def load_model():
    """Load the trained model"""
    global model, model_ready, class_names
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            # Validate model output classes against known class names
            try:
                output_classes = int(model.output_shape[-1])
                if len(class_names) != output_classes:
                    logger.warning(
                        "Model class count mismatch. Model outputs %s classes, but %s classes were found."
                        " Please retrain the model or update training_data.",
                        output_classes,
                        len(class_names)
                    )
                    model_ready = False
                else:
                    model_ready = True
            except Exception as e:
                logger.warning(f"Could not validate model output shape: {e}")
                model_ready = False

            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found. Training required.")
            model = None
            model_ready = False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        model_ready = False


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', plant_count=len(PLANT_DATABASE))


@app.route('/identify')
def identify():
    """Plant identification page"""
    return render_template('identify.html')


@app.route('/database')
def database():
    """Plant database page"""
    return render_template('database.html', plants=PLANT_DATABASE)


@app.route('/guide')
def guide():
    """Care guide page"""
    return render_template('guide.html')


@app.route('/training')
def training():
    """Training status page"""
    return render_template('training.html', model_exists=model is not None)


# API Routes
@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict plant from uploaded image"""
    if model is None or not model_ready:
        return jsonify({'error': 'Model not trained yet. Please train the model first.', 'success': False}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided', 'success': False}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'success': False}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif', 'success': False}), 400

    filepath = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        img_array = preprocess_image(filepath)
        if img_array is None:
            return jsonify({'error': 'Failed to process image', 'success': False}), 400

        predictions = model.predict(img_array, verbose=0)
        confidence = float(np.max(predictions[0]))
        predicted_class = class_names[np.argmax(predictions[0])]

        if confidence < 0.3:
            return jsonify({
                'success': False,
                'message': 'Cannot confidently identify the plant',
                'confidence': confidence
            })

        plant_info = PLANT_DATABASE.get(predicted_class, {})
        
        # Get comparison image from training data
        comparison_image = get_comparison_image(predicted_class)

        return jsonify({
            'success': True,
            'plant': predicted_class,
            'common_name': predicted_class.capitalize(),
            'confidence': confidence,
            'info': plant_info,
            'comparison_image': comparison_image
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500
    finally:
        # Clean up uploaded file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/plants')
def get_plants():
    """Get all plants in database"""
    return jsonify(PLANT_DATABASE)


@app.route('/api/plant/<plant_name>')
def get_plant(plant_name):
    """Get specific plant information"""
    plant = PLANT_DATABASE.get(plant_name.lower())
    if plant:
        return jsonify({plant_name: plant})
    return jsonify({'error': 'Plant not found'}), 404


@app.route('/api/model/status')
def model_status():
    """Get model training status"""
    return jsonify({
        'trained': model is not None,
        'model_path': MODEL_PATH,
        'classes': class_names,
        'num_classes': len(class_names)
    })


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Trigger model training"""
    return jsonify({
        'message': 'Training initiated. Check the training page or terminal for progress.',
        'note': 'Run train_model.py from the command line to train the model.'
    }), 202


if __name__ == '__main__':
    load_model()
    app.run(debug=False, port=5000, host='0.0.0.0')
