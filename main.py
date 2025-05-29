import os
from flask import Flask, request, render_template, redirect, url_for # type: ignore
from werkzeug.utils import secure_filename
import numpy as np
import keras
import cv2

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MODEL_PATH = 'autoencoder.keras' 
ANOMALY_THRESHOLD_VALUE = 0.1200 
FEATURES_PER_VIDEO = 20 
RESNET_MODEL_NAME = 'ResNet50'
RESNET_INPUT_SHAPE = (224, 224, 3)
RESNET_FEATURE_DIM = 2048 


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


GLOBAL_RESNET_MODEL = None

def load_resnet_model():
    global GLOBAL_RESNET_MODEL
    if GLOBAL_RESNET_MODEL is None:
        try:
            print(f"Loading pre-trained {RESNET_MODEL_NAME} for feature extraction...")
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=RESNET_INPUT_SHAPE
            )
            
            GLOBAL_RESNET_MODEL = keras.Sequential([
                base_model,
                keras.layers.GlobalAveragePooling2D() 
            ])
            print(f"{RESNET_MODEL_NAME} loaded successfully.")
        except Exception as e:
            print(f"Error loading {RESNET_MODEL_NAME}: {e}")
            GLOBAL_RESNET_MODEL = None
    return GLOBAL_RESNET_MODEL

autoencoder = None 
try:
    print("Loading autoencoder model...")
    autoencoder = keras.models.load_model(MODEL_PATH) 
    print("Autoencoder model loaded successfully!")
    if load_resnet_model() is None:
        raise RuntimeError(f"Failed to load {RESNET_MODEL_NAME}.")
except Exception as e:
    print(f"Error initializing: {e}")
    print(f"Please ensure '{MODEL_PATH}' and internet access for ResNet weights are available.")
    autoencoder = None
    GLOBAL_RESNET_MODEL = None

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features_from_video(video_path, num_features=FEATURES_PER_VIDEO):
    if GLOBAL_RESNET_MODEL is None:
        print("Error: ResNet model not loaded for feature extraction.")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    features = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames.")
        cap.release()
        return None
    if total_frames < num_features:
        indices = list(range(total_frames))
        print(f"Warning: Video has only {total_frames} frames, less than {num_features} requested. Taking all available.")
    else:
        indices = np.linspace(0, total_frames - 1, num_features, dtype=int)
    frames_processed = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break        
        if i in indices:
            resized_frame = cv2.resize(frame, (RESNET_INPUT_SHAPE[0], RESNET_INPUT_SHAPE[1]))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.expand_dims(rgb_frame, axis=0)
            preprocessed_frame = keras.applications.resnet50.preprocess_input(rgb_frame)
            resnet_feature_vector = GLOBAL_RESNET_MODEL.predict(preprocessed_frame, verbose=0)[0]
            features.append(resnet_feature_vector)
            frames_processed += 1
    cap.release()
    if not features:
        print(f"Error: No features extracted from {video_path} or video was too short.")
        return None   
    if len(features) < num_features:
        padding_needed = num_features - len(features)
        features.extend([np.zeros(RESNET_FEATURE_DIM)] * padding_needed)
    return np.array(features)

def predict_video(video_path):
    if autoencoder is None or GLOBAL_RESNET_MODEL is None:
        return "Error: Model components not loaded."
    features = extract_features_from_video(video_path, num_features=FEATURES_PER_VIDEO)
    if features is None:
        return "Prediction failed: Could not extract features from video."
    reconstructions = autoencoder.predict(features, verbose=0)
    per_frame_reconstruction_errors = np.mean(np.square(features - reconstructions), axis=1) 
    video_reconstruction_score = np.mean(per_frame_reconstruction_errors) 
    print(f"DEBUG: Mean Per-Frame Reconstruction Error for {os.path.basename(video_path)}: {video_reconstruction_score:.4f}")
    if video_reconstruction_score > ANOMALY_THRESHOLD_VALUE:
        return "Real Video"
    else:
        return "Deepfake Video"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)        
        file = request.files['video']        
        if file.filename == '':
            return redirect(request.url)        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)            
            prediction_result = predict_video(filepath)           
            return render_template('result.html', prediction=prediction_result)
        else:
            return render_template('upload.html', message="Invalid file type. Allowed: mp4, avi, mov, mkv")    
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)    
    app.run(debug=True)
