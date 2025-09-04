import json
import os
import sys
import time
import pickle
import base64
import tempfile
import warnings
from typing import Dict, Any

import numpy as np
import pika
import psycopg2
from huggingface_hub import hf_hub_download

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# New imports for image and audio processing
from PIL import Image
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

# Try to import YOLO with fallback
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("[worker] YOLO imported successfully", flush=True)
except ImportError as e:
    print(f"[worker] YOLO import failed: {e}", flush=True)
    print("[worker] YOLO functionality will be disabled", flush=True)
    YOLO_AVAILABLE = False

# ---- Config ----
AMQP_URL = os.getenv("AMQP_URL") or "amqp://{u}:{p}@{h}:{port}/%2f".format(
    u=os.getenv("RABBITMQ_USER", "demo"),
    p=os.getenv("RABBITMQ_PASSWORD", "demo"),
    h=os.getenv("RABBITMQ_HOST", "rabbitmq"),
    port=os.getenv("RABBITMQ_PORT", "5672"),
)
QUEUE_NAME = os.getenv("AMQP_QUEUE", "inference_requests")

DB_URL = os.getenv("DATABASE_URL") or "postgresql://{u}:{p}@{h}:{port}/{db}".format(
    u=os.getenv("DB_USER", "demo"),
    p=os.getenv("DB_PASSWORD", "demo"),
    h=os.getenv("DB_HOST", "postgres"),
    port=os.getenv("DB_PORT", "5432"),
    db=os.getenv("DB_NAME", "demo"),
)

# Hugging Face repos + files
IRIS_REPO = os.getenv(
    "MODEL_IRIS_REPO",
    "skops-tests/iris-sklearn-1.0-logistic_regression-without-config",
)
IRIS_FILE = os.getenv("MODEL_IRIS_FILE", "skops-ehiqc2lv.pkl")  # .pkl
DIAB_REPO = os.getenv(
    "MODEL_DIAB_REPO",
    "skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-with-config-pickle",
)
DIAB_FILE = os.getenv("MODEL_DIAB_FILE", "skops-xcxb87en.pkl")  # .pkl

# New models
YOLO_REPO = os.getenv("MODEL_YOLO_REPO", "ultralytics/yolov8n")
# Audio classifier is lightweight - no external repo needed

HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_HOME = os.getenv("HF_HOME", "/tmp/hf-cache")
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs("/tmp/uploads", exist_ok=True)  # For temporary file storage

def load_pickle_from_hf(repo: str, filename: str):
    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=HF_TOKEN,
        cache_dir=HF_HOME,
        local_dir="/tmp/models",
    )
    try:
        with open(local_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[worker] Failed to load model {repo}/{filename}: {e}", flush=True)
        print(f"[worker] This might be due to scikit-learn version compatibility", flush=True)
        # Re-raise the error so the application fails fast rather than silently continuing
        raise

def load_yolo_model(repo: str):
    """Load YOLO model from HuggingFace"""
    if not YOLO_AVAILABLE:
        print(f"[worker] YOLO not available, creating dummy model", flush=True)
        return DummyYOLOModel()
    
    print(f"[worker] loading YOLO model from {repo}...", flush=True)
    try:
        # Download model to local cache, YOLO will handle the rest
        model = YOLO(f"hf://{repo}")
        return model
    except Exception as e:
        print(f"[worker] Failed to load YOLO model: {e}", flush=True)
        print(f"[worker] Using dummy YOLO model instead", flush=True)
        return DummyYOLOModel()

class SimpleImageClassifier:
    """Simple image classifier using PIL and basic statistics"""
    def __init__(self):
        self.classes = ["bright_image", "dark_image", "colorful_image", "grayscale_image"]
    
    def analyze_image(self, image_path):
        """Analyze image using basic PIL operations"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get basic image statistics
                width, height = img.size
                pixels = list(img.getdata())
                
                # Calculate color statistics
                r_values = [p[0] for p in pixels]
                g_values = [p[1] for p in pixels]
                b_values = [p[2] for p in pixels]
                
                avg_brightness = sum(r + g + b for r, g, b in pixels) / (len(pixels) * 3)
                color_variance = np.var([np.var(r_values), np.var(g_values), np.var(b_values)])
                
                # Simple classification based on image characteristics
                if avg_brightness > 180:
                    predicted_class = "bright_image"
                    confidence = 0.8
                elif avg_brightness < 80:
                    predicted_class = "dark_image"  
                    confidence = 0.75
                elif color_variance > 2000:
                    predicted_class = "colorful_image"
                    confidence = 0.7
                else:
                    predicted_class = "grayscale_image"
                    confidence = 0.6
                
                return {
                    "type": "simple_image_classification",
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "image_size": [width, height],
                    "features": {
                        "avg_brightness": float(avg_brightness),
                        "color_variance": float(color_variance),
                        "pixel_count": len(pixels)
                    },
                    "message": "Using simple PIL-based image analysis (YOLO not available)"
                }
                
        except Exception as e:
            return {
                "type": "error",
                "message": f"Failed to analyze image: {str(e)}"
            }
    
    def __call__(self, image_path):
        """Make it callable like YOLO model"""
        return [self.analyze_image(image_path)]

class DummyYOLOModel:
    """Dummy YOLO model for when YOLO is not available"""
    def __init__(self):
        self.names = {0: "dummy_object", 1: "placeholder"}
        self.simple_classifier = SimpleImageClassifier()
    
    def __call__(self, image_path):
        """Use simple image classifier instead of dummy results"""
        return self.simple_classifier(image_path)

def create_audio_classifier():
    """Create lightweight audio classifier using librosa + basic ML"""
    print(f"[worker] creating lightweight audio classifier...", flush=True)
    
    # This is a simple rule-based + feature-based classifier
    # In a real scenario, you'd load a pre-trained model
    class AudioClassifier:
        def __init__(self):
            self.classes = ["speech", "music", "noise", "silence"]
            
        def extract_features(self, audio_path):
            """Extract basic audio features using librosa"""
            try:
                # Load audio file
                y, sr = librosa.load(audio_path, sr=22050, duration=30.0)  # Max 30 seconds
                
                # Basic features
                features = {}
                
                # Duration and basic stats
                features['duration'] = len(y) / sr
                features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
                features['max_amplitude'] = float(np.max(np.abs(y)))
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                features['spectral_centroid_std'] = float(np.std(spectral_centroids))
                
                # Zero crossing rate (indicates speech vs music)
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                features['zcr_mean'] = float(np.mean(zcr))
                
                # MFCC features (basic speech characteristics)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                for i in range(5):  # Just use first 5 MFCCs
                    features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                    features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                
                # Tempo (for music detection)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = float(tempo)
                
                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                features['spectral_rolloff_mean'] = float(np.mean(rolloff))
                
                return features
                
            except Exception as e:
                print(f"[audio] feature extraction error: {e}", flush=True)
                return {}
        
        def classify(self, audio_path):
            """Simple rule-based classification"""
            features = self.extract_features(audio_path)
            if not features:
                return {"type": "error", "message": "Failed to extract features"}
            
            # Simple heuristic classification
            predicted_class = "unknown"
            confidence = 0.5
            
            # Silence detection
            if features['rms_energy'] < 0.01:
                predicted_class = "silence"
                confidence = 0.95
            
            # Music vs speech heuristics
            elif features['zcr_mean'] < 0.1 and features['tempo'] > 60:
                predicted_class = "music"
                confidence = 0.75
                
            elif features['zcr_mean'] > 0.1 and features['mfcc_1_mean'] > -50:
                predicted_class = "speech"
                confidence = 0.7
                
            else:
                predicted_class = "noise"
                confidence = 0.6
            
            return {
                "type": "audio_classification",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "features": features,
                "classes": self.classes
            }
    
    return AudioClassifier()

def connect_db():
    for i in range(30):
        try:
            conn = psycopg2.connect(DB_URL)
            conn.autocommit = True
            return conn
        except Exception as e:
            print(f"[worker] waiting for DB ({i})... {e}", flush=True)
            time.sleep(2)
    raise RuntimeError("could not connect to Postgres")

def connect_mq():
    params = pika.URLParameters(AMQP_URL)
    for i in range(30):
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=False)
            return connection, channel
        except Exception as e:
            print(f"[worker] waiting for MQ ({i})... {e}", flush=True)
            time.sleep(2)
    raise RuntimeError("could not connect to RabbitMQ")

def run_inference(models: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    model_key = job["model"]
    
    if model_key == "iris":
        if "iris" not in models:
            return {"type": "error", "message": "Iris model not available due to compatibility issues"}
        feats = job["input"]["features"]
        X = np.array(feats, dtype=float).reshape(1, -1)
        clf = models["iris"]
        y = clf.predict(X)[0]
        # map to names for readability
        # 0=setosa, 1=versicolor, 2=virginica (typical sklearn iris labels)
        names = ["setosa", "versicolor", "virginica"]
        name = names[int(y)] if int(y) in (0, 1, 2) else f"class_{int(y)}"
        prob = None
        try:
            prob = clf.predict_proba(X)[0].tolist()
        except Exception:
            pass
        return {"type": "classification", "label_id": int(y), "label_name": name, "proba": prob}

    elif model_key == "diabetes":
        if "diabetes" not in models:
            return {"type": "error", "message": "Diabetes model not available due to compatibility issues"}
        f = job["input"]["features"]
        cols = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
        X = np.array([[f[c] for c in cols]], dtype=float)
        reg = models["diabetes"]
        y = float(reg.predict(X)[0])
        return {"type": "regression", "prediction": y}

    elif model_key == "yolo":
        # Image classification with YOLO
        image_data = job["input"]["image_data"]  # Base64 encoded
        filename = job["input"]["filename"]
        
        # Decode base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
            tmp_file.write(image_bytes)
            image_path = tmp_file.name
        
        try:
            yolo_model = models["yolo"]
            
            # Check if we're using dummy model (with simple classifier)
            if isinstance(yolo_model, DummyYOLOModel):
                # Use the simple image classifier
                simple_results = yolo_model(image_path)
                if simple_results and len(simple_results) > 0:
                    return simple_results[0]  # Return the simple classification result
                else:
                    return {
                        "type": "object_detection", 
                        "detections": [],
                        "num_detections": 0,
                        "image_size": [640, 480],
                        "message": "Image analysis failed"
                    }
            
            results = yolo_model(image_path)
            
            # Extract detection results
            detections = []
            if len(results) > 0:
                result = results[0]  # First image
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            "class_id": int(box.cls.item()),
                            "class_name": yolo_model.names[int(box.cls.item())],
                            "confidence": float(box.conf.item()),
                            "bbox": box.xyxy.tolist()[0] if len(box.xyxy) > 0 else []
                        }
                        detections.append(detection)
            
            return {
                "type": "object_detection",
                "detections": detections,
                "num_detections": len(detections),
                "image_size": result.orig_shape if len(results) > 0 else None
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except Exception:
                pass

    elif model_key == "audio":
        # Audio classification with lightweight classifier
        audio_data = job["input"]["audio_data"]  # Base64 encoded
        filename = job["input"]["filename"]
        
        # Decode base64 data
        audio_bytes = base64.b64decode(audio_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
            tmp_file.write(audio_bytes)
            audio_path = tmp_file.name
        
        try:
            audio_classifier = models["audio"]
            result = audio_classifier.classify(audio_path)
            
            return result
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except Exception:
                pass

    else:
        raise ValueError(f"unknown model: {model_key}")

def main():
    print("[worker] loading models...", flush=True)
    
    # Load models with fallback handling
    models = {}
    
    # Iris model
    try:
        iris_model = load_pickle_from_hf(IRIS_REPO, IRIS_FILE)
        models["iris"] = iris_model
        print("[worker] iris model loaded successfully", flush=True)
    except Exception as e:
        print(f"[worker] Failed to load iris model: {e}", flush=True)
        print("[worker] iris predictions will be disabled", flush=True)
    
    # Diabetes model  
    try:
        diab_model = load_pickle_from_hf(DIAB_REPO, DIAB_FILE)
        models["diabetes"] = diab_model
        print("[worker] diabetes model loaded successfully", flush=True)
    except Exception as e:
        print(f"[worker] Failed to load diabetes model: {e}", flush=True)
        print("[worker] diabetes predictions will be disabled", flush=True)
    
    # YOLO model (always works with fallback)
    yolo_model = load_yolo_model(YOLO_REPO)
    models["yolo"] = yolo_model
    print("[worker] YOLO/image model ready", flush=True)
    
    # Audio classifier (always works)
    audio_classifier = create_audio_classifier()
    models["audio"] = audio_classifier
    print("[worker] audio classifier ready", flush=True)
    
    print(f"[worker] loaded {len(models)} models successfully", flush=True)

    conn = connect_db()
    cur = conn.cursor()

    connection, channel = connect_mq()
    print("[worker] consuming queue:", QUEUE_NAME, flush=True)

    def on_message(ch, method, properties, body):
        try:
            job = json.loads(body.decode("utf-8"))
            job_id = job["job_id"]
            result = run_inference(models, job)
            # update DB
            cur.execute(
                "UPDATE jobs SET status=%s, result=%s, error=NULL, updated_at=now() WHERE id=%s",
                ("completed", json.dumps(result), job_id),
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[worker] job {job_id} done", flush=True)
        except Exception as e:
            print(f"[worker] error: {e}", flush=True)
            try:
                job_id = json.loads(body.decode("utf-8")).get("job_id")
                cur.execute(
                    "UPDATE jobs SET status=%s, error=%s, updated_at=now() WHERE id=%s",
                    ("failed", str(e), job_id),
                )
            except Exception as inner:
                print(f"[worker] failed to record error: {inner}", flush=True)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message, auto_ack=False)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        try:
            channel.stop_consuming()
        except Exception:
            pass
    connection.close()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
