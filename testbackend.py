# backend.py
import os

# ===== Hard caps BEFORE heavy imports =====
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Load .env in local dev (Render injects env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import logging
import tempfile
import shutil
import subprocess
from datetime import datetime, timezone
from threading import Lock

import numpy as np
import joblib
import librosa
import bcrypt
import speech_recognition as sr

from flask import Flask, request, jsonify, session, make_response
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, login_required
from flask_session import Session
from bson.objectid import ObjectId
from bson.binary import Binary

# ---- TensorFlow AFTER env caps ----
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

from tensorflow.keras.optimizers import Nadam

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # optional

# ===================== Deployment constants =====================
PROD_DOMAIN = os.getenv("PROD_DOMAIN", "persuasive.research.cs.dal.ca")
API_PREFIX  = os.getenv("API_PREFIX", "/smsys")
ALLOWED_ORIGINS = {
    # Local dev
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5173", "http://127.0.0.1:5173",
    # GitHub Pages (your SPA)
    "https://stacygirly.github.io",
    # Prod apex (if you front with reverse proxy under /smsys)
    "https://persuasive.research.cs.dal.ca",
    # Render backend host
    "https://smsys.onrender.com",
}

# ===================== Flask app & session =====================
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-only-fallback")
app.config["MONGO_URI"]  = os.getenv("MONGO_URI")

# Limit request body (~20 MB)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# File-system sessions (simple & reliable on Render free tier)
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_NAME="session",
    SESSION_COOKIE_HTTPONLY=True,
    # defaults for local; updated dynamically in @before_request
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_PATH="/",
    SESSION_COOKIE_DOMAIN=None,
)
Session(app)

# CORS: allow credentials, validate known origins
CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": list(ALLOWED_ORIGINS)}},
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
)

@app.after_request
def add_cors_headers(resp):
    """Echo the exact Origin so browsers will accept cookies."""
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return resp

@app.before_request
def scope_session_cookie():
    """
    Ensure the session cookie has the correct samesite/secure/path/domain
    for each environment (local, Render, prod under /smsys).
    """
    host = (request.host or "").split(":")[0]
    is_prod   = host.endswith(PROD_DOMAIN)
    is_render = host.endswith(".onrender.com")  # e.g. smsys.onrender.com

    if is_prod:
        app.config.update(
            SESSION_COOKIE_SAMESITE="None",
            SESSION_COOKIE_SECURE=True,
            SESSION_COOKIE_DOMAIN=PROD_DOMAIN,
            SESSION_COOKIE_PATH=API_PREFIX,
        )
    elif is_render:
        app.config.update(
            SESSION_COOKIE_SAMESITE="None",
            SESSION_COOKIE_SECURE=True,
            SESSION_COOKIE_DOMAIN=None,  # host-only
            SESSION_COOKIE_PATH="/",
        )
    else:
        app.config.update(
            SESSION_COOKIE_SAMESITE="Lax",
            SESSION_COOKIE_SECURE=False,
            SESSION_COOKIE_DOMAIN=None,
            SESSION_COOKIE_PATH="/",
        )

# Explicit preflight for proxies that are picky
@app.route("/<path:_any>", methods=["OPTIONS"])
def _options(_any):
    return make_response(("", 204))

# ===================== Logging =====================
class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{record.levelname}: {record.getMessage()}"

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("pymongo").setLevel(logging.ERROR)

# ===================== Mongo =====================
mongo = PyMongo(app)
mongo.db.users.create_index("username", unique=True)
try:
    mongo.cx.server_info()
    print("✅ Connected to MongoDB successfully")
except Exception as e:
    print("❌ Failed to connect to MongoDB", e)

# ===================== Login Manager =====================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"message": "Not logged in"}), 401

class User(UserMixin):
    def __init__(self, user_id, username, password):
        self.id = user_id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    return User(str(user["_id"]), user["username"], user["password"]) if user else None

# ===================== Audio / ML =====================
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

def _ffmpeg_to_wav(src_path, dst_path, max_seconds=75):
    """Convert to mono 16k WAV and hard-cap length to avoid timeouts."""
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path, "-t", str(max_seconds),
        "-ac", "1", "-ar", "16000", dst_path
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore") or "ffmpeg failed")

def extract_features(y, sr):
    try:
        zcr  = np.mean(librosa.feature.zero_crossing_rate(y=y))
        sc   = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sb   = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        sro  = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        intensity = np.mean(np.abs(y))
        speech_rate = np.array([len(librosa.effects.split(y, top_db=20)) / (len(y) / sr)])
        scontrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        feats = np.hstack([zcr, sc, sb, sro, mfcc, chroma, intensity, scontrast, speech_rate])
        if feats.size != 38:
            raise ValueError(f"Expected 38 features, got {feats.size}")
        return feats
    except Exception as e:
        app.logger.error(f"Feature extraction error: {e}")
        return None

LOW_BAND, HIGH_BAND, THRESHOLD = 0.12, 0.35, 0.23
def pace_suggestion_from_pred(p):
    if p >= HIGH_BAND: return "pace_down"
    if p <= LOW_BAND:  return "pace_up"
    return "steady"

# Lazy load model/scaler to avoid boot OOM/timeouts
_model, _scaler = None, None
_model_lock = Lock()

def _load_model_and_scaler():
    global _model, _scaler
    if _model is not None and _scaler is not None:
        return _model, _scaler
    with _model_lock:
        if _model is None or _scaler is None:
            mdl = joblib.load("./recordings/finalmodel.pkl")
            mdl.optimizer = None
            mdl.compile(optimizer=Nadam(), loss="binary_crossentropy", metrics=["accuracy"])
            scl = joblib.load("./recordings/finalscaler.pkl")
            _model, _scaler = mdl, scl
            print("✅ Model & scaler loaded for inference.")
    return _model, _scaler

recognizer = sr.Recognizer()

# ===================== Routes =====================
@app.route("/")
def home():
    return jsonify({"message": "Backend is running successfully!"})

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.route("/wake")
def wake():
    try:
        _load_model_and_scaler()
        return jsonify({"ok": True, "model": "ready"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ---- Auth ----
@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    confirm  = data.get("confirmPassword")
    if not username or not password or not confirm:
        return jsonify({"message": "Missing fields"}), 400
    if password != confirm:
        return jsonify({"message": "Passwords do not match"}), 400
    if mongo.db.users.find_one({"username": username}):
        return jsonify({"message": "Username already exists"}), 400
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    uid = mongo.db.users.insert_one({"username": username, "password": hashed}).inserted_id
    return jsonify({"message": "User registered", "user_id": str(uid)}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username", "")
    raw_pw   = (data.get("password") or "").encode("utf-8")

    user = mongo.db.users.find_one({"username": username})
    if not user:
        return jsonify({"message": "Invalid credentials"}), 401

    stored = user.get("password")
    if isinstance(stored, Binary):      stored_bytes = bytes(stored)
    elif isinstance(stored, (bytes, bytearray)): stored_bytes = stored
    elif isinstance(stored, str):       stored_bytes = stored.encode("utf-8")
    else:                               return jsonify({"message": "Invalid credentials"}), 401

    if bcrypt.checkpw(raw_pw, stored_bytes):
        user_obj = User(str(user["_id"]), user["username"], stored_bytes)
        login_user(user_obj)
        session["user_id"] = str(user["_id"])
        session["username"] = user["username"]
        return jsonify({"message": "Logged in", "id": str(user["_id"]), "username": user["username"]}), 200

    return jsonify({"message": "Invalid credentials"}), 401

@app.route("/@me")
def get_current_user_route():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"message": "Not logged in"}), 401
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    return jsonify({"id": str(user["_id"]), "username": user["username"], "message": "Session active"})

@app.route("/me")
def get_current_user_alias():
    return get_current_user_route()

@app.route("/logout", methods=["POST"])
def logout_user():
    session.clear()
    resp = jsonify({"message": "Logged out"})
    host = (request.host or "").split(":")[0]
    if host.endswith(PROD_DOMAIN):
        resp.delete_cookie("session", path=API_PREFIX, domain=PROD_DOMAIN, samesite="None", secure=True)
    elif host.endswith(".onrender.com"):
        resp.delete_cookie("session", path="/", samesite="None", secure=True)
    else:
        resp.delete_cookie("session", path="/")
    return resp, 200

@app.route("/update_username", methods=["POST"])
@login_required
def update_username():
    data = request.json or {}
    new_username = data.get("new_username")
    if not new_username:
        return jsonify({"message": "New username is required"}), 400
    user_id = session.get("user_id")
    mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"username": new_username}})
    session["username"] = new_username
    return jsonify({"message": "Username updated successfully", "new_username": new_username}), 200

# ---- Settings/logging helpers ----
@app.route("/settings/logging", methods=["POST", "GET"])
@login_required
def settings_logging():
    user_id = session.get("user_id")
    if request.method == "GET":
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "settings": 1})
        interval = (user or {}).get("settings", {}).get("logging_interval_sec")
        return jsonify({"logging_interval_sec": interval}), 200

    data = request.json or {}
    interval = data.get("intervalSeconds")
    if not isinstance(interval, (int, float)) or interval <= 0:
        return jsonify({"message": "intervalSeconds must be a positive number"}), 400

    mongo.db.users.update_one({"_id": ObjectId(user_id)},
                              {"$set": {"settings.logging_interval_sec": int(interval)}})
    return jsonify({"message": "Interval updated", "logging_interval_sec": int(interval)}), 200

@app.route("/log_status", methods=["POST"])
@login_required
def log_status():
    user_id = session.get("user_id")
    data = request.json or {}
    status = (data.get("status") or "not stressed").strip().lower()
    if status not in ("stressed", "not stressed"):
        return jsonify({"message": "status must be 'stressed' or 'not stressed'"}), 400

    model_pred = data.get("model_prediction")
    try:
        model_pred = float(model_pred) if model_pred is not None else (
            THRESHOLD - 0.05 if status == "not stressed" else THRESHOLD + 0.12
        )
    except Exception:
        model_pred = THRESHOLD - 0.05 if status == "not stressed" else THRESHOLD + 0.12

    pace = pace_suggestion_from_pred(model_pred)
    if status == "not stressed" and pace == "pace_up" and model_pred > LOW_BAND:
        pace = "steady"

    confidence = float(data.get("confidence") or 0.5)
    note = data.get("note") or "Interval snapshot"

    entry = {
        "emotion": status,
        "reason": note,
        "features": {},
        "model_prediction": float(model_pred),
        "confidence": float(confidence),
        "pace": pace,
        "source": "interval",
        "timestamp": datetime.now(timezone.utc)
    }
    mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$push": {"emotions": entry}})
    return jsonify(entry), 200

# ---- Prediction (webm/ogg/wav → wav → features → LSTM) ----
@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    try:
        model, scaler = _load_model_and_scaler()
    except Exception as e:
        app.logger.exception("Model load failed")
        return jsonify({"error": "model_load_failed", "detail": str(e)}), 500

    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    uploaded = request.files["file"]
    if not uploaded or uploaded.filename == "":
        return jsonify({"error": "empty filename"}), 400

    name_lower = (uploaded.filename or "").lower()
    mt = (uploaded.mimetype or "").lower()
    suffix = ".webm"
    if name_lower.endswith(".ogg") or "ogg" in mt: suffix = ".ogg"
    if name_lower.endswith(".wav") or "wav" in mt or "wave" in mt: suffix = ".wav"

    up_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix); up_tmp.close()
    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav"); wav_tmp.close()

    try:
        uploaded.save(up_tmp.name)
        if os.path.getsize(up_tmp.name) < 1500:
            raise RuntimeError("Uploaded file too small (likely truncated).")

        try:
            _ffmpeg_to_wav(up_tmp.name, wav_tmp.name, max_seconds=75)
        except Exception as fferr:
            if suffix == ".wav":
                shutil.copyfile(up_tmp.name, wav_tmp.name)
            else:
                raise RuntimeError(str(fferr))

        y, sr = librosa.load(wav_tmp.name, sr=None, mono=True)
        if y is None or y.size == 0:
            raise RuntimeError("Decoded audio is empty.")

        # Guard rails in memory too
        max_duration = 75.0
        duration = librosa.get_duration(y=y, sr=sr)
        if duration > max_duration + 1:
            y = y[: int(sr * max_duration)]
            duration = max_duration

        # Lightweight chunking
        num_frames = 4 if duration <= 60 else max(1, int(duration // 15))
        chunk_size = max(1, len(y) // num_frames)

        frame_preds = []
        last_features = None
        for i in range(num_frames):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_frames - 1 else len(y)
            feats = extract_features(y[start:end], sr)
            if feats is None or feats.size != 38:
                return jsonify({"error": "Feature extraction failed"}), 500
            last_features = feats
            norm = scaler.transform([feats])
            x = np.expand_dims(norm, axis=2)  # (1, 38, 1)
            yhat = model.predict(x, verbose=0)
            frame_preds.append(float(yhat[0][0]))

        mean_pred = float(np.mean(frame_preds))
        label = "stressed" if mean_pred >= THRESHOLD else "not stressed"
        confidence = float(max(0.0, min(1.0, abs(mean_pred - THRESHOLD) / max(THRESHOLD, 1 - THRESHOLD))))
        pace = pace_suggestion_from_pred(mean_pred)

        # map features (last window only)
        feature_names = (
            ["zero_crossings", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"] +
            [f"mfcc_{i+1}" for i in range(13)] +
            [f"chroma_{i+1}" for i in range(12)] +
            ["intensity"] +
            [f"spectral_contrast_{i+1}" for i in range(7)] +
            ["speech_rate"]
        )
        features_map = {}
        if last_features is not None and last_features.size == 38:
            features_map = {k: float(v) for k, v in zip(feature_names, last_features)}

        # Optional logging to user history if logged in
        user_id = session.get("user_id")
        ts = datetime.now(timezone.utc)
        reason = "Prediction threshold check"
        if user_id:
            try:
                mongo.db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$push": {"emotions": {
                        "emotion": label,
                        "reason": reason,
                        "features": features_map,
                        "model_prediction": mean_pred,
                        "confidence": confidence,
                        "pace": pace,
                        "source": "prediction",
                        "timestamp": ts
                    }}}
                )
            except Exception as db_err:
                app.logger.error(f"DB log error: {db_err}")

        return jsonify({
            "emotion": label,
            "model_prediction": mean_pred,
            "confidence": confidence,
            "pace_suggestion": pace,
            "reason": reason,
            "timestamp": ts.isoformat(),
            "features": features_map
        }), 200

    except RuntimeError as e:
        return jsonify({"error": "decode_failed", "detail": str(e)}), 415
    except Exception as e:
        app.logger.exception("predict_emotion crashed")
        return jsonify({"error": "server_error", "detail": str(e)}), 500
    finally:
        for p in (up_tmp.name, wav_tmp.name):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# ---- Chat (optional; requires OPENAI_API_KEY) ----
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if OpenAI and os.getenv("OPENAI_API_KEY") else None

@app.route("/chat", methods=["POST"])
@login_required
def chat_with_gpt():
    if not _openai_client:
        return jsonify({"error": "Chat service unavailable"}), 502
    data = request.json or {}
    user_message = (data.get("message") or "").strip()
    user_id = session.get("user_id")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant focused on stress/wellbeing."},
                {"role": "user", "content": user_message},
            ],
        )
        content = resp.choices[0].message.content
        mongo.db.chats.insert_one({
            "user_id": ObjectId(user_id),
            "messages": [{"role": "user", "content": user_message},
                         {"role": "assistant", "content": content}],
            "timestamp": datetime.now(timezone.utc),
        })
        return jsonify({"message": content}), 200
    except Exception as e:
        app.logger.error(f"OpenAI error: {e}")
        return jsonify({"error": "Chat service unavailable"}), 502

@app.route("/chats", methods=["GET"])
@login_required
def get_chats():
    user_id = session.get("user_id")
    chats = mongo.db.chats.find({"user_id": ObjectId(user_id)})
    chat_list = [{"messages": c["messages"], "timestamp": c["timestamp"]} for c in chats]
    return jsonify(chat_list), 200

# ---- Emotions Data ----
@app.route("/user_emotions", methods=["GET"])
def get_user_emotions():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify([]), 200
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "emotions": 1})
    emotions = user.get("emotions", []) if user else []
    emotions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return jsonify(emotions), 200

@app.route("/stress_dashboard", methods=["GET"])
def stress_dashboard():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify([]), 200
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "emotions": 1})
    data = user.get("emotions", []) if user else []
    data.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return jsonify(data), 200

# ---- Interventions ----
ALLOWED_INTERVENTIONS = {"breathing", "break", "dietVoice", "chatbot"}

def _parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else ts
    except Exception:
        return None

def _jsonify_intervention(doc):
    out = dict(doc)
    if "iid" in out and isinstance(out["iid"], ObjectId):
        out["iid"] = str(out["iid"])
    for key in ("start_time", "end_time", "stressed_detected_at"):
        if isinstance(out.get(key), datetime):
            out[key] = out[key].isoformat()
    if isinstance(out.get("actions"), list):
        norm = []
        for a in out["actions"]:
            aa = dict(a)
            if isinstance(aa.get("at"), datetime):
                aa["at"] = aa["at"].isoformat()
            norm.append(aa)
        out["actions"] = norm
    return out

@app.route("/interventions", methods=["GET"])
@login_required
def list_interventions():
    user_id = session.get("user_id")
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "interventions": 1})
    items = (user or {}).get("interventions", [])
    items.sort(key=lambda x: x.get("start_time", datetime.min), reverse=True)
    return jsonify([_jsonify_intervention(it) for it in items]), 200

@app.route("/interventions/start", methods=["POST"])
@login_required
def intervention_start():
    user_id = session.get("user_id")
    data = request.json or {}

    itype = (data.get("type") or "").strip()
    if itype not in ALLOWED_INTERVENTIONS:
        return jsonify({"message": f"type must be one of {sorted(ALLOWED_INTERVENTIONS)}"}), 400

    emotion_at_trigger = (data.get("emotion_at_trigger") or "").lower()
    if emotion_at_trigger not in ("stressed", "not stressed"):
        return jsonify({"message": "emotion_at_trigger must be 'stressed' or 'not stressed'"}), 400

    model_prediction = float(data.get("model_prediction")) if data.get("model_prediction") is not None else None
    confidence       = float(data.get("confidence")) if data.get("confidence") is not None else None
    pace             = data.get("pace") or None
    stressed_at      = _parse_iso(data.get("stressed_detected_at"))
    note             = data.get("note") or ""

    start = datetime.now(timezone.utc)
    iid = ObjectId()
    doc = {
        "iid": iid, "type": itype, "status": "started",
        "start_time": start, "end_time": None, "duration_ms": None,
        "emotion_at_trigger": emotion_at_trigger,
        "model_prediction": model_prediction, "confidence": confidence, "pace": pace,
        "stressed_detected_at": stressed_at, "actions": [], "note": note, "source": "ui",
    }
    mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$push": {"interventions": doc}})
    return jsonify({"iid": str(iid), "start_time": start.isoformat()}), 201

@app.route("/interventions/action", methods=["POST"])
@login_required
def intervention_action():
    user_id = session.get("user_id")
    data = request.json or {}
    iid = data.get("iid")
    action = (data.get("action") or "").strip()
    if not iid or not action:
        return jsonify({"message": "iid and action are required"}), 400

    try:
        aid = ObjectId(iid)
    except Exception:
        return jsonify({"message": "Invalid iid"}), 400

    evt = {"at": datetime.now(timezone.utc), "action": action, "meta": data.get("meta") or {}}
    result = mongo.db.users.update_one(
        {"_id": ObjectId(user_id), "interventions.iid": aid},
        {"$push": {"interventions.$.actions": evt}}
    )
    if result.matched_count == 0:
        return jsonify({"message": "Intervention not found"}), 404
    return jsonify({"ok": True}), 200

@app.route("/interventions/finish", methods=["POST"])
@login_required
def intervention_finish():
    user_id = session.get("user_id")
    data = request.json or {}
    iid = data.get("iid")
    outcome = (data.get("outcome") or "").strip().lower()
    if outcome not in ("completed", "dismissed", "skipped"):
        return jsonify({"message": "outcome must be completed|dismissed|skipped"}), 400
    if not iid:
        return jsonify({"message": "iid is required"}), 400

    try:
        aid = ObjectId(iid)
    except Exception:
        return jsonify({"message": "Invalid iid"}), 400

    now = datetime.now(timezone.utc)
    doc = mongo.db.users.find_one({"_id": ObjectId(user_id), "interventions.iid": aid}, {"interventions.$": 1})
    if not doc or not doc.get("interventions"):
        return jsonify({"message": "Intervention not found"}), 404

    started = doc["interventions"][0].get("start_time")
    server_dur = int((now - started).total_seconds() * 1000) if isinstance(started, datetime) else None

    duration_ms = data.get("duration_ms")
    if duration_ms is None:
        duration_ms = server_dur

    mongo.db.users.update_one(
        {"_id": ObjectId(user_id), "interventions.iid": aid},
        {"$set": {
            "interventions.$.end_time": now,
            "interventions.$.duration_ms": int(duration_ms) if duration_ms is not None else None,
            "interventions.$.status": outcome
        }}
    )
    return jsonify({"ok": True, "end_time": now.isoformat(), "duration_ms": duration_ms}), 200

# ---- Debug helper ----
@app.route("/debug/cookies")
def debug_cookies():
    return jsonify({
        "cookies_seen_by_server": {k: v for k, v in request.cookies.items()},
        "session_has_user_id": bool(session.get("user_id")),
        "host": request.host,
        "path": request.path,
    })

# ===================== Run (dev only) =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
