""" import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import librosa
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # Corrected the import statement

# Data Augmentation functions
def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

def pitch_shift(data, sr):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=np.random.uniform(-2, 2))

def time_stretch(data):
    return librosa.effects.time_stretch(data, rate=np.random.uniform(0.8, 1.2))

# Load the dataset with augmentation
def load_fsdd_data(data_path):
    X = []
    y = []
    print("Loading data...")
    for filename in os.listdir(data_path):
        if filename.endswith('.wav'):
            label = int(filename[0])  # Assuming the file name starts with the digit label
            file_path = os.path.join(data_path, filename)
            audio, sr = librosa.load(file_path, sr=None)
            print(f"Processing {filename}...")

            # Apply data augmentation and append label for each augmented sample
            augmented_data = [audio, add_noise(audio), pitch_shift(audio, sr), time_stretch(audio)]

            for data in augmented_data:
                mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
                mfccs = np.mean(mfccs.T, axis=0)
                zero = librosa.feature.zero_crossing_rate(y=data)
                zero = np.mean(zero.T, axis=0)
                chroma = librosa.feature.chroma_stft(y=data, sr=sr)
                chroma = np.mean(chroma.T, axis=0)
                spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
                spectral_centroid = np.mean(spectral_centroid.T, axis=0)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr)
                spectral_bandwidth = np.mean(spectral_bandwidth.T, axis=0)
                spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=sr)
                spectral_contrast = np.mean(spectral_contrast.T, axis=0)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)
                spectral_rolloff = np.mean(spectral_rolloff.T, axis=0)
                intensity = librosa.feature.rms(y=data)
                intensity = np.mean(intensity.T, axis=0)
                voiced_frames = librosa.effects.split(data, top_db=20)
                speech_rate = len(voiced_frames) / (len(data) / sr)
                speech_rate = np.array([speech_rate])
                total = np.concatenate((mfccs, zero, chroma, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, intensity, speech_rate))
                X.append(total)
                y.append(label)
    
    print("Data loading complete.")
    return np.array(X), np.array(y)

def preprocess_data(X, y):
    from sklearn.preprocessing import LabelBinarizer, StandardScaler
    from sklearn.model_selection import train_test_split

    print("Preprocessing data...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler

# Define the RNN model with additional complexity
def create_rnn_model(input_shape):
    print("Creating RNN model...")
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())  # Added batch normalization
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())  # Added batch normalization
    model.add(LSTM(64, return_sequences=False))  # Added another LSTM layer
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))  # Added a dense layer with ReLU
    model.add(Dense(1, activation='sigmoid'))

    # Changed optimizer to Nadam
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print("RNN model created.")
    return model

# Main script
if __name__ == "__main__":
    data_path = './Cleaned_Data_Set'  # Replace with your dataset path
    X, y = load_fsdd_data(data_path)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    input_shape = (X_train.shape[1], 1)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = create_rnn_model(input_shape)
    model.summary()

    print("Training model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Plot model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Generate predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Confusion Matrix
    labels = ['Class 0', 'Class 1']  # Adjust these labels based on your classes
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=labels))

    # Save the model and scaler
    model_save_path = './recordings/finalmodel.pkl'  # Corrected the file name typo
    scaler_save_path = './recordings/finalscaler.pkl'
    
    print(f"Saving model to {model_save_path}")
    joblib.dump(model, model_save_path)
    
    print(f"Saving scaler to {scaler_save_path}")
    joblib.dump(scaler, scaler_save_path)

    print("Model and scaler saved successfully.")
 """
import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Data Augmentation --------------------
def add_noise(data):
    noise = np.random.randn(len(data))
    return data + 0.005 * noise

def pitch_shift(data, sr):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=np.random.uniform(-2, 2))

def time_stretch(data):
    return librosa.effects.time_stretch(data, rate=np.random.uniform(0.8, 1.2))

# -------------------- Feature Extraction --------------------
def extract_features(data, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13).T, axis=0)
    zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr).T, axis=0)
    intensity = np.mean(librosa.feature.rms(y=data).T, axis=0)
    voiced_frames = librosa.effects.split(data, top_db=20)
    speech_rate = np.array([len(voiced_frames) / (len(data) / sr)])

    features = np.concatenate((mfccs, zero, chroma, spectral_centroid,
                               spectral_bandwidth, spectral_contrast,
                               spectral_rolloff, intensity, speech_rate))
    return features

# -------------------- Dataset Loading --------------------
def load_data(data_path):
    X, y = [], []
    print("Loading and processing dataset...")

    for filename in os.listdir(data_path):
        if filename.endswith('.wav'):
            label = int(filename[0])  # first char as label
            file_path = os.path.join(data_path, filename)
            audio, sr = librosa.load(file_path, sr=None)

            augmented_data = [audio, add_noise(audio), pitch_shift(audio, sr), time_stretch(audio)]

            for data in augmented_data:
                features = extract_features(data, sr)
                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

# -------------------- Data Preprocessing --------------------
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    return X_train, X_test, y_train, y_test, scaler

# -------------------- Model Definition --------------------
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Training Script --------------------
if __name__ == "__main__":
    data_path = './Cleaned_Data_Set'  # update path

    # Load & preprocess data
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Build & train model
    input_shape = (X_train.shape[1], 1)
    model = create_rnn_model(input_shape)
    model.summary()

    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_test, y_test))

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Evaluate & visualize results
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    labels = ['Class 0', 'Class 1']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=labels))

    # -------------------- Save Model & Scaler --------------------
    # Strip optimizer to prevent pickle shape errors
    model.optimizer = None

    model_save_path = './recordings/finalmodel.pkl2'
    scaler_save_path = './recordings/finalscaler.pkl2'

    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"Model saved to {model_save_path} without optimizer.")
    print(f"Scaler saved to {scaler_save_path}.")
