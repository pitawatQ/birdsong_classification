import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
layers = keras.layers
models = keras.models
import os

# ===== Step 1: Load CSV =====
df = pd.read_csv('birdsong_data.csv')

X = []
y = []

print("กำลังโหลดไฟล์เสียงและแปลงเป็น Spectrogram...")

# ===== Step 2: Preprocess Audio =====
for index, row in df.iterrows():
    filepath = row['file']
    label = row['species']
    try:
        y_audio, sr = librosa.load(filepath, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_db = mel_spec_db[:, :128]
        if mel_spec_db.shape[1] < 128:
            pad_width = 128 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0,0),(0,pad_width)), mode='constant')

        X.append(mel_spec_db)
        y.append(label)
    except Exception as e:
        print(f"❌ Error at index {index}: {filepath} -> {e}")


X = np.array(X)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  #เพิ่ม channel dimension

# ===== Step 3: Encode Labels =====
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded = tf.keras.utils.to_categorical(y_encoded)
print(le.classes_)
# ===== Step 4: Train-Test Split =====
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ===== Step 5: Build Model =====
model = models.Sequential([
    layers.Conv2D(32, (3,3),
    activation='relu', 
    input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
     layers.Dropout(0.5),
    layers.Dense(4, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===== Step 6: Setup Checkpoint =====
os.makedirs('model', exist_ok=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('model/BuddyBirdModel.keras', save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(patience=5)

# ===== Step 7: Train =====
print("เริ่มเทรนโมเดล...")
model.fit(X_train, y_train, epochs=50, batch_size=8,
          validation_data=(X_val, y_val),
          callbacks=[checkpoint, early_stop])

print("เทรนเสร็จแล้ว! โมเดลถูกบันทึกไว้ที่ model/BuddyBirdModel.keras")
