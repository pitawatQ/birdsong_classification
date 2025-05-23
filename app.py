from flask import Flask, request, render_template
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)

#โหลดโมเดลที่เทรนไว้
model = tf.keras.models.load_model('model/best_model.h5')
labels = ['crow','koel','myna','sparrow']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    
    file = request.files['file']
    y, sr = librosa.load(file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = mel_db[:, :128]
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0,0),(0,128 - mel_db.shape[1])), mode='constant')

    X_input = mel_db.reshape(1, 128, 128, 1)
    pred = model.predict(X_input)
    pred_class = labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    return f"นกที่คุณอัปโหลดคือ: **{pred_class}**<br>ความมั่นใจ: {confidence:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)
