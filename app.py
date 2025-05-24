from flask import Flask, request, render_template
import numpy as np
np.complex = complex
import os, tempfile
import librosa, librosa.display
import matplotlib.pyplot as plt
import scipy.signal
import tensorflow as tf

app = Flask(__name__)

# โหลดโมเดล
model = tf.keras.models.load_model('model/BuddyBirdModel.keras')
labels = ['crow', 'koel', 'myna', 'sparrow']

def analyze_bird_audio(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    threshold = np.mean(magnitude[(frequency < 1000) | (frequency > 8000)]) * 3
    noise_mask = (magnitude > threshold) & ((frequency < 1000) | (frequency > 8000))
    noise_freq = frequency[noise_mask]
    low_noise = noise_freq[noise_freq < 500]
    high_noise = noise_freq[noise_freq > 10000]
    return {
        'low_noise': (np.min(low_noise) if len(low_noise) > 0 else None,
                      np.max(low_noise) if len(low_noise) > 0 else None),
        'high_noise': (np.min(high_noise) if len(high_noise) > 0 else None,
                       np.max(high_noise) if len(high_noise) > 0 else None)
    }

def adaptive_filter(y, sr, analysis):
    nyquist = sr / 2

    if analysis['low_noise'][0] is not None:
        cutoff = min(300, analysis['low_noise'][1])
        if 0 < cutoff < nyquist:
            b, a = scipy.signal.butter(4, cutoff / nyquist, btype='highpass')
            y = scipy.signal.filtfilt(b, a, y)

    if analysis['high_noise'][0] is not None:
        cutoff = max(8000, analysis['high_noise'][0])
        if 0 < cutoff < nyquist:
            b, a = scipy.signal.butter(4, cutoff / nyquist, btype='lowpass')
            y = scipy.signal.filtfilt(b, a, y)

    return scipy.signal.wiener(y)
#ลือกdirectory สำหรับบันทึกภาพ Spectrogram ที่output_dir
def generate_bird_spectrogram(file, output_dir="spectrograms"):
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir) if f.startswith("BB") and f.endswith(".png")]
    numbers = [int(f[2:8]) for f in existing if f[2:8].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    id = f"BB{next_number:06d}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        path = tmp.name

    y, sr = librosa.load(path, sr=22050)
    y_filtered = adaptive_filter(y, sr, analyze_bird_audio(y, sr))
    y_filtered = librosa.util.normalize(y_filtered)

    S = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_fft=2048, hop_length=128, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # บันทึกภาพ
    filename =f"{id}_spectrogram.png" 
    img_path = os.path.join(output_dir, filename)
    S_img = np.clip(S_db, -60, 0)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_img, sr=sr, hop_length=128, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # เตรียม input AI
    avg_energy = S_db.mean(axis=0)
    center = np.argmax(avg_energy)
    start = max(0, center - 64)
    S_db = S_db[:, start:start+128]
    if S_db.shape[1] < 128:
        S_db = np.pad(S_db, ((0, 0), (0, 128 - S_db.shape[1])), mode='constant')
    S_db = S_db.reshape(1, 128, 128, 1)

    os.remove(path)
    return S_db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    try:
        X = generate_bird_spectrogram(file)
        pred = model.predict(X)
        label = labels[np.argmax(pred)]
        conf = np.max(pred) * 100
        return f"""
        <h2>ผลลัพธ์:</h2>
        <b>นก:</b> {label}<br>
        <b>ความมั่นใจ:</b> {conf:.2f}%<br>
        """
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {e}"

if __name__ == '__main__':
    app.run(debug=True)
