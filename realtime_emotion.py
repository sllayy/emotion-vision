import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from predict_emotion import plot_emotions  # Grafik fonksiyonunu ayrı dosyada tanımlamıştık
import time

# Emojilerle duygu eşleştirmesi
emoji_map = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😱",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😮",
    "Neutral": "😐"
}
emotion_labels = list(emoji_map.keys())

# Eğitimli modeli ve yüz algılama sınıflandırıcısını yükle
model = load_model("models/emotion_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Kamera başlat
cap = cv2.VideoCapture(0)
last_plot_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
    # ROI: sadece yüz alanını al

        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi_resized = cv2.resize(roi_gray, (48, 48))
        except:
            continue

        face_input = roi_resized.reshape(1, 48, 48, 1).astype("float32") / 255.0
        predictions = model.predict(face_input, verbose=0)[0]

        predicted_index = np.argmax(predictions)
        predicted_label = emotion_labels[predicted_index]
        predicted_emoji = emoji_map[predicted_label]

        # Yüz etrafına kare çiz ve duygu+emoji yaz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_label} {predicted_emoji}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Her 2 saniyede bir duygu dağılım grafiğini çiz
        if time.time() - last_plot_time > 2:
            plot_emotions(predictions, emotion_labels)
            last_plot_time = time.time()

    # Kamera görüntüsünü göster
    cv2.imshow("Real-Time Emotion Detection", frame)

    # 'q' tuşuyla çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
