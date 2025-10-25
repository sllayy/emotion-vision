import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 🎯 Modeli yükle
model = load_model("models/emotion_model.h5")

# 🎯 Duygu sınıfları
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 🎥 Kamera başlat
cap = cv2.VideoCapture(0)

print("📸 Kameradan yüz algılanıyor. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 👁‍🗨 Griye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🧠 Yüz algılayıcı (OpenCV default cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        # 🎯 Tahmin
        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]

        # 🖼️ Ekrana yaz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Göster
    cv2.imshow("Duygu Tahmini", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizle
cap.release()
cv2.destroyAllWindows()
