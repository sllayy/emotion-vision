import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Modeli yükle
model = tf.keras.models.load_model("models/emotion_model.h5")

# Duygu etiketleri (sıralama modelin eğitildiği sıraya göre)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Görüntüyü yükle ve işle
img_path = "test_image.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (48, 48))
input_img = resized.reshape(1, 48, 48, 1).astype("float32") / 255.0

# Tahmin yap
predictions = model.predict(input_img)
predicted_class = np.argmax(predictions)
predicted_emotion = emotion_labels[predicted_class]

# Sonucu göster
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Tahmin: {predicted_emotion}")
plt.axis('off')
plt.show()
