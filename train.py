import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# CSV dosyasını oku
csv_path = "dataset/fer2013_training_onehot.csv"
df = pd.read_csv(csv_path)

# Piksel ve etiket verilerini ayır
X = df.iloc[:, :2304].values.astype('float32') / 255.0  # Görseller (48x48 gri tonlamalı)
y = df.iloc[:, 2304:].values.astype('float32')          # One-hot encoded etiketler

# Görsel boyutlarını CNN girişine uygun hale getir
X = X.reshape(-1, 48, 48, 1)

# Eğitim ve doğrulama verisine böl
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN mimarisi
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 duygu sınıfı (one-hot)
])

# Derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Modeli kaydet
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "emotion_model.h5"))
print("✅ Model başarıyla eğitildi ve kaydedildi.")
