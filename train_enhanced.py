import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

# Etiket isimleri
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 1. VERİYİ YÜKLE
df = pd.read_csv("dataset/fer2013_training_onehot.csv")
X = df.iloc[:, :2304].values.astype("float32") / 255.0
y = df.iloc[:, 2304:].values.astype("float32")
X = X.reshape(-1, 48, 48, 1)

# 2. VERİYİ BÖL
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. VERİ ARTIRIMI
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# 4. MODEL MİMARİSİ
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 duygu sınıfı
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. ERKEN DURDURMA
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 6. MODELİ EĞİT
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# 7. KAYDET
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")
print("✅ Yeni model başarıyla eğitildi ve kaydedildi.")
