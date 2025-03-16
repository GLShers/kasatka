import os
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa
from sklearn.preprocessing import LabelEncoder
import time
import threading

# Параметры
SAMPLE_RATE = 6000
N_MFCC = 20
CONFIDENCE_THRESHOLD = 0.7  # Порог уверенности для предсказания
DROPOUT_RATE = 0.7  # Вероятность отключения нейронов
MODEL_PATH = 'sound_classifier_model.pth'  # Путь для сохранения модели

class SoundClassifier:
    def __init__(self, data_dir=None, valid_dir=None, test_dir=None):
        self.le = LabelEncoder()
        self.model = None
        self.audio_buffer_1 = []
        self.audio_buffer_2 = []
        self.running = True  # Флаг для остановки записи

        if data_dir and valid_dir and test_dir:
            self.data_dir = data_dir
            self.valid_dir = valid_dir
            self.test_dir = test_dir
            self.load_data()
            self.create_model()

    def load_data(self):
        X, y = [], []
        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.wav'):
                        file_path = os.path.join(label_dir, file)
                        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
                        mfcc = np.mean(mfcc.T, axis=0)  # Усреднение по временной оси
                        X.append(mfcc)
                        y.append(label)
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y)
        self.y_encoded = self.le.fit_transform(self.y)

        # Загрузка данных для валидации
        X_valid, y_valid = [], []
        for label in os.listdir(self.valid_dir):
            label_dir = os.path.join(self.valid_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.wav'):
                        file_path = os.path.join(label_dir, file)
                        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
                        mfcc = np.mean(mfcc.T, axis=0)  # Усреднение по временной оси
                        X_valid.append(mfcc)
                        y_valid.append(label)
        self.X_valid = np.array(X_valid, dtype=np.float32)
        self.y_valid = np.array(y_valid)
        self.y_encoded_valid = self.le.transform(self.y_valid)

        # Загрузка данных для тестирования
        X_test, y_test = [], []
        for label in os.listdir(self.test_dir):
            label_dir = os.path.join(self.test_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.wav'):
                        file_path = os.path.join(label_dir, file)
                        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
                        mfcc = np.mean(mfcc.T, axis=0)  # Усреднение по временной оси
                        X_test.append(mfcc)
                        y_test.append(label)
        self.X_test = np.array(X_test, dtype=np.float32)
        self.y_test = np.array(y_test)
        self.y_encoded_test = self.le.transform(self.y_test)

    def create_model(self):
        input_size = N_MFCC
        num_classes = len(np.unique(self.y_encoded))
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),  # Добавление Dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),  # Добавление Dropout
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs=10):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_encoded, dtype=torch.long)
        X_valid_tensor = torch.tensor(self.X_valid, dtype=torch.float32)
        y_valid_tensor = torch.tensor(self.y_encoded_valid, dtype=torch.long)

        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            print(f"Эпоха [{epoch+1}/{num_epochs}], Потеря: {loss.item():.4f}")

            # Валидация
            self.model.eval()
            with torch.no_grad():
                valid_outputs = self.model(X_valid_tensor)
                valid_loss = self.criterion(valid_outputs, y_valid_tensor)
                print(f"Валидационная потеря: {valid_loss.item():.4f}")

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'le_classes': self.le.classes_
        }, MODEL_PATH)
        print(f"Модель сохранена в {MODEL_PATH}")

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH)
            self.model = nn.Sequential(
                nn.Linear(N_MFCC, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(64, len(checkpoint['le_classes']))
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.le.classes_ = checkpoint['le_classes']
            print(f"Модель загружена из {MODEL_PATH}")
            return True
        print("Файл модели не найден")
        return False

    def test(self):
        self.model.eval()
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_encoded_test, dtype=torch.long)

        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            test_loss = self.criterion(test_outputs, y_test_tensor)
            print(f"Тестовая потеря: {test_loss.item():.4f}")

            # Вычисление точности
            predicted_classes = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted_classes == y_test_tensor).float().mean().item()
            print(f"Точность на тестовой выборке: {accuracy:.2f}")

    def predict(self, audio_data):
        if len(audio_data) < SAMPLE_RATE:
            return "Недостаточно данных для предсказания"
        mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Изменение формы для модели
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(mfcc)
            predicted_class = torch.argmax(prediction).item()
            confidence = torch.softmax(prediction, dim=1)[0][predicted_class].item()  # Уверенность в предсказании

            print(f"Предсказанный класс: {self.le.inverse_transform([predicted_class])[0]}, Уверенность: {confidence:.2f}")

            if confidence < CONFIDENCE_THRESHOLD:
                return "Не обнаружено"
            return self.le.inverse_transform([predicted_class])[0]

    def record_audio(self, mic_index, audio_buffer):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_data = np.frombuffer(indata, dtype=np.float32)
            audio_buffer.append(audio_data)

            # Вычисление уровня громкости
            volume_level = np.linalg.norm(audio_data)
            print(f"Уровень громкости микрофона {mic_index}: {volume_level:.2f}")

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback, device=mic_index):
                print(f"Запись с микрофона {mic_index} начата. Нажмите Ctrl+C для остановки.")
                while self.running:
                    time.sleep(1)
        except Exception as e:
            print(f"Ошибка при записи с микрофона {mic_index}: {e}")

    def start_recording(self):
        thread_1 = threading.Thread(target=self.record_audio, args=(0, self.audio_buffer_1))  # Индекс первого микрофона
        thread_2 = threading.Thread(target=self.record_audio, args=(1, self.audio_buffer_2))  # Индекс второго микрофона

        thread_1.start()
        thread_2.start()

        try:
            while self.running:
                if self.audio_buffer_1 or self.audio_buffer_2:  # Проверка на наличие данных в любом из буферов
                    if self.audio_buffer_1:
                        audio_data_1 = np.concatenate(self.audio_buffer_1)
                        self.audio_buffer_1.clear()
                        predicted_class_1 = self.predict(audio_data_1)
                        print(f"Результат предсказания с микрофона 1: {predicted_class_1}")

                    if self.audio_buffer_2:
                        audio_data_2 = np.concatenate(self.audio_buffer_2)
                        self.audio_buffer_2.clear()
                        predicted_class_2 = self.predict(audio_data_2)
                        print(f"Результат предсказания с микрофона 2: {predicted_class_2}")

                time.sleep(3)  # Проверка каждые 3 секунды
        except KeyboardInterrupt:
            self.running = False  # Установка флага остановки
            print("Запись остановлена.")

# Пример использования
if __name__ == "__main__":
    # Инициализация классификатора
    classifier = SoundClassifier()
    
    # Попытка загрузить существующую модель
    if not classifier.load_model():
        # Если модели нет - обучаем и сохраняем
        classifier = SoundClassifier('Learning', 'Valid', 'Test')
        classifier.train(num_epochs=50)
        classifier.test()
        classifier.save_model()
    
    # Запуск записи звука
    classifier.start_recording()