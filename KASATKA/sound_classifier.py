# sound_classifier.py
import os
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa
from sklearn.preprocessing import LabelEncoder
import time
import threading
import warnings
from torch.serialization import add_safe_globals
import numpy.core.multiarray
from config import *
from triangulation import triangulate_source

# Разрешаем необходимые глобальные объекты для загрузки модели
add_safe_globals([numpy.core.multiarray._reconstruct])

class SoundClassifier:
    def __init__(self, data_dir=None, valid_dir=None, test_dir=None):
        self.le = LabelEncoder()
        self.model = None
        self.audio_buffers = {}
        self.running = True
        self.devices_info = self.list_microphones()
        self.sample_rates = self.find_best_sample_rates()
        self.lock = threading.Lock()
        self.last_detection = {device_id: None for device_id in DEVICES}
        self.last_coords = (300.0, 300.0)
        
        if data_dir and valid_dir and test_dir:
            self.data_dir = data_dir
            self.valid_dir = valid_dir
            self.test_dir = test_dir
            try:
                self.load_data()
                self.create_model()
            except Exception as e:
                print(f"Ошибка инициализации: {str(e)}")
                exit(1)

    def list_microphones(self):
        return {
            i: sd.query_devices(i)
            for i in DEVICES
            if i < len(sd.query_devices())
        }

    def find_best_sample_rates(self):
        sample_rates = {}
        for device_id in DEVICES:
            try:
                device_info = sd.query_devices(device_id)
                supported_rates = [
                    8000, 11025, 16000, 22050, 
                    32000, 44100, 48000, 96000
                ]
                
                best_rate = 16000
                for rate in supported_rates:
                    try:
                        sd.check_input_settings(
                            device=device_id,
                            samplerate=rate,
                            channels=CHANNELS
                        )
                        best_rate = rate
                        break
                    except sd.PortAudioError:
                        continue
                
                sample_rates[device_id] = best_rate
                print(f"Устройство {device_id} использует частоту {best_rate} Hz")
                
            except Exception as e:
                print(f"Ошибка устройства {device_id}: {str(e)}")
                sample_rates[device_id] = 16000
        
        return sample_rates

    def print_audio_info(self):
        print("Используемые микрофоны:")
        for device_id in DEVICES:
            if device_id in self.devices_info:
                dev = self.devices_info[device_id]
                print(f"ID: {device_id}, Название: {dev['name']}")
                print(f"  Частота: {self.sample_rates[device_id]} Hz")
                print(f"  Каналы: {dev['max_input_channels']}\n")

    def load_data(self):
        def load_from_dir(dir_path):
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Директория {dir_path} не найдена")
                
            X, y = [], []
            for label in os.listdir(dir_path):
                label_dir = os.path.join(dir_path, label)
                if not os.path.isdir(label_dir):
                    continue
                    
                wav_files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
                if not wav_files:
                    print(f"Внимание: нет .wav файлов в {label_dir}")
                    continue
                    
                for file in wav_files:
                    file_path = os.path.join(label_dir, file)
                    try:
                        audio, sr = librosa.load(file_path, sr=None)
                        
                        if sr != TARGET_SAMPLE_RATE:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                            
                        mfcc = librosa.feature.mfcc(
                            y=audio, 
                            sr=TARGET_SAMPLE_RATE, 
                            n_mfcc=N_MFCC
                        )
                        mfcc = np.mean(mfcc.T, axis=0)
                        X.append(mfcc)
                        y.append(label)
                    except Exception as e:
                        print(f"Ошибка загрузки {file_path}: {str(e)}")
                        continue
                        
            if not X:
                raise ValueError(f"Нет данных в {dir_path}")
            return np.array(X, dtype=np.float32), np.array(y)

        print("Загрузка обучающих данных...")
        self.X, self.y = load_from_dir(self.data_dir)
        self.y_encoded = self.le.fit_transform(self.y)
        
        print("Загрузка валидационных данных...")
        self.X_valid, self.y_valid = load_from_dir(self.valid_dir)
        self.y_encoded_valid = self.le.transform(self.y_valid)
        
        print("Загрузка тестовых данных...")
        self.X_test, self.y_test = load_from_dir(self.test_dir)
        self.y_encoded_test = self.le.transform(self.y_test)
        
        print("\nДанные успешно загружены:")
        print(f"Обучающие образцы: {len(self.X)}")
        print(f"Валидационные образцы: {len(self.X_valid)}")
        print(f"Тестовые образцы: {len(self.X_test)}\n")

    def create_model(self):
        input_size = N_MFCC
        num_classes = len(self.le.classes_)
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        print("Модель успешно создана\n")

    def train(self, num_epochs=10):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_encoded, dtype=torch.long)
        
        print("Начало обучения...")
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                valid_outputs = self.model(torch.tensor(self.X_valid, dtype=torch.float32))
                valid_loss = self.criterion(valid_outputs, torch.tensor(self.y_encoded_valid, dtype=torch.long))
                
            print(f"Эпоха [{epoch+1}/{num_epochs}] | "
                  f"Потеря: {loss.item():.4f} | "
                  f"Валидационная потеря: {valid_loss.item():.4f}")

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'le_classes': self.le.classes_,
            'sample_rates': self.sample_rates,
            'input_size': N_MFCC,
            'num_classes': len(self.le.classes_)
        }, MODEL_PATH)
        print(f"Модель сохранена в {MODEL_PATH}")

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(
                    MODEL_PATH,
                    map_location='cpu',
                    weights_only=False
                )
                
                self.le.classes_ = checkpoint['le_classes']
                self.sample_rates = checkpoint.get('sample_rates', {id: 16000 for id in DEVICES})
                
                self.model = nn.Sequential(
                    nn.Linear(checkpoint['input_size'], 128),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    nn.Linear(64, checkpoint['num_classes'])
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Модель загружена из {MODEL_PATH}")
                return True
                
            except Exception as e:
                print(f"Ошибка загрузки модели: {str(e)}")
                return False
        print("Файл модели не найден")
        return False

    def predict(self, audio_data, device_id):
        with self.lock:
            try:
                target_rate = self.sample_rates[device_id]
                
                if target_rate != TARGET_SAMPLE_RATE:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=target_rate, 
                        target_sr=TARGET_SAMPLE_RATE
                    )
                
                audio_data = librosa.util.normalize(audio_data)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mfcc = librosa.feature.mfcc(
                        y=audio_data, 
                        sr=TARGET_SAMPLE_RATE, 
                        n_mfcc=N_MFCC,
                        n_fft=2048,
                        hop_length=512
                    )
                
                mfcc = np.mean(mfcc.T, axis=0)
                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    self.model.eval()
                    prediction = self.model(mfcc_tensor)
                    probabilities = torch.softmax(prediction, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    confidence_percent = confidence.item() * 100
                    
                    if confidence_percent < CONFIDENCE_THRESHOLD * 100:
                        return f"Не обнаружено ({confidence_percent:.2f}%)"
                    
                    class_name = self.le.inverse_transform([predicted.item()])[0]
                    return f"{class_name} ({confidence_percent:.2f}%)"
            
            except Exception as e:
                print(f"Ошибка предсказания: {str(e)}")
                return "Ошибка"

    def test(self):
        self.model.eval()
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_encoded_test, dtype=torch.long)

        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            test_loss = self.criterion(test_outputs, y_test_tensor)
            
            predicted = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted == y_test_tensor).float().mean()
            
            conf_matrix = torch.zeros(len(self.le.classes_), len(self.le.classes_))
            for t, p in zip(y_test_tensor, predicted):
                conf _matrix[t.item()][p.item()] += 1

            print(f"Тестовая потеря: {test_loss.item():.4f}")
            print(f"Точность: {accuracy.item() * 100:.2f}%")
            print("Матрица ошибок:")
            print(conf_matrix)

    def record_audio(self, duration, device_id):
        audio_buffer = []
        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_buffer.append(indata.copy())

        with sd.InputStream(samplerate=self.sample_rates[device_id], channels=CHANNELS, callback=callback):
            print(f"Запись аудио с устройства {device_id} на {duration} секунд...")
            sd.sleep(int(duration * 1000))

        return np.concatenate(audio_buffer, axis=0)

    def start_recording(self, duration, device_id):
        audio_data = self.record_audio(duration, device_id)
        coords = triangulate_source((0, 0), (100, 0), 0.1)  # Пример использования функции триангуляции
        prediction = self.predict(audio_data, device_id)
        print(f"Предсказание: {prediction}, Координаты: {coords}")

# Дополнительные функции и методы могут быть добавлены ниже при необходимости