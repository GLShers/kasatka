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
from scipy.optimize import minimize

# Разрешаем необходимые глобальные объекты для загрузки модели  
add_safe_globals([numpy.core.multiarray._reconstruct])

# Параметры
TARGET_SAMPLE_RATE = 16000
N_MFCC = 40  # Увеличиваем количество MFCC-коэффициентов
CONFIDENCE_THRESHOLD = 0.8
DROPOUT_RATE = 0.6  # Увеличиваем dropout для регуляризации
MODEL_PATH = 'sound_classifier_model.pth'
CHANNELS = 1
DEVICES = [1, 2]  # Укажите ID микрофонов
SPEED_OF_SOUND = 343.0  # м/с
MIC1_POS = (0, 0)       # Координаты 1-го микрофона
MIC2_POS = (0.5, 0)     # Координаты 2-го микрофона

# Новая функция триангуляции (добавлена вне класса)
def triangulate_source(mic1, mic2, delta_t, initial_guess=(50, 50)):
    """
    Находит координаты источника звука методом триангуляции
    
    Параметры:
    mic1 (tuple): Координаты первого микрофона (x, y) в метрах
    mic2 (tuple): Координаты второго микрофона (x, y) в метрах
    delta_t (float): Разница во времени прихода сигнала (в секундах)
    initial_guess (tuple): Начальное предположение для координат источника
    
    Возвращает:
    tuple: Найденные координаты (x, y)
    """
    def equation(point):
        x, y = point
        distance1 = np.sqrt((x - mic1[0])**2 + (y - mic1[1])**2)
        distance2 = np.sqrt((x - mic2[0])**2 + (y - mic2[1])**2)
        return abs((distance1 - distance2) - SPEED_OF_SOUND * delta_t)
    
    result = minimize(equation, initial_guess, method='L-BFGS-B')
    
    if result.success:
        return result.x
    else:
        raise ValueError("Оптимизация не сошлась")


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

        # Инициализация данных только если директории предоставлены
        if data_dir and valid_dir and test_dir:
            self._init_data(data_dir, valid_dir, test_dir)

    def _init_data(self, data_dir, valid_dir, test_dir):
        self.data_dir = data_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        try:
            self.load_data()
            self.create_model()
        except Exception as e:
            print(f"Ошибка инициализации данных: {str(e)}")
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
                            n_mfcc=N_MFCC,
                            n_fft=2048,
                            hop_length=512
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
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            verbose=True
        )

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
                    nn.Linear(checkpoint['input_size'], 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
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
                conf_matrix[t, p] += 1
                
            class_acc = conf_matrix.diag() / conf_matrix.sum(1)
            
        print("\nРезультаты тестирования модели:")
        print(f"Тестовая потеря: {test_loss.item():.4f}")
        print(f"Общая точность: {accuracy.item()*100:.2f}%")
        print("\nТочность по классам:")
        for i, class_name in enumerate(self.le.classes_):
            acc = class_acc[i].item() * 100
            print(f"{class_name}: {acc:.2f} %")

    def record_audio(self, buffer_name, device_id):
        def callback(indata, frames, time, status):
            if status:
                print(f"Ошибка ввода: {status}")
            with self.lock:
                self.audio_buffers[buffer_name].extend(indata[:, 0])

        try:
            with sd.InputStream(
                device=device_id,
                samplerate=self.sample_rates[device_id],
                channels=CHANNELS,
                callback=callback,
                blocksize=int(self.sample_rates[device_id] * 0.5)
            ):
                while self.running:
                    sd.sleep(100)
        except Exception as e:
            print(f"Ошибка записи с устройства {device_id}: {str(e)}")

    def start_recording(self):
        self.print_audio_info()
        self.audio_buffers = {f"mic_{id}": [] for id in DEVICES}
        
        threads = []
        for device_id in DEVICES:
            if device_id not in self.devices_info:
                print(f"Устройство {device_id} не найдено, пропускаем")
                continue
                
            thread = threading.Thread(
                target=self.record_audio,
                args=(f"mic_{device_id}", device_id),
                daemon=True
            )
            thread.start()
            threads.append(thread)

        try:
            while self.running:
                detections = {}
                
                for buffer_name in list(self.audio_buffers.keys()):
                    device_id = int(buffer_name.split('_')[1])
                    required_length = self.sample_rates[device_id] * 1
                    
                    if len(self.audio_buffers[buffer_name]) >= required_length:
                        with self.lock:
                            audio_data = np.array(self.audio_buffers[buffer_name][:required_length])
                            self.audio_buffers[buffer_name] = self.audio_buffers[buffer_name][required_length:]
                        
                        result = self.predict(audio_data, device_id)
                        print(f"Результат ({buffer_name}): {result}")
                        
                        if "class1" in result and "Не обнаружено" not in result:
                            detections[device_id] = time.time()

                if len(detections) == 2:
                    delta_t = abs(detections[DEVICES[0]] - detections[DEVICES[1]])
                    
                    if delta_t < 0.5:
                        try:
                            # Используем последние успешные координаты как начальное предположение
                            coords = triangulate_source(
                                mic1=MIC1_POS,
                                mic2=MIC2_POS,
                                delta_t=delta_t,
                                initial_guess=self.last_coords
                            )
                            x, y = coords
                            self.last_coords = (x, y)  # Обновляем последние координаты
                            print(f"\nТриангуляция успешна! Координаты: ({x:.2f}, {y:.2f}) м")
                            print(f"Разница времени: {delta_t:.4f} сек")
                            print(f"Следующее начальное предположение: ({x:.2f}, {y:.2f})\n")
                            
                        except ValueError as e:
                            print(f"\nОшибка триангуляции: {str(e)}")
                            print(f"Используем предыдущие координаты: {self.last_coords}\n")
                        except Exception as e:
                            print(f"\nНеизвестная ошибка: {str(e)}\n")
                    
                    self.last_detection = {device_id: None for device_id in DEVICES}
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.running = False
            print("\nОстановка записи...")
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1)
            print("Все потоки остановлены")


if __name__ == "__main__":
    try:
        classifier = SoundClassifier()
        
        # Пытаемся загрузить модель
        if not classifier.load_model():
            print("Файл модели не найден или поврежден. Требуется обучение.")
            data_dirs = ['Learning', 'Valid', 'Test']
            # Проверяем существование всех директорий
            if not all(os.path.isdir(d) for d in data_dirs):
                print("Ошибка: отсутствуют данные для обучения. Проверьте директории Learning, Valid, Test.")
                exit(1)
                
            print("Обучение новой модели...")
            try:
                classifier = SoundClassifier(
                    data_dir='Learning',
                    valid_dir='Valid',
                    test_dir='Test'
                )
                classifier.train(num_epochs=200)
                classifier.save_model()
                classifier.test()
            except Exception as e:
                print(f"Критическая ошибка при обучении: {str(e)}")
                exit(1)
        
        # Запуск записи
        classifier.start_recording()
        
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")