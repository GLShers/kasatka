import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from sklearn.preprocessing import LabelEncoder
import time
import threading
import socket
import struct

# Конфигурация
PORTS = [5000, 5001, 5002, 5003]
TARGET_SAMPLE_RATE = 44100
N_MFCC = 40
MODEL_PATH = 'sound_classifier_model.pth'
BUFFER_SIZE = 44100 * 2  # 2 секунды аудио
MAX_PACKET_SIZE = 4096

class SoundClassifier:
    def __init__(self):
        self.le = LabelEncoder()
        self.model = None
        self.running = True
        self.lock = threading.Lock()
        self.audio_buffers = {port: np.array([], dtype=np.float32) for port in PORTS}
        self.last_predictions = {}  # Для хранения последних результатов
        self.load_model()
        self.init_network()

    def init_network(self):
        """Инициализация сетевых сокетов"""
        self.sockets = {}
        for port in PORTS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('0.0.0.0', port))
            self.sockets[port] = sock
            print(f"Сокет инициализирован на порту {port}")

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location='cpu')
                self.le.classes_ = checkpoint['le_classes']
                self.model = nn.Sequential(
                    nn.Linear(N_MFCC, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(64, len(self.le.classes_))
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print("Модель успешно загружена")
            except Exception as e:
                print(f"Ошибка загрузки модели: {str(e)}")
                raise

    def network_listener(self, port):
        """Поток для приема данных с конкретного порта"""
        sock = self.sockets[port]
        print(f"Слушаем порт {port}...")
        while self.running:
            try:
                data, addr = sock.recvfrom(MAX_PACKET_SIZE)
                self.process_packet(port, data)
            except Exception as e:
                print(f"Ошибка на порту {port}: {str(e)}")

    def process_packet(self, port, data):
        """Обработка входящего пакета"""
        try:
            timestamp = struct.unpack('d', data[:8])[0]
            audio_chunk = np.frombuffer(data[8:], dtype=np.float32)
            
            with self.lock:
                self.audio_buffers[port] = np.concatenate([
                    self.audio_buffers[port], 
                    audio_chunk
                ])[-BUFFER_SIZE:]
                
        except Exception as e:
            print(f"Ошибка обработки пакета: {str(e)}")

    def predict(self, audio):
        try:
            if np.all(audio == 0):
                return {'class': 'error', 'confidence': 0, 'dBFS': -np.inf}

            audio = librosa.util.normalize(audio)
            
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=TARGET_SAMPLE_RATE,
                n_mfcc=N_MFCC,
                n_fft=2048,
                hop_length=512
            )
            mfcc = np.mean(mfcc.T, axis=0)

            if np.isnan(mfcc).any() or np.isinf(mfcc).any():
                print("Обнаружены некорректные значения в MFCC")
                return {'class': 'error', 'confidence': 0, 'dBFS': -np.inf}

            inputs = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                proba = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(proba, 1)
            
            rms = np.sqrt(np.mean(audio**2))
            dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf

            return {
                'class': self.le.inverse_transform([pred.item()])[0],
                'confidence': conf.item(),
                'dBFS': dBFS
            }
            
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            return {'class': 'error', 'confidence': 0, 'dBFS': -np.inf}

    def process_audio(self):
        """Обработка аудио из буферов"""
        while self.running:
            try:
                predictions = {}
                for port in PORTS:
                    with self.lock:
                        buffer = self.audio_buffers[port]
                        if len(buffer) >= TARGET_SAMPLE_RATE:
                            audio = buffer[:TARGET_SAMPLE_RATE]
                            self.audio_buffers[port] = buffer[TARGET_SAMPLE_RATE:]
                            predictions[port] = self.predict(audio)
                
                if predictions:
                    # Сохраняем предсказания с преобразованием портов в ID устройств
                    self.last_predictions = {
                        i+1: pred for i, (port, pred) in enumerate(predictions.items())
                    }
                    
                    sector = self.determine_sector()
                    print("\n" + "="*60)
                    print(f"{'Система мониторинга дронов':^60}")
                    print("="*60)
                    print(f"\nОпределенный сектор: \033[1m{sector}\033[0m\n")
                    
                    for port, pred in predictions.items():
                        status = "ДРОН ОБНАРУЖЕН!" if pred['class'] == 'class1' else "Фоновый шум"
                        confidence = f"{pred['confidence']:.1%}".rjust(8)
                        dBFS = f"{pred['dBFS']:+.1f} dBFS".rjust(12)
                        
                        color = "\033[92m" if pred['class'] == 'class1' else "\033[93m"
                        reset = "\033[0m"
                        
                        print(f"Порт {port}:")
                        print(f"{color}├─ Статус: {status}{reset}")
                        print(f"├─ Уровень достоверности: {confidence}")
                        print(f"└─ Уровень звука:    {dBFS}")
                    print("="*60)
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Ошибка обработки: {str(e)}")

    def determine_sector(self):
        """Определение сектора на основе последних предсказаний"""
        predictions = self.last_predictions
        required_devices = [1, 2, 3, 4]

        # Проверка наличия данных
        for dev_id in required_devices:
            if dev_id not in predictions:
                return "Не определено (недостаточно данных)"

        # Парсинг классов
        device_classes = {}
        for dev_id in required_devices:
            pred = predictions[dev_id]
            if pred['class'] == 'error':
                return "Ошибка в данных устройства"
            
            try:
                class_num = int(pred['class'].replace('class', ''))
                device_classes[dev_id] = class_num
            except:
                return f"Ошибка класса: {pred['class']}"

        # Предопределенные условия
        conditions = [
            (device_classes[1] == 1 and device_classes[2] == 1 and 
             device_classes[4] == 1 and device_classes[3] == 2, "СВЕРХУ-СЛЕВА"),
            (device_classes[1] == 1 and device_classes[2] == 1 and 
             device_classes[3] == 1 and device_classes[4] == 2, "СВЕРХУ-СПРАВА"),
            (device_classes[1] == 1 and device_classes[3] == 1 and 
             device_classes[4] == 1 and device_classes[2] == 2, "СНИЗУ"),
            (device_classes[2] == 1 and device_classes[3] == 1 and 
             device_classes[4] == 1 and device_classes[1] == 2, "Ошибка конфигурации"),
            (device_classes[2] == 2 and device_classes[3] == 2 and 
             device_classes[4] == 2 and device_classes[1] == 1, "Ошибка конфигурации")
        ]

        for condition, sector in conditions:
            if condition:
                return sector

        # Обработка всех class 1
        if all(cls == 1 for cls in device_classes.values()):
            sound_levels = {dev_id: predictions[dev_id]['dBFS'] for dev_id in required_devices}
            max_device = max(sound_levels, key=lambda k: sound_levels[k])
            remaining_devices = sorted([d for d in required_devices if d != max_device])
            
            combination = ''.join(map(str, remaining_devices))
            sectors = {
                '123': "СВЕРХУ-СПРАВА",
                '134': "СНИЗУ",
                '124': "СВЕРХУ-СЛЕВА",
                '234': "ОШИБКА КОНФИГУРАЦИИ"
            }
            
            return sectors.get(combination, f"Неизвестная комбинация: {combination}")

        return "Неопределенный сектор"

if __name__ == "__main__":
    classifier = SoundClassifier()
    
    # Запуск потоков для каждого порта
    listeners = []
    for port in PORTS:
        thread = threading.Thread(
            target=classifier.network_listener,
            args=(port,),
            daemon=True
        )
        thread.start()
        listeners.append(thread)
    
    # Запуск обработки аудио
    try:
        classifier.process_audio()
    finally:
        print("\nОстановка сервера...")
        for sock in classifier.sockets.values():
            sock.close()
        print("Сервер успешно остановлен.")