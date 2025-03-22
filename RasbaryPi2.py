import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import warnings
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
DROPOUT_RATE = 0.6
DATA_DIRS = {
    'train': 'train',
    'valid': 'valid',
    'test': 'test'
}

class SoundClassifier:
    def __init__(self):
        self.le = LabelEncoder()
        self.model = None
        self.running = True
        self.lock = threading.Lock()
        self.audio_buffers = {port: np.array([], dtype=np.float32) for port in PORTS}
        self.last_predictions = {}
        self.sample_rates = {}
        
        # Инициализация путей к данным
        self.data_dir = DATA_DIRS['train']
        self.valid_dir = DATA_DIRS['valid']
        self.test_dir = DATA_DIRS['test']
        
        if not self.load_model():  # Попытка загрузки модели
            self.init_training()    # Запуск обучения если модель не найдена
            
        self.init_network()

    # 🔄 Перемещен метод save_model выше load_model
    def save_model(self):
        """Сохранение модели на диск"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'le_classes': self.le.classes_,
                'sample_rates': self.sample_rates,
                'input_size': N_MFCC,
                'num_classes': len(self.le.classes_)
            }, MODEL_PATH)
            print(f"Модель сохранена в {MODEL_PATH}")
        except Exception as e:
            print(f"Ошибка сохранения: {str(e)}")
            raise

    def load_model(self):
        """Загрузка модели с диска"""
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(
                    MODEL_PATH,
                    map_location='cpu',
                    weights_only=False
                )
                self.le.classes_ = checkpoint['le_classes']
                self.sample_rates = checkpoint.get('sample_rates', {})
                
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

    def init_training(self):
        """Инициализация процесса обучения при отсутствии модели"""
        print("\n" + "="*60)
        print("Предупреждение: Модель не найдена! Запуск обучения...")
        print("="*60 + "\n")
        
        try:
            self.load_data()
            self.create_model()
            self.train(num_epochs=250)
            self.save_model()  # 🔄 Исправленный вызов
            print("\nОбучение успешно завершено!\n")
            self.load_model()
        except Exception as e:
            print(f"Критическая ошибка при обучении: {str(e)}")
            exit(1)

    def load_data(self):
        def load_from_dir(dir_path):
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Директория {dir_path} не найдена")
                
            X, y = [], []
            file_count = 0
            start_time = time.time()
            
            try:
                for label in os.listdir(dir_path):
                    if not self.running:
                        raise KeyboardInterrupt
                        
                    label_dir = os.path.join(dir_path, label)
                    if not os.path.isdir(label_dir):
                        print(f"Пропускаем {label_dir} - не директория")
                        continue
                        
                    print(f"Обработка класса: {label}")
                    files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
                    if not files:
                        print(f"Внимание: нет .wav файлов в {label_dir}")
                        continue
                        
                    for file in files:
                        try:
                            if not self.running:
                                raise KeyboardInterrupt
                                
                            file_path = os.path.join(label_dir, file)
                            file_count += 1
                            
                            audio, sr = librosa.load(file_path, sr=None, mono=True)
                            if sr != TARGET_SAMPLE_RATE:
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                                
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                mfcc = librosa.feature.mfcc(
                                    y=audio,
                                    sr=TARGET_SAMPLE_RATE,
                                    n_mfcc=N_MFCC,
                                    n_fft=2048,
                                    hop_length=512
                                )
                                
                            X.append(np.mean(mfcc.T, axis=0))
                            y.append(label)
                            
                            if file_count % 10 == 0:
                                elapsed = time.time() - start_time
                                print(f"Обработано {file_count} файлов ({elapsed:.1f} сек)")
                                
                        except KeyboardInterrupt:
                            print("\nПрерывание загрузки данных...")
                            self.running = False
                            raise
                            
                        except Exception as e:
                            print(f"Ошибка обработки {file_path}: {str(e)}")
                            continue
                            
            except KeyboardInterrupt:
                print("\nЗагрузка данных прервана пользователем")
                self.running = False
                raise
                
            print(f"Успешно загружено {file_count} файлов")
            return np.array(X, dtype=np.float32), np.array(y)

        try:
            for path in [self.data_dir, self.valid_dir, self.test_dir]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Директория {path} не существует")

            print("\nЗагрузка тренировочных данных...")
            self.X, self.y = load_from_dir(self.data_dir)
            self.y_encoded = self.le.fit_transform(self.y)
            
            print("\nЗагрузка валидационных данных...")
            self.X_valid, self.y_valid = load_from_dir(self.valid_dir)
            self.y_encoded_valid = self.le.transform(self.y_valid)
            
            print("\nЗагрузка тестовых данных...")
            self.X_test, self.y_test = load_from_dir(self.test_dir)
            self.y_encoded_test = self.le.transform(self.y_test)
            
            print("\nСтатистика датасета:")
            self._print_dataset_stats("Тренировочные", self.y)
            self._print_dataset_stats("Валидационные", self.y_valid)
            self._print_dataset_stats("Тестовые", self.y_test)
            
        except KeyboardInterrupt:
            print("\nПолное прерывание загрузки данных")
            self.running = False
            raise
            
        except Exception as e:
            print(f"Критическая ошибка при загрузке данных: {str(e)}")
            self.running = False
            raise

    def _print_dataset_stats(self, name, labels):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name} данные:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} примеров")
        print(f"Всего: {len(labels)} примеров\n")

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

    def train(self, num_epochs=250):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_encoded, dtype=torch.long)
    
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        learning_rates = []
    
        try:
            for epoch in range(num_epochs):
                if not self.running:
                    raise KeyboardInterrupt
                
            # Обучение
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, y_tensor)
                loss.backward()
                self.optimizer.step()
            
            # Расчет точности тренировки
                _, preds = torch.max(outputs, 1)
                correct = (preds == y_tensor).sum().item()
                train_accuracy = correct / len(y_tensor) * 100
            
                train_losses.append(loss.item())
                train_accuracies.append(train_accuracy)
            
            # Валидация
                self.model.eval()
                with torch.no_grad():
                    valid_outputs = self.model(torch.tensor(self.X_valid, dtype=torch.float32))
                    valid_loss = self.criterion(valid_outputs, torch.tensor(self.y_encoded_valid, dtype=torch.long))
                
                # Расчет точности валидации
                    _, valid_preds = torch.max(valid_outputs, 1)
                    valid_correct = (valid_preds == torch.tensor(self.y_encoded_valid, dtype=torch.long)).sum().item()
                    valid_accuracy = valid_correct / len(self.y_encoded_valid) * 100
                
                    valid_losses.append(valid_loss.item())
                    valid_accuracies.append(valid_accuracy)
            
                learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
                print(f"Эпоха [{epoch+1}/{num_epochs}] | "
                    f"Потеря: {loss.item():.4f} | Валидация: {valid_loss.item():.4f} | "
                    f"Точность: {train_accuracy:.2f}% | Валидационная точность: {valid_accuracy:.2f}%")

        except KeyboardInterrupt:
            print("\nОбучение прервано пользователем")
            self.running = False
        
        finally:
            self.analyze_training(
                train_losses, 
                valid_losses,
                train_accuracies,
                valid_accuracies,
                learning_rates
            )


    def analyze_training(self, train_losses, valid_losses, train_accuracies, valid_accuracies, learning_rates):
        print("\n" + "="*60)
        print("Детальный анализ обучения:")
        print("="*60)
    
    # Анализ точности
        max_train_acc = max(train_accuracies)
        min_train_acc = min(train_accuracies)
        final_train_acc = train_accuracies[-1]
    
        max_valid_acc = max(valid_accuracies)
        min_valid_acc = min(valid_accuracies)
        final_valid_acc = valid_accuracies[-1]
    
        print(f"\nТренировочная точность:")
        print(f"  Начальная: {train_accuracies[0]:.2f}%")
        print(f"  Максимальная: {max_train_acc:.2f}% (эпоха {train_accuracies.index(max_train_acc)+1})")
        print(f"  Минимальная: {min_train_acc:.2f}% (эпоха {train_accuracies.index(min_train_acc)+1})")
        print(f"  Финальная: {final_train_acc:.2f}%")
    
        print(f"\nВалидационная точность:")
        print(f"  Начальная: {valid_accuracies[0]:.2f}%")
        print(f"  Максимальная: {max_valid_acc:.2f}% (эпоха {valid_accuracies.index(max_valid_acc)+1})")
        print(f"  Минимальная: {min_valid_acc:.2f}% (эпоха {valid_accuracies.index(min_valid_acc)+1})")
        print(f"  Финальная: {final_valid_acc:.2f}%")
    
    # График точности
        print("\nДинамика точности:")
        for epoch, (train_acc, valid_acc) in enumerate(zip(train_accuracies, valid_accuracies)):
            diff = valid_acc - train_acc
            status = "↑↑" if diff > 5 else "↑↓" if diff < -5 else "≈"
            print(f"Эпоха {epoch+1:2d}: Train {train_acc:6.2f}% | Valid {valid_acc:6.2f}% | Разница {diff:+5.2f}% {status}")
    
    # Рекомендации
        print("\nРекомендации:")
        if final_valid_acc < 60:
            print("  ▸ Низкая точность! Попробуйте:")
            print("    - Увеличить размер датасета")
            print("    - Добавить аугментацию аудио")
            print("    - Изменить архитектуру модели")
        elif final_valid_acc < 80:
            print("  ▸ Средняя точность. Возможные улучшения:")
            print("    - Настроить гиперпараметры (LR, слои)")
            print("    - Добавить регуляризацию")
        else:
            print("  ▸ Отличный результат! Модель готова к использованию")
    
        print("="*60 + "\n")

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(
                    MODEL_PATH,
                    map_location='cpu',
                    weights_only=False
                )
                self.le.classes_ = checkpoint['le_classes']
                self.sample_rates = checkpoint.get('sample_rates', {})
                
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

    def init_network(self):
        """Инициализация сетевых сокетов"""
        self.sockets = {}
        for port in PORTS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('0.0.0.0', port))
            self.sockets[port] = sock
            print(f"Сокет инициализирован на порту {port}")

    def network_listener(self, port):
        """Прослушивание сетевого порта"""
        sock = self.sockets[port]
        print(f"Слушаем порт {port}...")
        while self.running:
            try:
                data, addr = sock.recvfrom(MAX_PACKET_SIZE)
                self.process_packet(port, data)
            except Exception as e:
                print(f"Ошибка на порту {port}: {str(e)}")

    def process_packet(self, port, data):
        """Обработка сетевых пакетов"""
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
            if len(audio) == 0:
                return {'class': 'error', 'confidence': 0, 'dBFS': -np.inf}
            
        # Проверка на тишину
            if np.all(np.abs(audio) < 1e-6):
                return {'class': 'silence', 'confidence': 0, 'dBFS': -np.inf}

        # Нормализация аудио
            audio = librosa.util.normalize(audio)
        
        # Вычисление MFCC
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=TARGET_SAMPLE_RATE,
                    n_mfcc=N_MFCC,
                    n_fft=2048,
                    hop_length=512
                )
            mfcc = np.mean(mfcc.T, axis=0)

        # Проверка на валидность MFCC
            if np.isnan(mfcc).any() or np.isinf(mfcc).any():
                return {'class': 'error', 'confidence': 0, 'dBFS': -np.inf}

        # Предсказание
            self.model.eval()  # Важно: переключаем в режим оценки
            with torch.no_grad():
                inputs = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                proba = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(proba, 1)
        
        # Расчет уровня звука
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

# Модификация метода process_audio()
    def process_audio(self):
        while self.running:
            try:
                predictions = {}
                for port in PORTS:
                    with self.lock:
                        buffer = self.audio_buffers[port]
                    # Накопление 2 секунд аудио перед обработкой
                        if len(buffer) >= BUFFER_SIZE:
                            audio = buffer[-BUFFER_SIZE:]  # Берем последние 2 секунды
                            self.audio_buffers[port] = np.array([], dtype=np.float32)
                            predictions[port] = self.predict(audio)
                
                if predictions:
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
        """Определение сектора по предсказаниям"""
        predictions = self.last_predictions
        required_devices = [1, 2, 3, 4]

        for dev_id in required_devices:
            if dev_id not in predictions:
                return "Не определено (недостаточно данных)"

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
    
    # Запуск сетевых потоков
    listeners = []
    for port in PORTS:
        thread = threading.Thread(
            target=classifier.network_listener,
            args=(port,),
            daemon=True
        )
        thread.start()
        listeners.append(thread)
    
    # Основной цикл обработки
    try:
        classifier.process_audio()
    finally:
        print("\nОстановка сервера...")
        for sock in classifier.sockets.values():
            sock.close()
        print("Сервер успешно остановлен.")