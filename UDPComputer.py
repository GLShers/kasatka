import socket
import numpy as np
import sounddevice as sd
import threading
import struct
import time
from collections import deque

# Конфигурационные параметры
PORTS = [5000, 5001, 5002, 5003]  # Порты для 4 микрофонов
SAMPLE_RATE = 44100                # Частота дискретизации
CHUNK_SIZE = 512                   # Размер аудиоблока
BUFFER_SIZE = 50                   # Глубина буфера (в пакетах)
MAX_DELAY = 0.5                    # Максимальная допустимая задержка (сек)
OUTPUT_DEVICE = None               # Автовыбор устройства вывода

# Инициализация буферов для каждого порта
buffers = {port: deque(maxlen=BUFFER_SIZE) for port in PORTS}
lock = threading.Lock()  # Блокировка для синхронизации потоков

def find_output_device():
    """Автоматический выбор устройства вывода звука"""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            print(f"Выбрано аудиоустройство: {dev['name']} (ID: {i})")
            return i
    raise RuntimeError("Не найдено доступных аудиоустройств!")

def udp_receiver(port):
    """Прием данных с конкретного порта"""
    print(f"Слушаем порт {port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # Увеличиваем буфер
    sock.bind(('0.0.0.0', port))  # Обязательно привязываем сервер к всем интерфейсам

    while True:
        try:
            data, addr = sock.recvfrom(4096)  # Принимаем пакет до 4096 байт
            if len(data) < 8:
                continue  # Пропускаем неполные пакеты

            # Распаковываем временную метку и аудиоданные
            timestamp = struct.unpack('d', data[:8])[0]
            audio = np.frombuffer(data[8:], dtype=np.float32)

            # Проверка размера блока
            if len(audio) != CHUNK_SIZE:
                print(f"Порт {port}: неверный размер блока ({len(audio)})")
                continue

            with lock:
                buffers[port].append((timestamp, audio))
                
                # Очистка устаревших данных
                while buffers[port] and (time.time() - buffers[port][0][0]) > MAX_DELAY:
                    buffers[port].popleft()

        except Exception as e:
            print(f"Порт {port} ошибка: {e}")

def sync_playback():
    """Синхронизация и воспроизведение аудио"""
    print("Запуск синхронизации...")
    while True:
        try:
            with lock:
                # Фильтрация непустых буферов
                valid_buffers = [buf for buf in buffers.values() if buf]
                if not valid_buffers:
                    time.sleep(0.01)
                    continue

                # Поиск общей временной метки
                all_timestamps = [packet[0] for buf in valid_buffers for packet in buf]
                common_ts = min(all_timestamps)
                mixed = np.zeros(CHUNK_SIZE, dtype=np.float32)

                # Смешивание каналов
                for port in PORTS:
                    # Удаление старых данных
                    while buffers[port] and buffers[port][0][0] < common_ts:
                        buffers[port].popleft()

                    # Добавление данных с совпадающей меткой
                    if buffers[port] and buffers[port][0][0] == common_ts:
                        ts, audio = buffers[port].popleft()
                        mixed[:len(audio)] += audio

                        # Вывод уровня громкости
                        rms = np.sqrt(np.mean(audio ** 2))
                        print(f"Порт {port}: уровень звука {rms:.4f}", end=' | ')

                print()  # Новая строка после вывода уровней

            # Воспроизведение смешанного сигнала
            if np.any(mixed):
                try:
                    sd.play(mixed * 0.8, samplerate=SAMPLE_RATE, device=OUTPUT_DEVICE)
                except Exception as e:
                    print(f"Ошибка воспроизведения: {e}")

            time.sleep(0.01)

        except Exception as e:
            print(f"Ошибка синхронизации: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    # Инициализация устройства вывода
    OUTPUT_DEVICE = find_output_device()

    # Тест аудиоустройства
    print("\nТест аудиоустройства...")
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, SAMPLE_RATE)).astype(np.float32)
    try:
        sd.play(test_signal, samplerate=SAMPLE_RATE, device=OUTPUT_DEVICE)
        sd.wait()
    except Exception as e:
        print(f"Ошибка теста аудиоустройства: {e}")

    # Запуск потоков приема
    for port in PORTS:
        threading.Thread(target=udp_receiver, args=(port,), daemon=True).start()

    # Запуск потока воспроизведения
    threading.Thread(target=sync_playback, daemon=True).start()

    # Основной цикл
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
        sd.stop()
