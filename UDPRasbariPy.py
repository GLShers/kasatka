import socket
import sounddevice as sd
import threading
import time
import struct

# Конфигурация
HOST_PC_IP = '192.168.1.1'  # Замените на реальный IP сервера
PORTS = [5000, 5001, 5002, 5003]  # Фиксированные порты для 4 микрофонов
SAMPLE_RATE = 44100
CHUNK_SIZE = 512
MAX_PACKET_SIZE = 4096  # Максимальный размер UDP-пакета

def get_input_devices():
    """Получаем и проверяем ровно 4 микрофона"""
    devices = sd.query_devices()
    input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
    
    if len(input_devices) < 4:
        raise RuntimeError(f"Требуется 4 микрофона! Найдено: {len(input_devices)}")
    
    print("\nДоступные микрофоны:")
    for i, dev_id in enumerate(input_devices[:4]):
        print(f"Микрофон {i+1}: {sd.query_devices(dev_id)['name']}")
    
    return input_devices[:4]  # Берем первые 4 устройства

def audio_streamer(device_id, port):
    """Поток для записи и отправки аудио"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # Увеличиваем буфер отправки
        
        device_name = sd.query_devices(device_id)['name']
        print(f"\nЗапущен микрофон {device_name} на порту {port}")

        def callback(indata, frames, time_info, status):
            """Обработчик аудиопотока"""
            if status:
                print(f"Ошибка в потоке (порт {port}): {status}")
                return

            # Формируем пакет: [8 байт timestamp] + [аудиоданные]
            timestamp = struct.pack('d', time.time())
            audio_data = indata.tobytes()
            packet = timestamp + audio_data

            # Проверка размера пакета
            if len(packet) > MAX_PACKET_SIZE:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Пакет {len(packet)} байт (порт {port})")
                return

            # Отправка данных
            try:
                sock.sendto(packet, (HOST_PC_IP, port))
            except Exception as e:
                print(f"Ошибка отправки (порт {port}): {e}")

        # Настройка аудиопотока
        with sd.InputStream(
            device=device_id,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=1,
            callback=callback
        ):
            threading.Event().wait()  # Бесконечное ожидание

    except Exception as e:
        print(f"Критическая ошибка (порт {port}): {e}")
        sock.close()

if __name__ == "__main__":
    print("Инициализация клиента...")
    
    try:
        # Получаем ровно 4 микрофона
        microphones = get_input_devices()
        
        # Запускаем потоки для каждого микрофона
        threads = []
        for i, (dev_id, port) in enumerate(zip(microphones, PORTS)):
            thread = threading.Thread(
                target=audio_streamer,
                args=(dev_id, port),
                daemon=True,
                name=f"Microphone-{i+1}"
            )
            thread.start()
            threads.append(thread)
            time.sleep(0.2)  # Задержка для инициализации

        print("\nКлиент запущен. Отправка данных на сервер...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nОстановка клиента...")
    except Exception as e:
        print(f"Ошибка: {e}")
