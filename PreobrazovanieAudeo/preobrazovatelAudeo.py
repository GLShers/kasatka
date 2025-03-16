import torchaudio
import os
from pathlib import Path

def resample_audio(input_dir, output_dir, target_sample_rate=16000):
    """
    Преобразует все .wav файлы в директории к целевому sample rate.
    
    Параметры:
        input_dir (str): Путь к папке с исходными файлами
        output_dir (str): Путь для сохранения преобразованных файлов
        target_sample_rate (int): Целевая частота дискретизации (по умолчанию 16000)
    """
    # Создаем выходную директорию если ее нет
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Получаем список всех .wav файлов
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for filename in file_list:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Загружаем аудио
        waveform, orig_sample_rate = torchaudio.load(input_path)
        
        # Пропускаем файлы с уже правильным sample rate
        if orig_sample_rate == target_sample_rate:
            print(f"Skipping {filename} (already {target_sample_rate} Hz)")
            continue
            
        # Инициализируем ресемплер
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=target_sample_rate
        )
        
        # Применяем преобразование
        resampled_waveform = resampler(waveform)
        
        # Сохраняем результат
        torchaudio.save(
            uri=output_path,
            src=resampled_waveform,
            sample_rate=target_sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )
        print(f"Converted: {filename} ({orig_sample_rate} -> {target_sample_rate} Hz)")

# Использование
input_directory = "/path/to/input/folder"
output_directory = "/path/to/output/folder"

resample_audio(input_directory, output_directory)