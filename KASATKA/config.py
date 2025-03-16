# config.py
import os
import torch

TARGET_SAMPLE_RATE = 16000
N_MFCC = 20
CONFIDENCE_THRESHOLD = 0.7
DROPOUT_RATE = 0.3
MODEL_PATH = 'sound_classifier_model.pth'
CHANNELS = 1
DEVICES = [1, 2]  # Укажите ID микрофонов
SPEED_OF_SOUND = 343.0  # м/с
MIC1_POS = (0, 0)       # Координаты 1-го микрофона
MIC2_POS = (0.5, 0)     # Координаты 2-го микрофона