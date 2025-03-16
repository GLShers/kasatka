# triangulation.py
import numpy as np
from scipy.optimize import minimize
from config import SPEED_OF_SOUND

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
    
    # Минимизируем разницу между расчетной и измеренной разницей расстояний
    result = minimize(
        equation, 
        initial_guess, 
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': False}
    )
    
    if result.success:
        return tuple(result.x.astype(float))
    else:
        raise ValueError(
            f"Оптимизация не сошлась: {result.message}\n"
            f"Последние координаты: {result.x}"
        )

# Дополнительные функции можно добавить ниже при необходимости