import numpy as np
import matplotlib.pyplot as plt

def least_squares_tdoa():
    """
    Вычисление координат источника звука методом наименьших квадратов (TDOA) 
    с итерационным уточнением и визуализацией гипербол.
    """
    # Ввод данных
    print("Введите координаты микрофонов (в метрах):")
    m1 = np.array([float(input("M1 x: ")), float(input("M1 y: "))])
    m2 = np.array([float(input("M2 x: ")), float(input("M2 y: "))])
    m3 = np.array([float(input("M3 x: ")), float(input("M3 y: "))])
    
    print("\nВведите TDOA (в секундах):")
    tdoa12 = float(input("TDOA (M1-M2): "))
    tdoa13 = float(input("TDOA (M1-M3): "))
    tdoa23 = float(input("TDOA (M2-M3): "))
    
    c = 343.0  # Скорость звука (м/с)
    delta_d12 = c * tdoa12
    delta_d13 = c * tdoa13
    delta_d23 = c * tdoa23

    # Итерационный метод
    x_guess = np.array([0.0, 0.0])
    for _ in range(10):  # Увеличим число итераций
        A = []
        b = []
        for (mi, mj, delta_d) in [(m1, m2, delta_d12), (m1, m3, delta_d13)]:  # Используем только 2 пары
            xi, yi = mi
            xj, yj = mj
            di = np.sqrt((x_guess[0] - xi)**2 + (x_guess[1] - yi)**2)
            dj = np.sqrt((x_guess[0] - xj)**2 + (x_guess[1] - yj)**2)
            # Корректные коэффициенты
            a = [(x_guess[0] - xj)/dj - (x_guess[0] - xi)/di,
                 (x_guess[1] - yj)/dj - (x_guess[1] - yi)/di]
            # Корректный вектор b
            b_val = ( (xj**2 + yj**2 - xi**2 - yi**2) - delta_d**2 ) / 2 + delta_d * (di - dj)
            A.append(a)
            b.append(b_val)
        
        A = np.array(A)
        b = np.array(b)
        dx, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if np.linalg.norm(dx) < 1e-6:  # Критерий остановки
            break
        x_guess += dx

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Сетка для расчетов
    x = np.linspace(min(m1[0], m2[0], m3[0])-10, max(m1[0], m2[0], m3[0])+10, 500)
    y = np.linspace(min(m1[1], m2[1], m3[1])-10, max(m1[1], m2[1], m3[1])+10, 500)
    X, Y = np.meshgrid(x, y)
    
    # Гиперболы
    F12 = np.sqrt((X - m1[0])**2 + (Y - m1[1])**2) - np.sqrt((X - m2[0])**2 + (Y - m2[1])**2) - delta_d12
    F13 = np.sqrt((X - m1[0])**2 + (Y - m1[1])**2) - np.sqrt((X - m3[0])**2 + (Y - m3[1])**2) - delta_d13
    
    # Отрисовка
    ax.contour(X, Y, F12, levels=[0], colors='red', linewidths=2, linestyles='--', label='M1-M2 Hyperbola')
    ax.contour(X, Y, F13, levels=[0], colors='blue', linewidths=2, linestyles='--', label='M1-M3 Hyperbola')
    
    # Микрофоны и решение
    ax.scatter([m1[0]], [m1[1]], color='black', s=100, marker='^', label='M1')
    ax.scatter([m2[0]], [m2[1]], color='gray', s=100, marker='^', label='M2')
    ax.scatter([m3[0]], [m3[1]], color='brown', s=100, marker='^', label='M3')
    ax.scatter(x_guess[0], x_guess[1], color='lime', s=200, marker='*', edgecolor='black', label='Решение')
    
    # Настройка графика
    ax.set_xlabel('X координата (м)', fontsize=12)
    ax.set_ylabel('Y координата (м)', fontsize=12)
    ax.set_title('Локализация источника по TDOA', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    least_squares_tdoa()