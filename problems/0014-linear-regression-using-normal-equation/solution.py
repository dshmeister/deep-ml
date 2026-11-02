import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Шаг 1: Переводим списки Python в удобные массивы NumPy
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Шаг 2: Транспонируем матрицу X (меняем строки и столбцы местами)
    # Подсказка: используйте свойство .T у массива X_arr
    X_transpose = X_arr.T
    
    # Шаг 3: Считаем матрицу (X^T * X)
    # Подсказка: для матричного умножения используйте оператор @
    X_transpose_X = X_transpose @ X_arr
    
    # Шаг 4: Находим обратную матрицу (X^T * X)^(-1)
    # Подсказка: используйте функцию инвертирования np.linalg.inv()
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    
    # Шаг 5: Считаем вектор (X^T * y)
    # Подсказка: снова используйте матричное умножение @
    X_transpose_y = X_transpose @ y_arr
    
    # Шаг 6: Находим итоговый вектор весов theta = (X^T * X)^(-1) @ (X^T * y)
    theta = X_transpose_X_inv @ (X_transpose_y)
    
    # Шаг 7: Округляем результат до 4 знаков после запятой
    # Подсказка: используйте np.round(..., 4)
    theta_rounded = np.round(theta, 4)
    
    # Шаг 8: Переводим массив NumPy обратно в обычный список Python
    # Подсказка: используйте метод .tolist() у округленного массива
    return theta
