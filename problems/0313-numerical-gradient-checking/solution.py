import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    """
    # Шаг 1: Создаем массив для численного градиента такой же формы, как x
    numerical_grad = np.zeros_like(x, dtype=float)
    
    # Шаг 2: Создаем итератор для обхода всех элементов массива x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index  # Получаем текущий многомерный индекс
        
        old_value = x[idx]     # Сохраняем исходное значение координаты
        
        # Вычисляем f(x + epsilon)
        x[idx] = old_value + epsilon
        fx_plus = f(x)
        
        # Вычисляем f(x - epsilon)
        x[idx] = old_value - epsilon
        fx_minus = f(x)
        
        # Возвращаем координату x в исходное состояние
        x[idx] = old_value
        
        # Вычисляем численную производную (центральная разность)
        numerical_grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        it.iternext()  # Переходим к следующей координате
        
    # Шаг 3: Вычисляем Евклидовы нормы векторов для формулы ошибки
    norm_diff = np.linalg.norm(numerical_grad - analytical_grad)
    norm_num = np.linalg.norm(numerical_grad)
    norm_analytic = np.linalg.norm(analytical_grad)
    
    # Шаг 4: Расчет относительной ошибки с защитой от деления на ноль
    denominator = norm_num + norm_analytic
    
    if denominator == 0:
        relative_error = 0.0
    else:
        relative_error = norm_diff / denominator
        
    return numerical_grad, relative_error
