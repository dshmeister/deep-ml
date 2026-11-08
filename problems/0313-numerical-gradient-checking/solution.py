import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Принудительно приводим x к типу float для корректных математических сдвигов
    x = x.astype(np.float64)
    
    # Создаем массив для численного градиента той же формы, что и x
    numerical_grad = np.zeros_like(x)
    
    # Итератор для последовательного обхода всех элементов массива
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # Шаг вперед
        x[idx] = old_value + epsilon
        fx_plus = f(x)
        
        # Шаг назад
        x[idx] = old_value - epsilon
        fx_minus = f(x)
        
        # Восстановление исходного значения
        x[idx] = old_value
        
        # Вычисление численной производной по центральной разности
        numerical_grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        it.iternext()
        
    # Вычисление относительной ошибки
    norm_diff = np.linalg.norm(analytical_grad - numerical_grad)
    norm_sum = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    
    # Обработка случая, когда оба градиента нулевые
    if norm_sum == 0.0:
        relative_error = 0.0
    else:
        relative_error = norm_diff / norm_sum
        
    return numerical_grad, relative_error
