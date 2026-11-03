import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    # Запускаем цикл обучения на заданное количество итераций
    for _ in range(iterations):
        # 1. Посчитать предсказания модели (h).
        # Формула: h = X умножить на theta
        # предсказания модели = матрица признаков * веса
        predictions = X @ theta

        # 2. Посчитать ошибку (разность предсказаний и реальности).
        # Формула: error = predictions - y
        error = predictions - y

        # 3. Вычислить градиент функции потерь.
        # Формула: gradient = (1 / m) * (X транспонированная умножить на error)
        gradient = (X.T @ error) / m

        # 4. Обновить веса theta, сделав шаг против градиента.
        # Формула: theta = theta - alpha * gradient
        theta = theta - alpha * gradient

    return theta.flatten()
