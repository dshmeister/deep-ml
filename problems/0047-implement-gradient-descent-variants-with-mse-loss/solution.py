import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    """
    m, n = X.shape
    # Делаем копию весов, чтобы работать с ними напрямую
    w = weights.copy()
    
    # Главный цикл по эпохам (полным проходам по датасету)
    for epoch in range(n_epochs):
        
        # === ВАРИАНТ 1: BATCH GRADIENT DESCENT ===
        if method == 'batch':
            # 1. Считаем предсказания для ВСЕЙ матрицы X
            predictions = X @ w
            # 2. Считаем ошибку для всего датасета
            error = predictions - y
            # 3. Считаем градиент по всему датасету (помним про деление на m)
            gradient = 2 * X.T @ error / m
            # 4. Обновляем веса
            w = w - learning_rate * gradient
            
        # === ВАРИАНТ 2: STOCHASTIC GRADIENT DESCENT ===
        elif method == 'stochastic':
            # запускаем вложенный цикл по строкам
            for i in range(X.shape[0]): # или просто по m
                # Считаем предсказания для i строки матрицы
                predictions_i = X[i] @ w
                error = predictions_i - y[i]
                gradient = 2 * X[i] * error
                w = w - gradient * learning_rate
            

                
        # === ВАРИАНТ 3: MINI-BATCH GRADIENT DESCENT ===
        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                preds_m = X_batch @ w
                error = preds_m - y_batch
                gradient = 2 * X_batch.T @ error / batch_size
                w = w - learning_rate * gradient

                
    return w
