import copy

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    """
    # 1. Считаем детерминант по исходной матрице
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if det == 0:
        return None
        
    # 2. Создаем новую матрицу для результата, чтобы не портить оригинал
    matrix_new = copy.deepcopy(matrix)
    
    # 3. Записываем расчеты ТОЛЬКО в новую матрицу matrix_new
    # Главная диагональ (меняются местами)
    matrix_new[0][0] = matrix[1][1] / det
    matrix_new[1][1] = matrix[0][0] / det
    
    # Побочная диагональ (остаются на местах, но меняют знак)
    matrix_new[0][1] = (-1) * matrix[0][1] / det
    matrix_new[1][0] = (-1) * matrix[1][0] / det
    
    # 4. Возвращаем именно новую матрицу
    return matrix_new
