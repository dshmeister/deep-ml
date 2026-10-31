
def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    if len(a[0]) != len(b):
        return -1

    result = []

    for row in a:
        total = 0

        for j in range(len(b)):
            elem = row[j] * b[j]
            total += elem

        result.append(total)

    return result
