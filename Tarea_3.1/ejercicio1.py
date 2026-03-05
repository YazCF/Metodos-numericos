import numpy as np
import matplotlib.pyplot as plt

def gauss_elimination(A, b):
    n = len(b)
    A = A.copy()
    b = b.copy()
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        # Eliminación hacia adelante
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Sustitución regresiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

A = np.array([[3, 2, -1, 4], [5, -3, 2, -1], [-1, 4, -2, 3], [2, -1, 3, 5]], dtype=float)
b = np.array([10, 5, -3, 8], dtype=float)

# Resolución del sistema
sol = gauss_elimination(A, b)
residual = b - A @ sol

# Imprimir la solución
print("Solución:", sol)
print("Residual:", residual)

plt.bar(range(1, len(residual)+1), residual)
plt.xlabel("Ecuación")
plt.ylabel("Residual")
plt.title("Error residual por ecuación (Eliminación Gaussiana)")
plt.show()