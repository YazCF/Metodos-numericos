import numpy as np
import matplotlib.pyplot as plt

# Definir el sistema de ecuaciones del ejemplo
A = np.array([[12, -2, 1, 0, 0, 0, 0],
              [-3, 18, -4, 2, 0, 0, 0],
              [1, -2, 16, -1, 1, 0, 0],
              [0, 2, -1, 11, -3, 1, 0],
              [0, 0, -2, 4, 15, -2, 1],
              [0, 0, 0, 1, -3, 2, 13]])

b = np.array([20, 35, -5, 19, -12, 25])

# Solución "exacta" en el sentido de mínimos cuadrados
sol_exacta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# Criterio de paro
tolerancia = 1e-6
max_iter = 100

# Implementación del método de Jacobi
def jacobi(A, b, tol, max_iter):
    n = A.shape[0]   # número de ecuaciones
    x = np.zeros(n)  # Aproximación inicial
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    print("Iteración\tError absoluto\tError relativo\tError cuadrático")
    
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(A.shape[1]) if j != i and j < n)
            x_new[i] = (b[i] - suma) / A[i, i]
        
        # Calcular errores respecto a la solución de mínimos cuadrados
        error_abs = np.linalg.norm(x_new - sol_exacta[:n], ord=1)
        error_rel = error_abs / np.linalg.norm(sol_exacta[:n], ord=1)
        error_cuad = np.linalg.norm(x_new - sol_exacta[:n], ord=2)
        
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        print(f"{k+1}\t{error_abs:.6f}\t{error_rel:.6f}\t{error_cuad:.6f}")
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        x = x_new
    
    return x, errores_abs, errores_rel, errores_cuad, k+1

# Ejecutar el método de Jacobi
sol_aprox, errores_abs, errores_rel, errores_cuad, iteraciones = jacobi(A, b, tolerancia, max_iter)
diferencia = np.abs(sol_exacta[:len(sol_aprox)] - sol_aprox)

print("\nComparación de soluciones:")
print(f"Solución aproximada: {sol_aprox}")
print(f"Solución mínimos cuadrados: {sol_exacta}")
print("Diferencia absoluta:", diferencia)

# Graficar los errores
plt.figure(figsize=(8,6))
plt.plot(range(1, iteraciones+1), errores_abs, label="Error absoluto", marker='o')
plt.plot(range(1, iteraciones+1), errores_rel, label="Error relativo", marker='s')
plt.plot(range(1, iteraciones+1), errores_cuad, label="Error cuadrático", marker='d')
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Convergencia de los errores en el método de Jacobi")
plt.legend()
plt.grid()
plt.savefig("errores_jacobi_ejercicio3.png")
plt.show()
