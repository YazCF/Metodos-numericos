import numpy as np
import matplotlib.pyplot as plt


# Función original
def f(x):
    return np.sin(x) - x/2

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección con errores
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    approximations = []
    error_abs = []
    error_rel = []
    error_cuad = []
    
    for i in range(max_iter):
        c = (a + b) / 2
        approximations.append(c)
        
        if i > 0:
            ea = abs(c - approximations[i-1])
            er = ea / abs(c) if c != 0 else 0
            eq = ea**2
            
            error_abs.append(ea)
            error_rel.append(er)
            error_cuad.append(eq)
        
        if abs(func(c)) < tol or (b - a)/2 < tol:
            break
        
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    
    return c, approximations, error_abs, error_rel, error_cuad

# Puntos de interpolación
x0 = 0.0
x1 = 1.0
x2 = 2.0

x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio
x_vals = np.linspace(x0, x2, 200)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio
root, approximations, error_abs, error_rel, error_cuad = bisect(
    lambda x: lagrange_interpolation(x, x_points, y_points),
    x0, x2
)

# Imprimir resultados
print(f"\nLa raíz aproximada usando interpolación es: {root:.6f}\n")

print("Iter | x_n | Error abs | Error rel | Error cuad")
for i in range(1, len(approximations)):
    print(f"{i:4d} | "
          f"{approximations[i]:.6f} | "
          f"{error_abs[i-1]:.2e} | "
          f"{error_rel[i-1]:.2e} | "
          f"{error_cuad[i-1]:.2e}")

# Gráfica función e interpolación
plt.figure(figsize=(8,6))
plt.plot(x_vals, f(x_vals), linestyle='dashed', label="f(x) = sin(x) - x/2")
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.axhline(0, linestyle='--')
plt.axvline(root, linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.title("Interpolación de Lagrange - Ejercicio 2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Gráfica de convergencia
plt.figure(figsize=(8,6))
plt.plot(range(1, len(error_abs)+1), error_abs, label="Error absoluto")
plt.plot(range(1, len(error_rel)+1), error_rel, label="Error relativo")
plt.plot(range(1, len(error_cuad)+1), error_cuad, label="Error cuadrático")

plt.yscale("log")
plt.title("Evolución de errores - Ejercicio 2")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()