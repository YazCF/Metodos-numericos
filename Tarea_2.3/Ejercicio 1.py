import numpy as np
import matplotlib.pyplot as plt

# Función original
def f(x):
    return x**3 - 6*x**2 + 11*x - 6

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

# Método de Bisección
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iteraciones = []
    c_old = a
    
    for i in range(max_iter):
        c = (a + b) / 2
        
        error_abs = abs(c - c_old)
        error_rel = abs((c - c_old)/c) if c != 0 else 0
        error_cuad = (c - c_old)**2
        
        iteraciones.append((i+1, c, error_abs, error_rel, error_cuad))
        
        if abs(func(c)) < tol:
            break
        
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
        
        c_old = c
    
    return c, iteraciones

# Selección de tres puntos de interpolación
x0 = 1.5
x1 = 2.2
x2 = 3.0
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante
# mediante interpolacion de Lagrange
x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
# en el intervalo inducido por los puntos donde se hace la interpolacion
root, iteraciones = bisect(
    lambda x: lagrange_interpolation(x, x_points, y_points),
    x0, x2
)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 6x^2 + 11x - 6", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")  # Guarda la imagen
plt.show()

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.4f}")

print("\nIter | x_n | Error abs | Error rel | Error cuad")
for it in iteraciones:
    print(f"{it[0]:4d} | {it[1]:.6f} | {it[2]:.2e} | {it[3]:.2e} | {it[4]:.2e}")

errores_abs = [it[2] for it in iteraciones]
errores_rel = [it[3] for it in iteraciones]
errores_cuad = [it[4] for it in iteraciones]

plt.figure()
plt.plot(range(1,len(errores_abs)+1), errores_abs, label="Error absoluto")
plt.plot(range(1,len(errores_rel)+1), errores_rel, label="Error relativo")
plt.plot(range(1,len(errores_cuad)+1), errores_cuad, label="Error cuadrático")
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.title("Evolución de errores - Ejercicio 1")
plt.legend()
plt.grid(True)
plt.show()