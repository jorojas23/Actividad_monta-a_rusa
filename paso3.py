import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.special import legendre
import csv  # Para leer datos de CSV
# import pandas as pd # Otra opción para leer datos

# --- Paso 1: Trazador Cúbico Sujeto ---
class CubicSplineInterpolator:
    def __init__(self, x_data, y_data, bc_type='natural'):
        self.x_data = x_data
        self.y_data = y_data
        self.bc_type = bc_type
        self.spline = CubicSpline(x_data, y_data, bc_type=bc_type)

    def evaluate(self, x_vals):
        return self.spline(x_vals)

    def plot(self, x_vals, title="Trayectoria de la Montaña Rusa"):
        y_vals = self.evaluate(x_vals)
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_data, self.y_data, 'o', label='Puntos de control')
        plt.plot(x_vals, y_vals, label='Trazador cúbico')
        plt.title(title)
        plt.xlabel('Posición horizontal (x)')
        plt.ylabel('Altura (y)')
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Paso 2: Polinomio de Mínimos Cuadrados ---
class LeastSquaresPolynomial:
    def __init__(self, x_data, y_data, degree=2):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.degree = degree
        self.coefficients = np.polyfit(self.x_data, self.y_data, self.degree)
        self.polynomial = np.poly1d(self.coefficients)

    def evaluate(self, x_vals):
        x_vals = np.array(x_vals)
        return self.polynomial(x_vals)

    def plot(self, x_vals, title="Ajuste por Mínimos Cuadrados (Datos de Tensión/Compresión)"):
        y_vals = self.evaluate(x_vals)
        y_fit = self.evaluate(self.x_data)  # Valores ajustados en los puntos x_data
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_data, self.y_data, 'ro', label='Datos experimentales')
        plt.plot(x_vals, y_vals, 'b-', label=f'Polinomio de grado {self.degree}')
        
        # Dibujar líneas verticales desde los puntos experimentales hasta la curva ajustada
        for xi, yi, y_fit_i in zip(self.x_data, self.y_data, y_fit):
            plt.plot([xi, xi], [yi, y_fit_i], 'g--', alpha=0.5)
        
        plt.title(title)
        plt.xlabel('Deformación (mm)')
        plt.ylabel('Fuerza (kN)')
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Paso 3: Polinomios Ortogonales ---
class LegendrePlotter:
    def __init__(self, degree, x_range):
        self.degree = degree
        self.x_range = x_range

    def plot(self):
        x = np.linspace(self.x_range[0], self.x_range[1], 500)
        plt.figure(figsize=(10, 6))
        for i in range(self.degree + 1):
            Pn = legendre(i)
            plt.plot(x, Pn(x), label=f'P{i}(x)')
        plt.title('Polinomios de Legendre')
        plt.xlabel('x')
        plt.ylabel('Pn(x)')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        plt.grid()
        plt.show()

# --- Paso 4: Resolución de Ecuaciones ---
class SistemaEcuaciones:
    def __init__(self, A, b):
        self.A = np.array(A)
        self.b = np.array(b)

    def resolver(self):
        try:
            x = np.linalg.solve(self.A, self.b)
            return x
        except np.linalg.LinAlgError:
            return "El sistema no tiene solución única (puede ser singular o incompatible)."

if __name__ == "__main__":
    # --- Paso 1: Definir y graficar la trayectoria ---
    # Cargar puntos de control desde CSV (ejemplo)
    try:
        x_control = []
        y_control = []
        with open('trayectoria.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader) # Saltar la primera fila (encabezados)
            for row in reader:
                x_control.append(float(row[0]))
                y_control.append(float(row[1]))
        x_control = np.array(x_control)
        y_control = np.array(y_control)
    except FileNotFoundError:
        print("Archivo 'puntos_control.csv' no encontrado. Usando datos de ejemplo.")
        x_control = np.array([0, 1, 2, 3, 4, 5])
        y_control = np.array([0.5, 0.8, 1.0, 0.9, 1.2, 0.7])

    spline_interpolator = CubicSplineInterpolator(x_control, y_control, bc_type=((1, 0), (1, 0)))
    x_trayectoria = np.linspace(x_control.min(), x_control.max(), 200)
    spline_interpolator.plot(x_trayectoria, title="Trayectoria de la Montaña Rusa")

    # --- Paso 2: Ajustar polinomio a datos del material y graficar ---
    # Cargar datos del material desde CSV (ejemplo)
    try:
        x_material = []
        y_material = []
        with open('datos_tension.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader) # Saltar la primera fila (encabezados)
            for row in reader:
                x_material.append(float(row[0]))
                y_material.append(float(row[1]))
        x_material = np.array(x_material)
        y_material = np.array(y_material)
    except FileNotFoundError:
        print("Archivo 'datos_material.csv' no encontrado. Usando datos de ejemplo.")
        x_material = np.array([0, 1, 2, 3, 4])
        y_material = np.array([11.1, 3.5, 2.8, 4.2, 5.0])

    ls_polynomial = LeastSquaresPolynomial(x_material, y_material, degree=1)
    x_ajuste = np.linspace(x_material.min(), x_material.max(), 100)
    ls_polynomial.plot(x_ajuste)

    # --- Paso 3: Graficar Polinomios de Legendre (Optimización requerirá más lógica) ---
    legendre_plotter = LegendrePlotter(degree=3, x_range=[-1, 1]) # Rango típico para Legendre
    legendre_plotter.plot()
    print("Para la optimización con Polinomios de Legendre, necesitarás definir el tramo a optimizar y una función objetivo.")

    # --- Paso 4: Resolver sistema de ecuaciones para fuerzas ---
    # Definir la matriz A y el vector b (ESTO DEPENDE DE TU MODELO ESTRUCTURAL)
    A_fuerzas = [[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]]
    b_fuerzas = [15, 10, 10, 10]

    sistema_fuerzas = SistemaEcuaciones(A_fuerzas, b_fuerzas)
    soluciones_fuerzas = sistema_fuerzas.resolver()
    print("\nFuerzas en los puntos críticos (ejemplo):", soluciones_fuerzas)
