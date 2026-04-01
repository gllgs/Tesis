#%% Paqueterías
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp

plt.style.use('seaborn-v0_8-dark')

# Ruta de guardado
ruta = '/Users/abrilgallegos/Desktop/Tesis2026/ImagenesTesis2026'

#%% SISTEMA
def Modelo(y, t, f_KLK, f_release, f_production, k_prod, k_dif, k_deg):
    Dif, Bas = y
    factor = (1 + f_KLK * Dif) / (1 + f_release * Dif)

    dDdt = k_dif * factor * Bas - k_deg * Dif
    dBdt = k_prod / (1 + f_production * Dif) - k_dif * factor * Bas

    return [dDdt, dBdt]

#%% EQUILIBRIO (más robusto)
def Equilibrio_pos(f_KLK, f_release, f_production, k_prod, k_dif, k_deg):
    Dif, Bas = sp.symbols('Dif Bas')

    eq1 = sp.Eq(k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas - k_deg * Dif, 0)
    eq2 = sp.Eq(k_prod / (1 + f_production * Dif) - k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas, 0)

    soluciones = sp.solve((eq1, eq2), (Dif, Bas), dict=True)

    for sol in soluciones:
        Dif_val = float(sol[Dif])
        Bas_val = float(sol[Bas])

        if Dif_val > 0 and Bas_val > 0:
            return [Dif_val, Bas_val]

    return None  # si no hay solución positiva

#%% PARÁMETROS
params_best = {
    "f_{KLK}": 0.40509243993417915,
    "f_{release}": 0.02474920863730841,
    "f_{production}": 3.660472590236123,
    "k_{prod}": 1.530472558632681,
    "k_{dif}": 1.4399311031304896,
    "k_{deg}": 0.42859621128509695
}

# Variación
delta = 0.9

def rango(valor):
    return valor * (1 - delta), valor * (1 + delta)

#%% CONDICIONES INICIALES
y0 = [0.577, 0.194]
times = np.linspace(0, 10, 100)

# Solución base
solution_base = odeint(
    Modelo, y0, times,
    args=tuple(params_best.values())
)
D_base, B_base = solution_base.T

#%% FUNCIÓN GENERAL PARA GRAFICAR
def graficar_variacion(param_name):
    p_min, p_max = rango(params_best[param_name])

    # Copias de parámetros
    params_min = params_best.copy()
    params_max = params_best.copy()

    params_min[param_name] = p_min
    params_max[param_name] = p_max

    # Simulaciones
    sol_min = odeint(Modelo, y0, times, args=tuple(params_min.values()))
    sol_max = odeint(Modelo, y0, times, args=tuple(params_max.values()))

    D_min, B_min = sol_min.T
    D_max, B_max = sol_max.T

    # Gráfica
    plt.figure(figsize=(10, 6))

    # mínimo
    plt.plot(times, D_min, color='darkseagreen', linewidth=1)
    plt.plot(times, B_min, color='cornflowerblue', linewidth=1)

    # base
    plt.plot(times, D_base, color='darkseagreen', label='D(t)', linewidth=2)
    plt.plot(times, B_base, color='cornflowerblue', label='B(t)', linewidth=2)

    # máximo
    plt.plot(times, D_max, color='darkseagreen', linewidth=3)
    plt.plot(times, B_max, color='cornflowerblue', linewidth=3)

    plt.title(f'Dinámica celular variando ${param_name}$')
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Concentración')
    plt.legend()
    plt.grid(True)

    # Guardar
    nombre_archivo = f'var_{param_name}.png'
    ruta_completa = os.path.join(ruta, nombre_archivo)
    plt.savefig(ruta_completa, dpi=600)
    #plt.show()
    plt.close()

#%% GENERAR TODAS LAS GRÁFICAS
for param in params_best.keys():
    graficar_variacion(param)
# %%
