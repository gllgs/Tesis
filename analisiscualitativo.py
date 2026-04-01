#%% Paqueterias necesarias
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import optuna
from scipy.integrate import odeint
import matplotlib.lines as mlines
import sympy as sp
from sympy import Matrix
plt.style.use('seaborn-v0_8-dark')
ruta= '/Users/abrilgallegos/Desktop/Tesis2026/ImagenesTesis2026'
#%% SISTEMA
# Definición del sistema de ecuaciones diferenciales
def Modelo(y, t, f_KLK, f_release, f_production,k_prod, k_dif, k_deg):
    Dif = y[0]
    Bas = y[1]
    dDdt = k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas - k_deg * Dif
    dBdt = k_prod*(1 / (1 + f_production * Dif)) - k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas
    return [dDdt, dBdt]
# Definir las funciones de las derivadas
def ceroclina_D(Dif, Bas,f_KLK, f_release, f_production,k_prod,k_dif, k_deg):
    return k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas - k_deg * Dif
    
def ceroclina_B(Dif, Bas, f_KLK, f_release, f_production,k_prod,k_dif, k_deg):
    return k_prod*(1 / (1 + f_production * Dif)) - k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas
    
def Equilibrio_pos(f_KLK, f_release, f_production,  k_prod,k_dif, k_deg):
    Dif = sp.symbols('Dif')
    Bas = sp.symbols('Bas')
    ecuacion1 = sp.Eq(k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas - k_deg * Dif,0)
    ecuacion2 = sp.Eq(k_prod*(1 / (1 + f_production * Dif)) - k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas,0)
    solucion = solve((ecuacion1, ecuacion2), (Dif, Bas))
    sol1 = [float(solucion[0][0]),float(solucion[0][1])]
    sol2= [float(solucion[1][0]),float(solucion[1][1])]
    if sol1[0] and sol1[1]>0:
        return sol1
    if sol2[0]>0 and sol2[1]>0:
        return sol2

   
#%%
#####Valores de parámetro arbitrarios######
f_KLK = 3
f_release = 4
f_production = 3
k_prod = 2
k_deg = 5
k_dif =1

####Condiciones iniciales#########
D0 = 1
B0 = 0.5
y0=[D0, B0]

tiempo_final = 15
times = np.linspace(0, tiempo_final, 100000)

solution = odeint(Modelo, y0, times, args=(f_KLK, f_release, f_production,k_prod, k_dif, k_deg))
Dpoints, Bpoints = solution.T

Point = Equilibrio_pos(f_KLK, f_release, f_production,  k_prod,k_dif, k_deg)
cord_D = Point[0]
cord_B = Point[1]


# Definir los posibles puntos para D y B
d = np.linspace(0, 0.5, 100)  # Crea un arreglo de 100 puntos de L que varían de 0 a 2.0.
b = np.linspace(0, 2.0, 100)  # Crea un arreglo de 100 puntos de pL que varían de 0 a 0.04.

# Crear una cuadrícula de valores de D y B
D, B = np.meshgrid(d, b)  # Genera una malla 2D con los valores de L y pL en las coordenadas respectivas.

# Calcular las trayectorias en el espacio de fase
dD = ceroclina_D(D,B, f_KLK, f_release, f_production, k_prod, k_dif, k_deg)
dB = ceroclina_B(D,B,f_KLK, f_release, f_production, k_prod, k_dif, k_deg)

#%% Dínamica de células
plt.figure(figsize=(10, 6))
plt.plot(times, Dpoints, color='darkseagreen', label='D(t)')
plt.plot(times, Bpoints, color='cornflowerblue', label='B(t)')
plt.axhline(y=cord_D, color='gray', linestyle='--')
plt.axhline(y=cord_B, color='gray', linestyle='--')
plt.title('Dinámica de céluas basales y diferenciadas')
plt.xlabel('Tiempo (días)')
plt.ylabel('Concentración')
plt.legend()
plt.grid(True)
nombre_archivo = 'DinamicaInicial.png'
ruta_completa = os.path.join(ruta, nombre_archivo)
plt.savefig(ruta_completa, dpi=600)
plt.close() 
# %% ESPACIO FASE
# Graficar el espaciofase
plt.streamplot(D, B, dD, dB, color='darkgray', density=1)
plt.scatter(cord_D,cord_B,color='black', zorder=5)
plt.xlabel('Céulas diferenciadas (D)')
plt.ylabel('Céulas basales (B)')
plt.title('Espacio fase del modelo',fontsize=14)

# Graficar las ceroclinas
plt.contour(D, B, dD, levels=[0], colors='darkseagreen', linewidths=2, linestyles='solid')
plt.contour(D, B, dB, levels=[0], colors='cornflowerblue', linewidths=2, linestyles='solid')

# Crear objetos Line2D para representar las ceroclinas
# Añadir leyenda con los objetos proxy
# Crear objetos Line2D para representar las ceroclíneas
legendD = mlines.Line2D([], [], color='darkseagreen', label='Ceroclina de D')
# Crea un objeto para la leyenda que representa la ceroclina de D (células diferenciadas) con color 'darkseagreen'.
legendB = mlines.Line2D([], [], color='cornflowerblue', label='Ceroclina de B')
# Crea un objeto para la leyenda que representa la ceroclina de B (células basales) con color 'cornflowerblue'.

# Añadir leyenda con los objetos proxy
plt.legend(handles=[legendD, legendB], loc=(0.70, 0.8))
# Añade la leyenda a la gráfica, posicionada en las coordenadas (0.70, 0.8) con los objetos de ceroclina de D y B.
nombre_archivo = 'EspacioFase.png'
ruta_completa = os.path.join(ruta, nombre_archivo)
plt.savefig(ruta_completa, dpi=600)
plt.close() 
# %%
