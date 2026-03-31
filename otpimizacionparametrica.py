#%% Paqueterias necesarias
# Importamos las bibliotecas necesarias para el análisis y la optimización
import matplotlib.pyplot as plt          # Para generar gráficas
import numpy as np                       # Para manejo de arreglos y operaciones matemáticas
import optuna                            # Biblioteca para optimización de hiperparámetros
from scipy.integrate import odeint       # Integrador de ecuaciones diferenciales ordinarias
import sympy as sp                        # Matemática simbólica para resolver ecuaciones

# Establecemos el estilo de las gráficas (oscuro, similar a seaborn)
plt.style.use('seaborn-v0_8-dark')
ruta = '/Users/abrilgallegos/Desktop/Tesis2026/ImagenesTesis2026'  # Ruta opcional para guardar imágenes

# %%
# Definición del sistema de ecuaciones diferenciales
# Este modelo describe la dinámica de dos poblaciones celulares:
#   Dif (diferenciadas) y Bas (basales).
# Los parámetros son:
#   f_KLK, f_release, f_production : factores de regulación (adimensionales)
#   k_prod : tasa de producción de células basales
#   k_dif  : tasa de diferenciación
#   k_deg  : tasa de degradación/muerte de células diferenciadas
def Modelo(y, t, f_KLK, f_release, f_production, k_prod, k_dif, k_deg):
    # y es el vector de estado [Dif, Bas]
    Dif = y[0]
    Bas = y[1]
    
    # Ecuación para las células diferenciadas:
    #   diferenciación (término positivo) - degradación (término negativo)
    dDdt = k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas - k_deg * Dif
    
    # Ecuación para las células basales:
    #   producción (en función de Dif) - diferenciación (pérdida hacia Dif)
    dBdt = k_prod * (1 / (1 + f_production * Dif)) - k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas
    
    return [dDdt, dBdt]   # Retorna las derivadas

# %%
def Equilibrio_pos(f_KLK, f_release, f_production, k_prod, k_dif, k_deg):
    """
    Encuentra el punto de equilibrio del sistema con ambas variables positivas.
    Retorna una lista [D_eq, B_eq] si existe, o None si no hay.
    """
    # Definimos símbolos para las variables, restringiendo a valores reales positivos
    D, B = sp.symbols('D B', real=True, positive=True)
    
    # Definimos las ecuaciones de equilibrio (derivadas = 0)
    eq1 = k_dif * ((1 + f_KLK * D) / (1 + f_release * D)) * B - k_deg * D
    eq2 = k_prod * (1 / (1 + f_production * D)) - k_dif * ((1 + f_KLK * D) / (1 + f_release * D)) * B
    
    try:
        # Resolvemos el sistema simbólicamente; pedimos las soluciones como diccionario
        soluciones = sp.solve([eq1, eq2], (D, B), dict=True)
    except:
        return None  # Si la resolución falla (por ejemplo, sistema demasiado complejo), retornamos None

    # Filtramos las soluciones que sean reales y positivas
    for sol in soluciones:
        D_val = sol[D]
        B_val = sol[B]
        # Verificamos que ambas sean números reales y mayores que cero
        if D_val.is_real and B_val.is_real and D_val > 0 and B_val > 0:
            # Convertimos a float (para usar en cálculos numéricos)
            try:
                return [float(D_val), float(B_val)]
            except:
                continue  # Si falla la conversión, probamos la siguiente solución
    return None  # No se encontró ninguna solución positiva


# %%
# Datos objetivo (experimentales) proporcionados por Hoffman (2015)
# time_data: tiempos en días en los que se tomaron las mediciones
# D_data: concentración de células diferenciadas observada
time_data = np.array([0, 0.98, 1.98, 2.98, 5.98, 7.98, 9.98])
D_data = np.array([0.577, 0.711, 0.812, 0.845, 0.852, 0.864, 0.87])

# Condición inicial para células diferenciadas: tomamos el primer dato experimental
D_initial = D_data[0]

def forwardmap(theta):
    """
    Simula el modelo desde t=0 hasta los tiempos en time_data,
    usando D_initial (fijo, dato experimental) y B_initial calculado del equilibrio.
    Si no se puede calcular B_initial válido, retorna un arreglo de NaN.
    """
    # Desempaquetamos el vector de parámetros theta
    f_KLK, f_release, f_production, k_prod, k_dif, k_deg = theta
    
    # Calculamos el equilibrio positivo para estos parámetros
    eq = Equilibrio_pos(f_KLK, f_release, f_production, k_prod, k_dif, k_deg)
    if eq is None:
        # No hay equilibrio positivo: devolvemos NaN para que el error sea enorme y el optimizador descarte estos parámetros
        return np.full_like(time_data, np.nan)
    
    # La condición inicial para Basales la tomamos del equilibrio (suponemos que el sistema parte del equilibrio en basal)
    B_initial = eq[1]
    y0 = [D_initial, B_initial]
    
    try:
        # Integramos el sistema únicamente en los puntos de tiempo donde tenemos datos
        sol = odeint(Modelo, y0, time_data, args=(f_KLK, f_release, f_production, k_prod, k_dif, k_deg))
    except:
        # Si ocurre un error en la integración (por ejemplo, valores que llevan a inestabilidad), devolvemos NaN
        return np.full_like(time_data, np.nan)
    
    # Extraemos solo la variable de interés (células diferenciadas)
    D_sim, _ = sol.T
    return D_sim

# %%
# ##### Valores de parámetro arbitrarios (para prueba inicial) ######
# Asignamos valores de ejemplo para los parámetros (no optimizados)
f_KLK = 2
f_release = 1.5
f_production = 4
k_prod = 1.9
k_deg = 0.5
k_dif = 0.5

# Condiciones iniciales para la simulación de prueba
D0 = 1
B0 = 0.5
y0 = [D0, B0]

thet = [f_KLK,f_release,f_production,k_prod,k_dif, k_deg]
sim = forwardmap(thet)
plt.plot(time_data,sim,label='Simulación con parámetros al tanteo', color = 'darkseagreen')
plt.scatter(x = time_data, y= D_data, color='slategrey', label = 'Datos objetivo')
plt.title('Pre - optimización')
plt.xlabel('Tiempo (días)')
plt.ylabel('Células diferenciadas')
plt.grid()
plt.legend()
plt.show()


# %%
# Función objetivo para la optimización
def objective_function(theta):
    """
    Calcula el error cuadrático entre la simulación y los datos experimentales.
    Si la simulación devuelve NaN (porque no hay equilibrio válido), se asigna un error muy grande.
    """
    sim = forwardmap(theta)
    # Verificamos si hay NaN en la simulación
    if np.any(np.isnan(sim)):
        return 1e12   # Penalización enorme para que el optimizador evite estos parámetros
    # Error cuadrático total
    error = np.sum((sim - D_data)**2)
    return error

#%%
# Optimización con Optuna
def objective(trial):
    """
    Función que Optuna llamará en cada prueba (trial).
    Sugiere valores para cada parámetro dentro de rangos específicos y calcula el error.
    """
    # Sugerimos valores flotantes para cada parámetro en intervalos definidos
    f_KLK = trial.suggest_float('f_KLK', 0, 4)
    f_release = trial.suggest_float('f_release', 0, 3)
    f_production = trial.suggest_float('f_production', 2, 6)
    k_prod = trial.suggest_float('k_prod', 0, 4)
    k_dif = trial.suggest_float('k_dif', 0, 2)
    k_deg = trial.suggest_float('k_deg', 0, 2)
    
    # Lista de parámetros
    params = [f_KLK, f_release, f_production, k_prod, k_dif, k_deg]
    
    # Calculamos el error para esta combinación
    error = objective_function(params)
    return error

# Creamos un estudio de Optuna con dirección de minimización
study = optuna.create_study(direction='minimize')
# Ejecutamos la optimización con 1000 pruebas
study.optimize(objective, n_trials=100)

# Extraemos los mejores parámetros encontrados
optimal_params_optuna = [
    study.best_params['f_KLK'],
    study.best_params['f_release'],
    study.best_params['f_production'],
    study.best_params['k_prod'],
    study.best_params['k_dif'],
    study.best_params['k_deg']
]

# Calculamos el error final con los parámetros óptimos
Error = objective_function(optimal_params_optuna)
print("Parámetros óptimos (Optuna):", optimal_params_optuna)
print("Costo:", Error)

# %%
# Simulación final con los parámetros óptimos
# Desempaquetamos los parámetros óptimos
f_KLK_opt, f_release_opt, f_production_opt, k_prod_opt, k_dif_opt, k_deg_opt = optimal_params_optuna

# Calculamos el equilibrio para estos parámetros (nos dará también D de equilibrio, aunque no lo usaremos como inicial)
eq_opt = Equilibrio_pos(f_KLK_opt, f_release_opt, f_production_opt, k_prod_opt, k_dif_opt, k_deg_opt)
if eq_opt is None:
    print("Error: No se encontró equilibrio positivo con los parámetros óptimos.")
    # Podríamos detenernos aquí o manejar el error de otra forma
else:
    # Obtenemos las coordenadas del equilibrio (no usamos D_initial_opt como condición inicial, solo para graficar línea)
    D_initial_opt, B_initial_opt = eq_opt
    # Para la simulación usamos D_initial (primer dato experimental) y B_initial_opt (del equilibrio)
    y0_opt = [D_initial, B_initial_opt]

    # Definimos un tiempo final y una malla fina para una gráfica suave
    tf = 10
    times_fine = np.linspace(0, tf, 100)
    # Integramos con los parámetros óptimos
    solution = odeint(Modelo, y0_opt, times_fine,
                      args=(f_KLK_opt, f_release_opt, f_production_opt, k_prod_opt, k_dif_opt, k_deg_opt))
    D_opt, B_opt = solution.T

    # Gráfico de resultados
    plt.figure(figsize=(10, 6))
    plt.plot(times_fine, D_opt, color='darkseagreen', label='Células diferenciadas')
    plt.plot(times_fine, B_opt, color='cornflowerblue', label='Células basales')
    plt.scatter(time_data, D_data, label='Datos de Hoffman (2015)', color='slategrey')
    # Líneas horizontales que indican los valores de equilibrio (opcional, para referencia)
    plt.axhline(B_initial_opt, color='slategrey', linestyle='--', alpha=0.5)
    plt.axhline(D_initial_opt, color='slategrey', linestyle='--', alpha=0.5)
    plt.title('Modelo optimizado')
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Concentración celular')
    plt.legend()
    plt.grid(True)
    plt.show()



#%%% SIMULACIÓN CON VALORES ÓPTIMOS
f_KLK_BEST = 0.40509243993417915
f_release_BEST = 0.02474920863730841
f_production_BEST = 3.660472590236123
k_dif_BEST = 1.4399311031304896
k_deg_BEST = 0.42859621128509695
k_prod_BEST = 1.530472558632681
parametrosoptimos = (f_KLK_BEST, f_release_BEST, f_production_BEST, k_prod_BEST, k_dif_BEST, k_deg_BEST)
Costo_minimo = objective_function(parametrosoptimos)
print('El costo es: ' + str(Costo_minimo))
eq_BEST = Equilibrio_pos(f_KLK_BEST, f_release_BEST, f_production_BEST, k_prod_BEST, k_dif_BEST, k_deg_BEST)

D_initial_BEST, B_initial_BEST = eq_BEST
# Para la simulación usamos D_initial (primer dato experimental) y B_initial_BEST (del equilibrio)
y0_BEST = [D_initial, B_initial_BEST]

# Definimos un tiempo final y una malla fina para una gráfica suave
tf = 10
times_fine = np.linspace(0, tf, 100)
# Integramos con los parámetros óptimos
solution = odeint(Modelo, y0_BEST, times_fine,
                      args=(f_KLK_BEST, f_release_BEST, f_production_BEST, k_prod_BEST, k_dif_BEST, k_deg_BEST))
D_BEST, B_BEST = solution.T

# Gráfico de resultados
plt.figure(figsize=(10, 6))
plt.plot(times_fine, D_BEST, color='darkseagreen', label='Células diferenciadas')
plt.plot(times_fine, B_BEST, color='cornflowerblue', label='Células basales')
plt.scatter(time_data, D_data, label='Datos de Hoffman (2015)', color='slategrey')
# Líneas horizontales que indican los valores de equilibrio (opcional, para referencia)
plt.axhline(B_initial_BEST, color='slategrey', linestyle='--', alpha=0.5)
plt.axhline(D_initial_BEST, color='slategrey', linestyle='--', alpha=0.5)
plt.title('Modelo optimizado')
plt.xlabel('Tiempo (días)')
plt.ylabel('Concentración celular')
plt.legend()
plt.grid(True)
nombre_archivo = 'Optimizacion.png'
ruta_completa = os.path.join(ruta, nombre_archivo)
plt.savefig(ruta_completa, dpi=600)
plt.close() 

# %% COMPARACIÓN
fig1 = plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(time_data,sim,label='Simulación con parámetros al tanteo', color = 'darkseagreen')
plt.scatter(x = time_data, y= D_data, color='slategrey', label = 'Datos objetivo')
plt.title('Ajuste manual')
plt.xlabel('Tiempo (días)')
plt.ylabel('Células diferenciadas')
plt.grid()
plt.legend()

plt.subplot(122)
plt.plot(times_fine, D_BEST, color='darkseagreen', label='Células diferenciadas')
plt.plot(times_fine, B_BEST, color='cornflowerblue', label='Células basales')
plt.scatter(time_data, D_data, label='Datos de Hoffman (2015)', color='slategrey')
# Líneas horizontales que indican los valores de equilibrio (opcional, para referencia)
plt.axhline(B_initial_BEST, color='slategrey', linestyle='--', alpha=0.5)
plt.axhline(D_initial_BEST, color='slategrey', linestyle='--', alpha=0.5)
plt.title('Optuna')
plt.xlabel('Tiempo (días)')
plt.ylabel('Concentración celular')

fig1.tight_layout()
nombre_archivo = 'Comparacion.png'
ruta_completa = os.path.join(ruta, nombre_archivo)
plt.savefig(ruta_completa, dpi=600)
plt.close() 
# %%
