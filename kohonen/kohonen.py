import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Carga y Preparación de Datos
# ==========================================
try:
    df = pd.read_csv('../data/europe.csv') # Ajusta la ruta si es necesario
except FileNotFoundError:
    print("Error: Por favor asegura que 'europe.csv' esté en la ruta correcta.")
    exit()

countries = df['Country'].values
data = df.drop('Country', axis=1).values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
num_paises = len(data_scaled)

# ==========================================
# 2. Configuración Fija y Grid Search de LR
# ==========================================
# Fijamos las dimensiones óptimas que elegiste para clustering real
ROWS, COLS = 4, 4 
SIGMA_FIJO = 1.0
ITERACIONES_FIJAS = 2000

# Generamos las tasas de aprendizaje desde 1.0 hasta 0.1 bajando de a 0.1
learning_rates = np.arange(1.0, 0.0, -0.1)

resultados = []
modelos_entrenados = {}

print(f"Iniciando Grid Search sobre {len(learning_rates)} tasas de aprendizaje distintas (Grilla {ROWS}x{ROWS})...\n")

# ==========================================
# 3. Bucle de Evaluación Automatizada
# ==========================================
for lr in learning_rates:
    lr_round = round(lr, 1)
    nombre_escenario = f"LR_{lr_round}"
    
    # Inicializar y entrenar
    som = MiniSom(x=ROWS, y=COLS, input_len=data_scaled.shape[1], 
                  sigma=SIGMA_FIJO, learning_rate=lr_round, random_seed=42)
    som.random_weights_init(data_scaled)
    som.train_random(data_scaled, num_iteration=ITERACIONES_FIJAS)
    
    # Calcular métricas
    hit_map = som.activation_response(data_scaled)
    neuronas_activas = np.count_nonzero(hit_map)
    neuronas_muertas = (ROWS * COLS) - neuronas_activas
    qe = som.quantization_error(data_scaled)
    te = som.topographic_error(data_scaled)
    
    resultados.append({
        "Escenario": nombre_escenario,
        "Dimensiones": f"{ROWS}x{COLS}",
        "Radio (Sigma)": SIGMA_FIJO,
        "Learning Rate": lr_round,
        "Iteraciones": ITERACIONES_FIJAS,
        "Neuronas Muertas": neuronas_muertas,
        "Error Cuantización (QE)": round(qe, 4),
        "Error Topológico (TE)": round(te, 4)
    })
    
    modelos_entrenados[nombre_escenario] = (som, hit_map)

# ==========================================
# 4. Cuadro Comparativo en Consola
# ==========================================
df_comparativo = pd.DataFrame(resultados)
print("="*85)
print("                    RESULTADOS DE LA EVALUACIÓN DE LEARNING RATES")
print("="*85)
print(df_comparativo.to_string(index=False))
print("="*85 + "\n")

# ==========================================
# 5. Selección Automatizada del LR Óptimo
# ==========================================
# Tu criterio: Primero menor cantidad de neuronas muertas. Desempate por menor QE.
ganador_df = df_comparativo.sort_values(by=["Neuronas Muertas", "Error Cuantización (QE)"]).iloc[0]
mejor_escenario_nombre = ganador_df["Escenario"]
lr_optimo = ganador_df["Learning Rate"]
epocas_optimas = round(ITERACIONES_FIJAS / num_paises, 1)

print("="*50)
print("         HIPERPARAMETROS OPTIMOS DETECTADOS")
print("="*50)
print(f" Escenario Ganador:          {mejor_escenario_nombre}")
print(f" Tamaño de la Grilla:        {ganador_df['Dimensiones']}")
print(f" Radio Inicial (Sigma):      {ganador_df['Radio (Sigma)']}")
print(f" Tasa de Aprendizaje ÓPTIMA: {lr_optimo}  <-- ¡ESTE ES EL GANADOR!")
print(f" Iteraciones Totales:        {ganador_df['Iteraciones']}")
print(f" Épocas Equivalentes:        {epocas_optimas}")
print("-"*50)
print(" METRICAS DE LA CORRIDA GANADORA:")
print(f" Neuronas Muertas:           {ganador_df['Neuronas Muertas']}")
print(f" Error de Cuantización (QE): {ganador_df['Error Cuantización (QE)']}")
print(f" Error Topológico (TE):      {ganador_df['Error Topológico (TE)']}")
print("="*50 + "\n")

# ==========================================
# 6. Gráfico Final del Modelo Óptimo Elegido
# ==========================================
som_opt, hit_map_opt = modelos_entrenados[mejor_escenario_nombre]
fig_opt = plt.figure(figsize=(18, 5))

# 1. Mapa de Países Óptimo
ax1_opt = fig_opt.add_subplot(1, 3, 1)
ax1_opt.set_title(f"1. Mapa de Países (Óptimo con LR = {lr_optimo})")
ax1_opt.pcolor(np.zeros((ROWS, COLS)), cmap='Greys', edgecolors='k', alpha=0) 
for i, x in enumerate(data_scaled):
    w = som_opt.winner(x)
    ax1_opt.text(w[0] + 0.5 + np.random.uniform(-0.2, 0.2), w[1] + 0.5 + np.random.uniform(-0.2, 0.2), 
             countries[i], ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, lw=0))
ax1_opt.set_xlim([0, ROWS]); ax1_opt.set_ylim([0, COLS])
ax1_opt.grid(True, linestyle=':', alpha=0.6)

# 2. Matriz U Óptima
ax2_opt = fig_opt.add_subplot(1, 3, 2)
ax2_opt.set_title(f"2. Matriz U (Modelo Óptimo LR = {lr_optimo})")
cax_opt = ax2_opt.pcolor(som_opt.distance_map().T, cmap='viridis', edgecolors='k') 
fig_opt.colorbar(cax_opt, ax=ax2_opt)

# 3. Hits Óptimos
ax3_opt = fig_opt.add_subplot(1, 3, 3)
ax3_opt.set_title("3. Elementos por Neurona (Modelo Óptimo)")
cax3_opt = ax3_opt.pcolor(hit_map_opt.T, cmap='Blues', edgecolors='k')
fig_opt.colorbar(cax3_opt, ax=ax3_opt)
for i in range(ROWS):
    for j in range(COLS):
        if hit_map_opt[i, j] > 0:
            ax3_opt.text(i + 0.5, j + 0.5, str(int(hit_map_opt[i, j])), ha='center', va='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig("Kohonen_Network_Results.png", dpi=300)
print(f" -> Gráfico definitivo guardado como: 'Kohonen_Network_Results.png' usando LR={lr_optimo}")