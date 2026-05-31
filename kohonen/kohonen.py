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
# 2. Definición de Escenarios de Hiperparámetros
# ==========================================
escenarios = [
    {"nombre": "Base",            "rows": 6, "cols": 6, "sigma": 1.0, "lr": 0.5, "iter": 2000},
    {"nombre": "Grilla_Chica",    "rows": 4, "cols": 4, "sigma": 1.0, "lr": 0.5, "iter": 2000},
    {"nombre": "Grilla_Grande",   "rows": 8, "cols": 8, "sigma": 1.2, "lr": 0.5, "iter": 2000},
    {"nombre": "Bajo_Aprendizaje","rows": 6, "cols": 6, "sigma": 0.5, "lr": 0.1, "iter": 2000},
    {"nombre": "Muchas_Epocas",   "rows": 6, "cols": 6, "sigma": 1.0, "lr": 0.5, "iter": 8000},
    {"nombre": "Pocas_Epocas",    "rows": 6, "cols": 6, "sigma": 1.0, "lr": 0.5, "iter": 200}
]

resultados = []
modelos_entrenados = {}

print("Iniciando entrenamiento y generación de gráficos comparativos...\n")

# ==========================================
# 3. Bucle de Entrenamiento y Guardado de Imágenes
# ==========================================
for esc in escenarios:
    r, c = esc["rows"], esc["cols"]
    total_neuronas = r * c
    
    som = MiniSom(x=r, y=c, input_len=data_scaled.shape[1], 
                  sigma=esc["sigma"], learning_rate=esc["lr"], random_seed=42)
    som.random_weights_init(data_scaled)
    som.train_random(data_scaled, num_iteration=esc["iter"])
    
    hit_map = som.activation_response(data_scaled)
    neuronas_activas = np.count_nonzero(hit_map)
    neuronas_muertas = total_neuronas - neuronas_activas
    qe = som.quantization_error(data_scaled)
    te = som.topographic_error(data_scaled)
    
    resultados.append({
        "Escenario": esc["nombre"],
        "Dimensiones": f"{r}x{c}",
        "Radio (Sigma)": esc["sigma"],
        "Learning Rate": esc["lr"],
        "Iteraciones": esc["iter"],
        "Neuronas Muertas": neuronas_muertas,
        "Error Cuantización (QE)": round(qe, 4),
        "Error Topológico (TE)": round(te, 4)
    })
    
    modelos_entrenados[esc["nombre"]] = (som, r, c, hit_map)

    # Gráfico para la PPT
    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle(f"Escenario: {esc['nombre']} | Grilla: {r}x{c} | Sigma: {esc['sigma']} | LR: {esc['lr']} | Iter: {esc['iter']}", fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("1. Mapa de Países")
    ax1.pcolor(np.zeros((r, c)), cmap='Greys', edgecolors='k', alpha=0) 
    for i, x in enumerate(data_scaled):
        w = som.winner(x)
        ax1.text(w[0] + 0.5 + np.random.uniform(-0.2, 0.2), w[1] + 0.5 + np.random.uniform(-0.2, 0.2), 
                 countries[i], ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, lw=0))
    ax1.set_xlim([0, r]); ax1.set_ylim([0, c])
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("2. Matriz U (Distancias)")
    cax = ax2.pcolor(som.distance_map().T, cmap='viridis', edgecolors='k') 
    fig.colorbar(cax, ax=ax2)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("3. Elementos por Neurona")
    cax3 = ax3.pcolor(hit_map.T, cmap='Blues', edgecolors='k')
    fig.colorbar(cax3, ax=ax3)
    for i in range(r):
        for j in range(c):
            if hit_map[i, j] > 0:
                ax3.text(i + 0.5, j + 0.5, str(int(hit_map[i, j])), ha='center', va='center', color='red', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"Comparativa_{esc['nombre']}.png", dpi=300)
    plt.close(fig)
    print(f" -> Guardado: Comparativa_{esc['nombre']}.png")

# ==========================================
# 4. Cuadro Comparativo en Consola
# ==========================================
df_comparativo = pd.DataFrame(resultados)
print("\n" + "="*85)
print("                       CUADRO COMPARATIVO DE HIPERPARÁMETROS")
print("="*85)
print(df_comparativo.to_string(index=False))
print("="*85 + "\n")

# ==========================================
# 5. Selección e Impresión del Óptimo Matemático
# ==========================================
ganador_df = df_comparativo.sort_values(by="Neuronas Muertas").iloc[0]
mejor_escenario_nombre = ganador_df["Escenario"]

print("="*50)
print("         HIPERPARAMETROS OPTIMOS DETECTADOS")
print("="*50)
print(f" Escenario Ganador:      {mejor_escenario_nombre}")
print(f" Tamaño de la Grilla:    {ganador_df['Dimensiones']}")
print(f" Error de Cuantización:  {ganador_df['Error Cuantización (QE)']}")
print(f" Error Topológico:       {ganador_df['Error Topológico (TE)']}")
print(f" Neuronas Muertas:       {ganador_df['Neuronas Muertas']}")
print("="*50 + "\n")

# Re-generar el gráfico ganador con el nombre final solicitado
som_opt, r_opt, c_opt, hit_map_opt = modelos_entrenados[mejor_escenario_nombre]
fig_opt = plt.figure(figsize=(18, 5))

ax1_opt = fig_opt.add_subplot(1, 3, 1)
ax1_opt.set_title(f"1. Mapa de Países (Óptimo: {mejor_escenario_nombre})")
ax1_opt.pcolor(np.zeros((r_opt, c_opt)), cmap='Greys', edgecolors='k', alpha=0) 
for i, x in enumerate(data_scaled):
    w = som_opt.winner(x)
    ax1_opt.text(w[0] + 0.5 + np.random.uniform(-0.2, 0.2), w[1] + 0.5 + np.random.uniform(-0.2, 0.2), 
             countries[i], ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, lw=0))
ax1_opt.set_xlim([0, r_opt]); ax1_opt.set_ylim([0, c_opt])
ax1_opt.grid(True, linestyle=':', alpha=0.6)

ax2_opt = fig_opt.add_subplot(1, 3, 2)
ax2_opt.set_title("2. Matriz U (Modelo Óptimo)")
cax_opt = ax2_opt.pcolor(som_opt.distance_map().T, cmap='viridis', edgecolors='k') 
fig_opt.colorbar(cax_opt, ax=ax2_opt)

ax3_opt = fig_opt.add_subplot(1, 3, 3)
ax3_opt.set_title("3. Elementos por Neurona (Modelo Óptimo)")
cax3_opt = ax3_opt.pcolor(hit_map_opt.T, cmap='Blues', edgecolors='k')
fig_opt.colorbar(cax3_opt, ax=ax3_opt)
for i in range(r_opt):
    for j in range(c_opt):
        if hit_map_opt[i, j] > 0:
            ax3_opt.text(i + 0.5, j + 0.5, str(int(hit_map_opt[i, j])), ha='center', va='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig("Kohonen_Network_Results.png", dpi=300)
print(" -> Gráfico final guardado como: 'Kohonen_Network_Results.png'")