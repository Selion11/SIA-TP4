import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. Carga y Preparación de Datos
# ==========================================
try:
    df = pd.read_csv('../data/europe.csv')
except FileNotFoundError:
    print("Error: Por favor asegura que 'europe.csv' esté en la ruta correcta.")
    exit()

countries = df['Country'].values
data = df.drop('Country', axis=1).values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
num_paises = len(data_scaled)

# ==========================================
# 2. Definición de Hiperparámetros (Escenarios)
# ==========================================
# Aquí definimos explícitamente los valores para cada corrida comparativa
escenarios = [
    {"nombre": "Base",            "rows": 6, "cols": 6, "sigma": 1.0, "lr": 0.5, "iter": 2000},
    {"nombre": "Grilla_Chica",    "rows": 4, "cols": 4, "sigma": 1.0, "lr": 0.5, "iter": 2000},
    {"nombre": "Grilla_Grande",   "rows": 8, "cols": 8, "sigma": 1.2, "lr": 0.5, "iter": 2000},
    {"nombre": "Bajo_Aprendizaje","rows": 6, "cols": 6, "sigma": 0.5, "lr": 0.1, "iter": 2000},
    {"nombre": "Muchas_Epocas",   "rows": 6, "cols": 6, "sigma": 1.0, "lr": 0.5, "iter": 8000},
    {"nombre": "Pocas_Epocas",    "rows": 6, "cols": 6, "sigma": 1.0, "lr": 0.5, "iter": 200}
]

resultados = []
modelos_entrenados = {} # Para guardar la información y regenerar el óptimo al final

print("Iniciando entrenamiento y generación de gráficos comparativos...\n")

# ==========================================
# 3. Entrenamiento y Generación de Gráficos por Escenario
# ==========================================
for esc in escenarios:
    r, c = esc["rows"], esc["cols"]
    total_neuronas = r * c
    epocas = round(esc["iter"] / num_paises, 1)
    
    # Inicializar y entrenar
    som = MiniSom(x=r, y=c, input_len=data_scaled.shape[1], 
                  sigma=esc["sigma"], learning_rate=esc["lr"], random_seed=42)
    som.random_weights_init(data_scaled)
    som.train_random(data_scaled, num_iteration=esc["iter"])
    
    # Calcular métricas
    hit_map = som.activation_response(data_scaled)
    neuronas_activas = np.count_nonzero(hit_map)
    neuronas_muertas = total_neuronas - neuronas_activas
    qe = som.quantization_error(data_scaled)
    te = som.topographic_error(data_scaled)
    
    # Guardar métricas en la tabla de resultados
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
    
    # Guardar modelo en memoria
    modelos_entrenados[esc["nombre"]] = (som, r, c)

    # --- GRAFICAR Y GUARDAR ESTE ESCENARIO PARA LA PPT ---
    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle(f"Escenario: {esc['nombre']} | Grilla: {r}x{c} | Sigma: {esc['sigma']} | LR: {esc['lr']} | Iteraciones: {esc['iter']}", fontsize=14, fontweight='bold')

    # 1. Mapa de Países
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("1. Mapa de Países")
    ax1.pcolor(np.zeros((r, c)), cmap='Greys', edgecolors='k', alpha=0) 
    for i, x in enumerate(data_scaled):
        w = som.winner(x)
        ax1.text(w[0] + 0.5 + np.random.uniform(-0.25, 0.25), w[1] + 0.5 + np.random.uniform(-0.25, 0.25), 
                 countries[i], ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, lw=0))
    ax1.set_xlim([0, r]); ax1.set_ylim([0, c])
    ax1.set_xticks(np.arange(r)); ax1.set_yticks(np.arange(c))
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Matriz U
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("2. Matriz U (Fronteras y Clústeres)")
    cax = ax2.pcolor(som.distance_map().T, cmap='viridis', edgecolors='k') 
    fig.colorbar(cax, ax=ax2)

    # 3. Aciertos por Neurona
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("3. Países asociados por Neurona")
    cax3 = ax3.pcolor(hit_map.T, cmap='Blues', edgecolors='k')
    fig.colorbar(cax3, ax=ax3)
    for i in range(r):
        for j in range(c):
            if hit_map[i, j] > 0:
                ax3.text(i + 0.5, j + 0.5, str(int(hit_map[i, j])), ha='center', va='center', color='red', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout para el suptitle
    
    # Guardar imagen para la comparativa
    nombre_archivo = f"Comparativa_{esc['nombre']}.png"
    plt.savefig(nombre_archivo, dpi=300)
    plt.close(fig) # Cerrar la figura para no saturar la memoria
    print(f" -> Guardado gráfico comparativo: {nombre_archivo}")

# ==========================================
# 4. Cuadro Comparativo de Resultados
# ==========================================
df_comparativo = pd.DataFrame(resultados)
print("\n" + "="*85)
print("                       CUADRO COMPARATIVO DE HIPERPARÁMETROS")
print("="*85)
print(df_comparativo.to_string(index=False))
print("="*85 + "\n")
df_comparativo.to_csv("cuadro_comparativo_kohonen.csv", index=False)

# ==========================================
# 5. Generar el Gráfico Final (El Óptimo)
# ==========================================
# Definimos el modelo óptimo como aquel que minimiza el Error de Cuantización (QE)
# (Puedes cambiar esta métrica si prefieres basarte en el menor Error Topológico)
mejor_escenario_nombre = df_comparativo.sort_values(by="Error Cuantización (QE)").iloc[0]["Escenario"]
print(f"*** El modelo óptimo seleccionado automáticamente es: '{mejor_escenario_nombre}' ***")

# Recuperamos el modelo óptimo y sus dimensiones
som_opt, r_opt, c_opt = modelos_entrenados[mejor_escenario_nombre]
hit_map_opt = som_opt.activation_response(data_scaled)

fig_opt = plt.figure(figsize=(18, 5))

# 1. Mapa de Países Óptimo
ax1_opt = fig_opt.add_subplot(1, 3, 1)
ax1_opt.set_title("1. Mapa de Países (Modelo Óptimo)")
ax1_opt.pcolor(np.zeros((r_opt, c_opt)), cmap='Greys', edgecolors='k', alpha=0) 
for i, x in enumerate(data_scaled):
    w = som_opt.winner(x)
    ax1_opt.text(w[0] + 0.5 + np.random.uniform(-0.25, 0.25), w[1] + 0.5 + np.random.uniform(-0.25, 0.25), 
             countries[i], ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, lw=0))
ax1_opt.set_xlim([0, r_opt]); ax1_opt.set_ylim([0, c_opt])
ax1_opt.set_xticks(np.arange(r_opt)); ax1_opt.set_yticks(np.arange(c_opt))
ax1_opt.grid(True, linestyle=':', alpha=0.6)

# 2. Matriz U Óptima
ax2_opt = fig_opt.add_subplot(1, 3, 2)
ax2_opt.set_title("2. Matriz U (Modelo Óptimo)")
cax_opt = ax2_opt.pcolor(som_opt.distance_map().T, cmap='viridis', edgecolors='k') 
fig_opt.colorbar(cax_opt, ax=ax2_opt)

# 3. Hits Óptimos
ax3_opt = fig_opt.add_subplot(1, 3, 3)
ax3_opt.set_title("3. Elementos por Neurona (Modelo Óptimo)")
cax3_opt = ax3_opt.pcolor(hit_map_opt.T, cmap='Blues', edgecolors='k')
fig_opt.colorbar(cax3_opt, ax=ax3_opt)
for i in range(r_opt):
    for j in range(c_opt):
        if hit_map_opt[i, j] > 0:
            ax3_opt.text(i + 0.5, j + 0.5, str(int(hit_map_opt[i, j])), ha='center', va='center', color='red', fontweight='bold')

plt.tight_layout()

# Guardamos específicamente con el nombre que solicitaste
plt.savefig("Kohonen_Network_Results.png", dpi=300)
print(" -> Gráfico óptimo guardado como: 'Kohonen_Network_Results.png'")
# plt.show() # Descomentar si deseas que se abra la ventana del gráfico al terminar