import json
import csv
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def save_evolution_data(history, trained_patterns, json_path, csv_path):
    """
    Guarda los datos de la evolución en formatos JSON y CSV.
    Registra el paso, época, energía, neurona modificada y la distancia Hamming a todos los patrones.
    """
    # Asegurar que existan los directorios
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    formatted_steps = []
    
    for i, step in enumerate(history):
        state = step['state']
        energy = float(step['energy'])
        neuron = int(step['neuron_updated'])
        epoch = int(step['epoch'])
        desc = step['description']
        
        # Calcular distancia Hamming para cada patrón entrenado
        hammings = {}
        for name, pattern in trained_patterns.items():
            hammings[name] = int(np.sum(state != pattern))
            hammings[f"{name}_inv"] = int(np.sum(-state != pattern))
            
        step_data = {
            'step': i,
            'epoch': epoch,
            'energy': energy,
            'neuron_updated': neuron,
            'description': desc,
            'hamming_distances': hammings,
            'state': state.tolist()
        }
        formatted_steps.append(step_data)
        
    # 1. Guardar en JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_steps, f, indent=2, ensure_ascii=False)
        
    # 2. Guardar en CSV
    if formatted_steps:
        # Definir cabeceras
        headers = ['step', 'epoch', 'energy', 'neuron_updated', 'description']
        # Agregar cabeceras de distancias Hamming
        pattern_names = list(trained_patterns.keys())
        for name in pattern_names:
            headers.append(f"hamming_{name}")
            headers.append(f"hamming_{name}_inv")
            
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for step in formatted_steps:
                row = [
                    step['step'],
                    step['epoch'],
                    step['energy'],
                    step['neuron_updated'],
                    step['description']
                ]
                for name in pattern_names:
                    row.append(step['hamming_distances'][name])
                    row.append(step['hamming_distances'][f"{name}_inv"])
                writer.writerow(row)

def create_evolution_gif(history, shape, gif_path, scale=40, duration=200):
    """
    Genera un GIF animado pixel-art de alta definición a partir del historial de estados.
    Cada estado se dibuja con píxeles escalados y colores vibrantes.
    """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    h, w = shape
    frames = []
    
    # Paleta de colores
    COLOR_ACTIVE = (0, 229, 255)    # Cyan brillante
    COLOR_INACTIVE = (33, 33, 33)   # Gris carbón oscuro
    COLOR_HIGHLIGHT = (255, 235, 59) # Amarillo para la neurona recién actualizada
    
    for i, step in enumerate(history):
        state = step['state'].reshape(shape)
        u_neuron = step['neuron_updated']
        
        # Crear imagen en blanco
        img = Image.new("RGB", (w * scale, h * scale), color=COLOR_INACTIVE)
        draw = ImageDraw.Draw(img)
        
        for r in range(h):
            for c in range(w):
                idx = r * w + c
                val = state[r, c]
                
                # Coordenadas del bloque pixelado
                x0, y0 = c * scale, r * scale
                x1, y1 = x0 + scale, y0 + scale
                
                # Definir color de relleno
                if idx == u_neuron and i > 0 and step['changed']:
                    # Resaltar la neurona que acaba de cambiar de estado en amarillo
                    fill_color = COLOR_HIGHLIGHT
                elif val == 1:
                    fill_color = COLOR_ACTIVE
                else:
                    fill_color = COLOR_INACTIVE
                    
                # Dibujar bloque de pixel
                draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=(20, 20, 20), width=1)
                
        # Agregar cuadro
        frames.append(img)
        
    # Duplicar el cuadro final para mantener el estado final en pantalla un momento
    if frames:
        for _ in range(5):
            frames.append(frames[-1])
            
    # Guardar GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

def plot_evolution_graphs(history, trained_patterns, target_name, png_path):
    """
    Genera un gráfico de alta calidad con dos ejes y lo guarda en formato PNG.
    - Eje Izquierdo: Decaimiento de la energía de Lyapunov.
    - Eje Derecho: Distancia Hamming al patrón objetivo.
    """
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    steps = list(range(len(history)))
    energies = [step['energy'] for step in history]
    
    # Calcular Hamming al patrón objetivo en cada paso
    target_pattern = trained_patterns.get(target_name.replace("-", "")) # Limpiar si tiene '-' de invertido
    if target_pattern is None:
        # Si no hay objetivo, tomamos el más cercano del primer paso
        first_state = history[0]['state']
        closest_name = None
        min_dist = len(first_state) + 1
        for name, pat in trained_patterns.items():
            d = np.sum(first_state != pat)
            if d < min_dist:
                min_dist = d
                closest_name = name
        target_name = closest_name
        target_pattern = trained_patterns[target_name]
        
    hammings = []
    for step in history:
        state = step['state']
        # Evaluar distancia Hamming normal o invertida (la menor)
        d_normal = np.sum(state != target_pattern)
        d_inv = np.sum(-state != target_pattern)
        hammings.append(min(d_normal, d_inv))
        
    # Configurar estilo moderno y limpio
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Gráfico de Energía (Eje Izquierdo)
    color1 = '#0070c0'
    ax1.set_xlabel('Pasos de actualización', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Energía de Lyapunov (E)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(steps, energies, color=color1, linewidth=2.5, marker='o', markersize=5, label='Energía (E)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico de Distancia Hamming (Eje Derecho)
    ax2 = ax1.twinx()
    color2 = '#c00000'
    ax2.set_ylabel(f'Distancia Hamming a la letra {target_name}', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(steps, hammings, color=color2, linewidth=2, linestyle='--', marker='s', markersize=5, label=f'Distancia a {target_name}')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False) # Evitar grillas superpuestas molestas
    
    # Leyendas combinadas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc')
    
    plt.title(f'Trayectoria de Convergencia: Recuperación de la Letra {target_name}\n(Energía inicial vs. Evolución asincrónica)', 
              fontsize=14, fontweight='bold', pad=15)
    
    fig.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
