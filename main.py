import os
import sys
import time
import numpy as np

# Asegurar que las rutas locales se importen correctamente
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pattern_loader import load_patterns, pattern_to_str
from src.hopfield import HopfieldNetwork

# Constantes de colores para consola ANSI
CLEAR_SCREEN = "\033[H\033[J"
BG_BLUE = "\033[44m"
TEXT_WHITE = "\033[97m"
TEXT_CYAN = "\033[96m"
TEXT_GREEN = "\033[92m"
TEXT_YELLOW = "\033[93m"
TEXT_RED = "\033[91m"
TEXT_MAGENTA = "\033[95m"
TEXT_DIM = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header():
    print(f"{TEXT_CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{TEXT_CYAN}{BOLD}║                RED DE HOPFIELD - PATRONES DE LETRAS          ║{RESET}")
    print(f"{TEXT_CYAN}{BOLD}║         SIA - TRABAJO PRÁCTICO 4 - ARQUITECTURA GENÉRICA     ║{RESET}")
    print(f"{TEXT_CYAN}{BOLD}╚══════════════════════════════════════════════════════════════╝{RESET}")

def render_grid_interactive(state, shape, highlight_idx=-1):
    """
    Renderiza el estado actual en una grilla de alta fidelidad,
    resaltando la neurona recién actualizada en amarillo/rojo si corresponde.
    """
    h, w = shape
    matrix = state.reshape(shape)
    lines = []
    for r in range(h):
        line_chars = []
        for c in range(w):
            idx = r * w + c
            val = matrix[r, c]
            
            if val == 1:
                # Bloque activo (Cyan brillante)
                char = "██"
                color = TEXT_CYAN
            else:
                # Bloque inactivo (Gris oscuro)
                char = "░░"
                color = TEXT_DIM
                
            if idx == highlight_idx:
                # Neurona que se acaba de actualizar
                char = "▓▓"
                color = TEXT_YELLOW + BOLD
                
            line_chars.append(f"{color}{char}{RESET}")
        lines.append("  " + "".join(line_chars))
    return "\n".join(lines)

def display_patterns_side_by_side(patterns, shape):
    """
    Muestra los patrones originales cargados en columnas alineadas.
    """
    names = list(patterns.keys())
    grids = [patterns[name].reshape(shape) for name in names]
    h, w = shape
    
    # Imprimir nombres centrados
    header_line = ""
    for name in names:
        header_line += f"    Patrón: {name: <6}  "
    print(f"\n{BOLD}{TEXT_GREEN}{header_line}{RESET}\n")
    
    for r in range(h):
        row_str = ""
        for grid in grids:
            grid_row = grid[r]
            row_chars = []
            for val in grid_row:
                if val == 1:
                    row_chars.append(f"{TEXT_CYAN}██{RESET}")
                else:
                    row_chars.append(f"{TEXT_DIM}░░{RESET}")
            row_str += "    " + "".join(row_chars) + "    "
        print(row_str)
    print()

def animate_evolution(network, history, shape, step_delay=0.1, mode="auto"):
    """
    Anima la evolución paso a paso en la terminal.
    
    mode: "auto" para animación fluida automática, "manual" para avanzar con Enter.
    """
    total_steps = len(history)
    
    for i, step in enumerate(history):
        os.system('cls' if os.name == 'nt' else 'clear')
        print_header()
        
        state = step['state']
        energy = step['energy']
        u_neuron = step['neuron_updated']
        changed = step['changed']
        epoch = step['epoch']
        desc = step['description']
        
        closest_name, distance = network.get_closest_pattern(state)
        
        print(f"\n{BOLD}{TEXT_MAGENTA}=== EVOLUCIÓN ASINCRÓNICA PASO A PASO ==={RESET}")
        print(f"Paso {i}/{total_steps-1} | Época: {epoch} | {TEXT_YELLOW}{desc}{RESET}")
        print(f"Energía de Lyapunov (E): {TEXT_GREEN}{energy:.4f}{RESET}")
        print(f"Patrón más cercano: {TEXT_CYAN}{closest_name}{RESET} (Distancia Hamming: {TEXT_RED}{distance}{RESET})")
        print("-" * 62)
        
        # Renderizar grilla
        print(render_grid_interactive(state, shape, highlight_idx=u_neuron))
        print("-" * 62)
        
        if i == 0:
            print(f"{TEXT_YELLOW}Estado Inicial Ruidoso.{RESET}")
        elif i == total_steps - 1:
            print(f"{TEXT_GREEN}{BOLD}¡Estado Final Estable Alcanzado!{RESET}")
        else:
            if changed:
                row = u_neuron // shape[1]
                col = u_neuron % shape[1]
                print(f"Actualización: {TEXT_GREEN}Neurona {u_neuron} (Fila {row}, Col {col}) CAMBIÓ a {state[u_neuron]}{RESET}")
            else:
                print("Evaluación: Sin cambios de estado en este paso.")
                
        if i < total_steps - 1:
            if mode == "manual":
                input(f"\nPresione {BOLD}[Enter]{RESET} para el siguiente paso...")
            else:
                time.sleep(step_delay)
        else:
            print(f"\n{TEXT_GREEN}{BOLD}Evolución completada.{RESET}")

def menu_ruido(network, patterns, shape):
    """
    Submenú para seleccionar letra y cantidad de ruido a aplicar.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print_header()
    print(f"\n{BOLD}{TEXT_GREEN}--- SIMULADOR DE ASOCIACIÓN CON RUIDO ---{RESET}\n")
    
    names = list(patterns.keys())
    for i, name in enumerate(names):
        print(f" {i+1}. Letra {name}")
    
    try:
        sel = int(input(f"\nSeleccione una letra (1-{len(names)}): "))
        if sel < 1 or sel > len(names):
            print(f"{TEXT_RED}Selección inválida.{RESET}")
            time.sleep(1.5)
            return
        letter_name = names[sel-1]
    except ValueError:
        print(f"{TEXT_RED}Debe ingresar un número.{RESET}")
        time.sleep(1.5)
        return
        
    original_pattern = patterns[letter_name]
    
    # Selección de ruido
    print(f"\n¿Cómo desea ingresar el ruido?")
    print(" 1. Porcentaje de bits invertidos aleatoriamente (ej. 20%)")
    print(" 2. Cantidad exacta de bits invertidos (ej. 5 bits de 25)")
    
    try:
        opc_ruido = int(input("\nSeleccione opción (1-2): "))
    except ValueError:
        opc_ruido = 1
        
    if opc_ruido == 2:
        try:
            exact_bits = int(input(f"Ingrese la cantidad de bits a invertir (0 a {network.N}): "))
            noisy_state = network.add_exact_noise(original_pattern, exact_bits)
            noise_desc = f"{exact_bits} bits invertidos"
        except ValueError:
            print(f"{TEXT_RED}Entrada inválida. Usando 5 bits por defecto.{RESET}")
            noisy_state = network.add_exact_noise(original_pattern, 5)
            noise_desc = "5 bits invertidos"
    else:
        try:
            pct = float(input("Ingrese el porcentaje de ruido (0 a 100): ")) / 100.0
            if pct < 0 or pct > 1:
                pct = 0.2
            noisy_state = network.add_noise(original_pattern, pct)
            noise_desc = f"{pct*100:.1f}% de ruido"
        except ValueError:
            print(f"{TEXT_RED}Entrada inválida. Usando 20% por defecto.{RESET}")
            noisy_state = network.add_noise(original_pattern, 0.2)
            noise_desc = "20% de ruido"
            
    # Selección de modo de evolución
    print(f"\n¿Cómo desea visualizar la evolución?")
    print(" 1. Animación automática continua (Hacker style!)")
    print(" 2. Control manual paso a paso (presionando Enter)")
    
    try:
        modo_evol = int(input("\nSeleccione opción (1-2): "))
    except ValueError:
        modo_evol = 1
        
    anim_mode = "auto" if modo_evol == 1 else "manual"
    
    # Ejecutar evolución
    history, converged = network.evolve(noisy_state, random_order=True)
    animate_evolution(network, history, shape, step_delay=0.15, mode=anim_mode)
    
    # Preguntar si desea exportar
    print(f"\n{BOLD}{TEXT_CYAN}¿Desea exportar y guardar los resultados de esta simulación?{RESET}")
    print(f" (Se guardará: GIF animado, Gráfico de energía PNG y archivos de datos CSV y JSON)")
    opc = input(" [s/N] > ").strip().lower()
    
    if opc in ['s', 'si', 'y', 'yes']:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        gif_file = os.path.join(export_dir, f"recuperacion_{letter_name}_{timestamp}.gif")
        png_file = os.path.join(export_dir, f"grafico_{letter_name}_{timestamp}.png")
        json_file = os.path.join(export_dir, f"datos_{letter_name}_{timestamp}.json")
        csv_file = os.path.join(export_dir, f"datos_{letter_name}_{timestamp}.csv")
        
        print(f"\n{TEXT_YELLOW}Generando y guardando archivos de exportación en 'exports/'...{RESET}")
        
        from src.visualization import save_evolution_data, create_evolution_gif, plot_evolution_graphs
        
        try:
            save_evolution_data(history, patterns, json_file, csv_file)
            create_evolution_gif(history, shape, gif_file, scale=40)
            plot_evolution_graphs(history, patterns, letter_name, png_file)
            
            print(f"\n{TEXT_GREEN}{BOLD}¡EXPORTACIÓN EXITOSA!{RESET}")
            print(f"  - GIF animado:  {TEXT_CYAN}exports/{os.path.basename(gif_file)}{RESET}")
            print(f"  - Gráfico PNG:  {TEXT_CYAN}exports/{os.path.basename(png_file)}{RESET}")
            print(f"  - Datos JSON:   {TEXT_CYAN}exports/{os.path.basename(json_file)}{RESET}")
            print(f"  - Datos CSV:    {TEXT_CYAN}exports/{os.path.basename(csv_file)}{RESET}")
        except Exception as e:
            print(f"\n{TEXT_RED}Error al exportar los datos: {e}{RESET}")
            
        input(f"\nPresione {BOLD}[Enter]{RESET} para continuar...")
    else:
        input(f"\nPresione {BOLD}[Enter]{RESET} para volver al menú...")

def menu_spurious(network, patterns, shape):
    """
    Submenú para identificar estados espúreos de forma empírica (parte b).
    Genera un patrón altamente ruidoso o completamente aleatorio y evalúa su convergencia.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print_header()
    print(f"\n{BOLD}{TEXT_RED}--- IDENTIFICAR ESTADOS ESPÚREOS (Parte b) ---{RESET}\n")
    print("Un estado espúreo es un punto fijo (mínimo local de energía) de la red")
    print("que no corresponde a ningún patrón cargado ni a su negativo exacto.")
    print("Suelen alcanzarse al ingresar patrones extremadamente ruidosos o aleatorios.")
    print("-" * 65)
    
    print("\n¿Cómo quiere generar el estado inicial?")
    print(" 1. Estado 100% aleatorio (cada píxel con probabilidad 50% de ser +1 o -1)")
    print(" 2. Patrón original altamente ruidoso (ej. 60% de ruido)")
    
    try:
        opcion = int(input("\nSeleccione opción (1-2): "))
    except ValueError:
        opcion = 1
        
    if opcion == 2:
        # Altamente ruidoso
        names = list(patterns.keys())
        print("\nSeleccione patrón base:")
        for idx, name in enumerate(names):
            print(f"  {idx+1}. {name}")
        try:
            sel = int(input(f"Seleccione (1-{len(names)}): "))
            base_pattern = patterns[names[sel-1]]
        except (ValueError, IndexError):
            base_pattern = patterns[names[0]]
            
        noisy_state = network.add_noise(base_pattern, 0.6)  # 60% de ruido
        state_desc = "Patrón base con 60% de ruido"
    else:
        # 100% aleatorio
        noisy_state = np.random.choice([1, -1], size=network.N)
        state_desc = "Estado 100% aleatorio"
        
    print(f"\n{TEXT_YELLOW}Evolucionando la red desde el {state_desc}...{RESET}")
    
    # Evolucionar de forma asincrónica hasta la convergencia
    history, converged = network.evolve(noisy_state, random_order=True)
    final_state = history[-1]['state']
    final_energy = history[-1]['energy']
    
    # Evaluar si es un estado espúreo
    closest_name, distance = network.get_closest_pattern(final_state)
    
    # Un estado es un patrón original si la distancia es 0
    # (El método get_closest_pattern ya considera patrones invertidos, devolviendo '-Nombre')
    is_spurious = (distance > 0)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print_header()
    print(f"\n{BOLD}{TEXT_RED}=== RESULTADO DE CONVERGENCIA ==={RESET}\n")
    print(f"Estado Inicial: {TEXT_YELLOW}{state_desc}{RESET}")
    print(f"Convergencia: {'SI' if converged else 'NO'}")
    print(f"Pasos de actualización con cambios: {len(history)-2}")
    print(f"Energía Final (E): {TEXT_GREEN}{final_energy:.4f}{RESET}")
    print(f"Patrón más cercano: {TEXT_CYAN}{closest_name}{RESET} (Distancia Hamming: {TEXT_RED}{distance} de {network.N} bits{RESET})")
    print("-" * 65)
    
    # Dibujar estado inicial vs final side-by-side
    h, w = shape
    init_matrix = noisy_state.reshape(shape)
    final_matrix = final_state.reshape(shape)
    
    print(f"\n    {BOLD}Estado Inicial Ruidoso{RESET}          {BOLD}Estado Convergido Final{RESET}\n")
    for r in range(h):
        init_chars = []
        for val in init_matrix[r]:
            init_chars.append(f"{TEXT_CYAN}██{RESET}" if val == 1 else f"{TEXT_DIM}░░{RESET}")
            
        final_chars = []
        for val in final_matrix[r]:
            final_chars.append(f"{TEXT_CYAN}██{RESET}" if val == 1 else f"{TEXT_DIM}░░{RESET}")
            
        print("    " + "".join(init_chars) + "            " + "".join(final_chars))
        
    print("-" * 65)
    
    if is_spurious:
        print(f"{BOLD}{TEXT_RED}¡ESTADO ESPÚREO IDENTIFICADO!{RESET}")
        print("Explicación:")
        print("La red convergió a un estado estable (mínimo local de energía) que")
        print("NO es ninguno de los patrones originales de entrenamiento (ni su opuesto).")
        print(f"Este mínimo espúreo tiene una energía de {final_energy:.4f} y se comporta")
        print("como una 'falsa memoria' creada por la superposición de los patrones reales.")
    else:
        print(f"{BOLD}{TEXT_GREEN}¡CONVERGIÓ A UN PATRÓN REGISTRADO!{RESET}")
        print(f"La red logró recuperar con éxito el patrón: {TEXT_CYAN}{closest_name}{RESET}")
        print("No se detectó un estado espúreo en esta ejecución.")
        print("Tip: Intente con un estado 100% aleatorio o aumente el nivel de ruido.")
        
    # Mostrar historial de energía para ver el decaimiento
    print(f"\n{BOLD}{TEXT_MAGENTA}Evolución de la Energía en cada paso intermedio:{RESET}")
    steps_energy = [step['energy'] for step in history if step['neuron_updated'] != -1 or step['description'] == 'Estado Inicial']
    energy_str = " -> ".join(f"{e:.2f}" for e in steps_energy[:10])
    if len(steps_energy) > 10:
        energy_str += f" ... -> {steps_energy[-1]:.2f}"
    print(f"  {energy_str}")
    
    # Preguntar si desea exportar
    print(f"\n{BOLD}{TEXT_CYAN}¿Desea exportar y guardar los resultados de esta simulación?{RESET}")
    print(f" (Se guardará: GIF animado, Gráfico de energía PNG y archivos de datos CSV y JSON)")
    opc = input(" [s/N] > ").strip().lower()
    
    if opc in ['s', 'si', 'y', 'yes']:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Identificar nombre base
        base_name = f"espureo_{closest_name}" if is_spurious else f"recuperacion_{closest_name}"
        base_name = base_name.replace("-", "inv_")
        
        gif_file = os.path.join(export_dir, f"{base_name}_{timestamp}.gif")
        png_file = os.path.join(export_dir, f"grafico_{base_name}_{timestamp}.png")
        json_file = os.path.join(export_dir, f"datos_{base_name}_{timestamp}.json")
        csv_file = os.path.join(export_dir, f"datos_{base_name}_{timestamp}.csv")
        
        print(f"\n{TEXT_YELLOW}Generando y guardando archivos de exportación en 'exports/'...{RESET}")
        
        from src.visualization import save_evolution_data, create_evolution_gif, plot_evolution_graphs
        
        try:
            save_evolution_data(history, patterns, json_file, csv_file)
            create_evolution_gif(history, shape, gif_file, scale=40)
            plot_evolution_graphs(history, patterns, closest_name, png_file)
            
            print(f"\n{TEXT_GREEN}{BOLD}¡EXPORTACIÓN EXITOSA!{RESET}")
            print(f"  - GIF animado:  {TEXT_CYAN}exports/{os.path.basename(gif_file)}{RESET}")
            print(f"  - Gráfico PNG:  {TEXT_CYAN}exports/{os.path.basename(png_file)}{RESET}")
            print(f"  - Datos JSON:   {TEXT_CYAN}exports/{os.path.basename(json_file)}{RESET}")
            print(f"  - Datos CSV:    {TEXT_CYAN}exports/{os.path.basename(csv_file)}{RESET}")
        except Exception as e:
            print(f"\n{TEXT_RED}Error al exportar los datos: {e}{RESET}")
            
        input(f"\nPresione {BOLD}[Enter]{RESET} para continuar...")
    else:
        input(f"\nPresione {BOLD}[Enter]{RESET} para volver al menú...")

def main():
    # Cargar patrones dinámicamente
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    patterns_file = os.path.join(data_dir, "patterns.txt")
    
    # Si hay múltiples archivos de patrones, permitir elegir
    try:
        txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
        if len(txt_files) > 1:
            print_header()
            print(f"\n{BOLD}Se detectaron múltiples archivos de patrones en 'data/':{RESET}")
            for idx, f_name in enumerate(txt_files):
                try:
                    temp_p, temp_s = load_patterns(os.path.join(data_dir, f_name))
                    info = f"({temp_s[0]}x{temp_s[1]}, {len(temp_p)} patrones: {', '.join(temp_p.keys())})"
                except Exception:
                    info = "(Error al cargar)"
                print(f"  {idx+1}. {f_name} {TEXT_DIM}{info}{RESET}")
            
            try:
                sel = input(f"\nSeleccione el archivo a cargar (1-{len(txt_files)}) [default: 1]: ").strip()
                if sel:
                    sel_idx = int(sel) - 1
                    if 0 <= sel_idx < len(txt_files):
                        patterns_file = os.path.join(data_dir, txt_files[sel_idx])
            except ValueError:
                pass
    except Exception:
        pass

    try:
        patterns, shape = load_patterns(patterns_file)
        N = shape[0] * shape[1]
    except Exception as e:
        print(f"\033[91mError cargando patrones: {e}\033[0m")
        return
        
    # Instanciar y entrenar la red de Hopfield
    network = HopfieldNetwork(N)
    network.train(patterns)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_header()
        
        print(f"\n{BOLD}Configuración actual:{RESET}")
        print(f"  - Tamaño de matriz cargado: {TEXT_CYAN}{shape[0]}x{shape[1]}{RESET} (Genérico para {N} neuronas)")
        print(f"  - Cantidad de patrones cargados: {TEXT_CYAN}{len(patterns)}{RESET} ({', '.join(patterns.keys())})")
        print("-" * 62)
        print(f"\n{BOLD}Seleccione una opción:{RESET}")
        print("  1. Ver patrones originales almacenados")
        print("  2. Simular recuperación de patrones con ruido (Paso a paso / Animación)")
        print("  3. Identificar estados espúreos (Parte b - Altamente ruidoso / Aleatorio)")
        print("  4. Salir")
        
        try:
            choice = input(f"\nOpción > ")
            if choice == '1':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
                print(f"\n{BOLD}{TEXT_GREEN}--- PATRONES ORIGINALES ALMACENADOS ---{RESET}")
                display_patterns_side_by_side(patterns, shape)
                input(f"Presione {BOLD}[Enter]{RESET} para volver al menú...")
            elif choice == '2':
                menu_ruido(network, patterns, shape)
            elif choice == '3':
                menu_spurious(network, patterns, shape)
            elif choice == '4':
                print(f"\n{TEXT_GREEN}¡Gracias por usar el simulador de Hopfield! Saliendo...{RESET}\n")
                break
            else:
                print(f"{TEXT_RED}Opción inválida. Intente nuevamente.{RESET}")
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n\n{TEXT_GREEN}Programa interrumpido. Saliendo...{RESET}\n")
            break

if __name__ == "__main__":
    main()
