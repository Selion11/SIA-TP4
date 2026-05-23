import os
import numpy as np

def load_patterns(filepath):
    """
    Carga patrones desde un archivo de texto de forma dinámica y genérica.
    
    Formato del archivo:
    #: NombrePatron
    * * * . .
    . . * . .
    
    Retorna:
        patterns: dict de {nombre: np.array de forma (N,)} donde N = alto * ancho
        shape: tupla de (alto, ancho) de los patrones cargados.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo de patrones en: {filepath}")
        
    patterns = {}
    current_name = None
    current_grid = []
    shape = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#:'):
                # Guardar el patrón anterior si existe
                if current_name is not None and current_grid:
                    pattern_matrix = np.array(current_grid, dtype=int)
                    if shape is None:
                        shape = pattern_matrix.shape
                    elif shape != pattern_matrix.shape:
                        raise ValueError(
                            f"El patrón '{current_name}' tiene forma {pattern_matrix.shape}, "
                            f"pero se esperaba {shape}. Todos los patrones deben tener el mismo tamaño."
                        )
                    patterns[current_name] = pattern_matrix.flatten()
                    current_grid = []
                current_name = line[2:].strip()
            elif line.startswith('#'):
                continue
            else:
                if current_name is None:
                    continue
                row = []
                # Limpiar espacios
                clean_line = line.replace(" ", "").replace("\t", "")
                for char in clean_line:
                    if char == '*':
                        row.append(1)
                    elif char == '.':
                        row.append(-1)
                    else:
                        raise ValueError(
                            f"Carácter inválido '{char}' en el patrón '{current_name}'. "
                            f"Solo se permiten '*' y '.'."
                        )
                if row:
                    current_grid.append(row)
                    
    # Guardar el último patrón leído
    if current_name is not None and current_grid:
        pattern_matrix = np.array(current_grid, dtype=int)
        if shape is None:
            shape = pattern_matrix.shape
        elif shape != pattern_matrix.shape:
            raise ValueError(
                f"El patrón '{current_name}' tiene forma {pattern_matrix.shape}, "
                f"pero se esperaba {shape}. Todos los patrones deben tener el mismo tamaño."
            )
        patterns[current_name] = pattern_matrix.flatten()
        
    if not patterns:
        raise ValueError("No se pudo cargar ningún patrón válido desde el archivo.")
        
    return patterns, shape

def pattern_to_str(vector, shape, active_char="█", inactive_char="░", double_width=True):
    """
    Convierte un vector 1D de +1/-1 en una representación visual en cadena de caracteres.
    Para que las matrices de consola no se vean 'aplastadas' verticalmente, 
    usamos dos caracteres por columna por defecto (double_width=True).
    """
    matrix = vector.reshape(shape)
    lines = []
    char_act = active_char * 2 if double_width else active_char
    char_inact = inactive_char * 2 if double_width else inactive_char
    
    for row in matrix:
        line = "".join(char_act if val == 1 else char_inact for val in row)
        lines.append(line)
    return "\n".join(lines)
