# Red de Hopfield Genérica y Escalable (SIA-TP4)

Este repositorio contiene una implementación genérica y altamente configurable del **Modelo de Red Neuronal de Hopfield** de forma asincrónica. Está diseñado bajo una arquitectura modular y limpia, preparada para soportar matrices de cualquier dimensión (por ejemplo, expandirse de $5 \times 5$ a tamaños mayores como $10 \times 10$ o $15 \times 15$ para representar caracteres complejos como **Hiragana y Katakana** de japonés).

---

## 🛠️ Arquitectura y Modularidad Genérica

El proyecto está diseñado para desacoplar completamente la definición visual de los caracteres de la lógica matemática del modelo de Hopfield:

1.  **`data/patterns.txt` (Base de Datos Visual):** Define los caracteres usando caracteres sencillos (`*` para píxeles activos y `.` para inactivos). El cargador dinámico detecta de forma autónoma las filas y columnas. **Para cambiar de tamaño o agregar nuevos alfabetos, solo tienes que reescribir este archivo sin tocar una sola línea de código Python.**
2.  **`src/pattern_loader.py` (Cargador y Renderizador):** Parsea el archivo, valida la consistencia dimensional de todos los patrones, los aplana a vectores 1D de tamaño $N$, y proporciona funciones de renderizado de alta definición para consola.
3.  **`src/hopfield.py` (Lógica de Hopfield):** Implementa el entrenamiento de Hebb mediante el truco matricial ($W = \frac{1}{N} K K^T$), cálculo de energía de Lyapunov ($E = -\frac{1}{2} S^T W S$), aplicación de ruido uniforme y evolución asincrónica paso a paso.
4.  **`main.py` (Interfaz de Consola de Alta Fidelidad):** Panel interactivo con animaciones en consola usando colores ANSI y gráficos de bloques de alta definición para acompañar el paso a paso.

---

## 🚀 Cómo Ejecutar en WSL (Windows Subsystem for Linux)

1.  Abre tu terminal de WSL.
2.  Navega al directorio del proyecto en la máquina virtual:
    ```bash
    cd "/mnt/c/Users/Feli/OneDrive/Documentos/ITBA/20261Q/sia/SIA-TP4"
    ```
3.  Ejecuta el script interactivo principal:
    ```bash
    python3 main.py
    ```

---

## 📖 Guía Explicativa: Conceptos Clave

### 1. Actualización Asincrónica (Neurona por Neurona)
A diferencia de la actualización sincrónica (donde todas las neuronas calculan su nuevo estado al mismo tiempo), la **actualización asincrónica** selecciona **una sola neurona** en cada paso, calcula su activación ponderada, y actualiza su estado de inmediato en el vector global.

La regla de actualización para la neurona $i$ es:
$$s_i(t+1) = \text{sign}\left( \sum_{j=1}^N W_{ij} s_j(t) \right)$$

En esta implementación, para evitar sesgos secuenciales, en cada **época** de evolución barajamos de forma aleatoria el orden de evaluación de las $N$ neuronas (índices $\{1, \dots, N\}$).

### 2. Criterio de Convergencia
La red evoluciona de forma asincrónica hasta que se alcanza un **punto estable (punto fijo)**. El criterio de parada implementado determina que la red ha convergido si **se realiza una época completa (se evalúan las $N$ neuronas) y no se produce ningún cambio de signo en ninguna de ellas**.

### 3. El Truco de la Matriz de Pesos $W$
Para evitar un bucle anidado por cada patrón, entrenamos la red de manera puramente matricial:
1.  Creamos la matriz $K$ de dimensión $N \times P$ (donde $P$ es la cantidad de patrones y cada columna es un patrón).
2.  Calculamos los pesos iniciales como:
    $$W = \frac{1}{N} K K^T$$
3.  Establecemos la diagonal de la matriz a cero (`np.fill_diagonal(W, 0)`) para impedir la auto-retroalimentación de las neuronas, asegurando que la energía decaiga estrictamente.

### 4. Función de Energía de Lyapunov
Cada estado del sistema tiene asociada una energía calculada como:
$$E = -\frac{1}{2} S^T W S$$
A medida que la red evoluciona asíncronamente corrigiendo el ruido, la energía **decae monótonamente** (o se mantiene constante), nunca aumenta. Esto demuestra matemáticamente que la red converge a un mínimo local.

### 5. Estados Espúreos (Falsas Memorias)
La red tiene mínimos locales adicionales creados por la combinación matemática de las memorias cargadas. Al ingresar estados aleatorios o con excesivo ruido (superior al 50%), la red suele caer en estos **estados espúreos** en lugar de recuperar las letras originales. El programa interactivo cuenta con una opción específica para detectar y analizar estos estados.

---

## 🇯🇵 Expansión Futura a Hiragana y Katakana (> 5x5)

La arquitectura es **100% genérica**. Si deseas utilizar matrices más grandes (por ejemplo, de $10 \times 10$ o $15 \times 15$) para caracteres en japonés:

1.  Abre `data/patterns.txt`.
2.  Reemplaza las letras actuales por tus dibujos de Hiragana o Katakana respetando la dimensión elegida. Por ejemplo, para un carácter de $10 \times 10$:
    ```text
    #: Hiragana_A
    . . * * * * * * . .
    . * * . . . . * * .
    * * . . . . . . * *
    * * . . * * . . * *
    * * * * * * * * * *
    * * . . * * . . * *
    * * . . . . . . * *
    . * * . . . . * * .
    . . * * * * * * . .
    . . . . . . . . . .
    ```
3.  ¡Y listo! Al iniciar `python3 main.py`, la aplicación detectará el nuevo tamaño ($10 \times 10$), creará la red con $N = 100$ neuronas, calculará la nueva matriz de pesos $W$ de $100 \times 100$, y renderizará la grilla de alta fidelidad automáticamente de forma perfecta.

