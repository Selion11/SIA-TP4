import numpy as np

class HopfieldNetwork:
    def __init__(self, N):
        """
        Inicializa una red de Hopfield genérica con N neuronas.
        """
        self.N = N
        self.W = np.zeros((N, N))
        self.trained_patterns = {}  # {nombre: vector_original}
        
    def train(self, patterns_dict):
        """
        Entrena la red usando el conjunto de patrones provisto en un diccionario.
        Usa la fórmula matricial: W = (1/N) * K * K^T, con la diagonal seteada a cero.
        
        Args:
            patterns_dict: dict de {nombre: np.array de forma (N,)}
        """
        self.trained_patterns = patterns_dict
        if not patterns_dict:
            raise ValueError("No se pasaron patrones para entrenar.")
            
        # Apilar todos los patrones como columnas para formar la matriz K (N x P)
        pattern_vectors = list(patterns_dict.values())
        K = np.column_stack(pattern_vectors)  # Dimensión (N, P)
        
        # Calcular W = (1/N) * K * K^T
        # K tiene dimensión (N, P), por lo que K * K^T tiene dimensión (N, N)
        self.W = (1.0 / self.N) * np.dot(K, K.T)
        
        # Seteamos la diagonal principal a 0 (W_ii = 0) para evitar la auto-retroalimentación
        np.fill_diagonal(self.W, 0)

    def calculate_energy(self, state):
        """
        Calcula la función de energía de Lyapunov para el estado actual de la red.
        Formula: E = -0.5 * S^T * W * S
        """
        return -0.5 * np.dot(state.T, np.dot(self.W, state))
        
    def add_noise(self, state, noise_fraction=0.2):
        """
        Genera una versión ruidosa de un estado invirtiendo (flip) el signo 
        de una fracción aleatoria de las neuronas.
        """
        noisy_state = state.copy()
        num_to_flip = int(self.N * noise_fraction)
        # Seleccionar índices aleatorios únicos
        flip_indices = np.random.choice(self.N, num_to_flip, replace=False)
        noisy_state[flip_indices] *= -1
        return noisy_state

    def add_exact_noise(self, state, num_bits_to_flip):
        """
        Genera ruido invirtiendo exactamente una cantidad dada de bits.
        """
        noisy_state = state.copy()
        num_to_flip = min(self.N, max(0, num_bits_to_flip))
        flip_indices = np.random.choice(self.N, num_to_flip, replace=False)
        noisy_state[flip_indices] *= -1
        return noisy_state

    def evolve_asynchronous_step(self, state, neuron_idx):
        """
        Realiza la actualización asincrónica de una sola neurona específica.
        
        Formula: s_i(t+1) = sign( sum_j W_ij * s_j(t) )
        Si la suma de activaciones es exactamente 0, la neurona conserva su estado actual.
        
        Retorna:
            new_state: el estado actualizado (copia o in-place)
            changed: booleano que indica si el estado de la neurona cambió
        """
        activation = np.dot(self.W[neuron_idx], state)
        old_val = state[neuron_idx]
        
        if activation > 0:
            new_val = 1
        elif activation < 0:
            new_val = -1
        else:
            new_val = old_val  # Mantener estado en caso de empate exacto
            
        changed = (new_val != old_val)
        new_state = state.copy()
        new_state[neuron_idx] = new_val
        return new_state, changed

    def evolve(self, initial_state, max_epochs=20, random_order=True):
        """
        Evoluciona la red de forma asincrónica hasta que converja.
        La convergencia ocurre cuando una época completa (pasar por todas las neuronas)
        se realiza sin ningún cambio en el estado.
        
        Args:
            initial_state: vector 1D de tamaño N con el estado inicial.
            max_epochs: número máximo de épocas completas.
            random_order: si es True, baraja el orden de actualización de las neuronas en cada época.
            
        Retorna:
            history: lista de diccionarios con información detallada de cada paso:
                     [{'state': state, 'energy': energy, 'neuron_updated': idx, 'changed': bool, 'epoch': int}]
            converged: booleano de si llegó a un punto estable.
        """
        state = initial_state.copy()
        history = []
        
        # Registrar estado inicial
        initial_energy = self.calculate_energy(state)
        history.append({
            'state': state.copy(),
            'energy': initial_energy,
            'neuron_updated': -1,
            'changed': False,
            'epoch': 0,
            'description': 'Estado Inicial'
        })
        
        converged = False
        epoch = 0
        
        while epoch < max_epochs and not converged:
            epoch += 1
            # Definir orden de actualización de las neuronas para esta época
            indices = np.arange(self.N)
            if random_order:
                np.random.shuffle(indices)
                
            changes_in_epoch = 0
            
            for idx in indices:
                state, changed = self.evolve_asynchronous_step(state, idx)
                if changed:
                    changes_in_epoch += 1
                    current_energy = self.calculate_energy(state)
                    history.append({
                        'state': state.copy(),
                        'energy': current_energy,
                        'neuron_updated': idx,
                        'changed': True,
                        'epoch': epoch,
                        'description': f"Época {epoch}: Neurona {idx} cambió a {state[idx]}"
                    })
                else:
                    # Opcional: registrar paso sin cambio para el historial de análisis detallado,
                    # pero para no sobrecargar el historial en la interfaz visual, solo registramos cambios
                    # o guardamos una traza ligera.
                    pass
            
            # Criterio de convergencia: si no hubo ningún cambio en toda la época
            if changes_in_epoch == 0:
                converged = True
                break
                
        # Registrar estado final (incluso si no cambió en la última época, para marcar la convergencia)
        final_energy = self.calculate_energy(state)
        history.append({
            'state': state.copy(),
            'energy': final_energy,
            'neuron_updated': -1,
            'changed': False,
            'epoch': epoch,
            'description': 'Convergencia Alcanzada' if converged else 'Límite de Épocas Alcanzado'
        })
        
        return history, converged

    def get_closest_pattern(self, state):
        """
        Compara el estado dado con los patrones de entrenamiento 
        y devuelve el nombre del patrón más cercano y su distancia Hamming.
        """
        closest_name = None
        min_distance = self.N + 1
        
        for name, pattern in self.trained_patterns.items():
            # Distancia Hamming: número de bits diferentes
            distance = np.sum(state != pattern)
            # También consideramos el patrón invertido (Hopfield es simétrico respecto al signo)
            distance_inv = np.sum(-state != pattern)
            
            effective_dist = min(distance, distance_inv)
            is_inverted = distance_inv < distance
            
            if effective_dist < min_distance:
                min_distance = effective_dist
                closest_name = f"-{name}" if is_inverted else name
                
        return closest_name, min_distance
