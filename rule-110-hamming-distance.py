import numpy as np
import matplotlib.pyplot as plt

def step_eca(state):
    """Avança a Regra 110 em 1 geração (anel periódico)."""
    rule_table = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int8) # Regra 110 invertida
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    idx = 4 * left + 2 * state + right
    return rule_table[idx]

# --- Parâmetros ---
L = 100            # Malha
T_setup = 200      # Tempo para o éter estabilizar
T_sim = 50         # Tempo de medição da velocidade

# --- Condição Inicial Específica ---
# Começamos com éter e um glider 'A' comum
# Fonte da configuração: Genaro J. Martinez
ether_pattern = np.array([0,0,0,1,0,0,1,1,0,1,1,1,1,1]) # Um padrão base do éter
grid_ether = np.tile(ether_pattern, L // len(ether_pattern) + 1)[:L]

# Inserimos uma perturbação para criar o glider no centro
state = grid_ether.copy()
centro = L // 2
state[centro-2:centro+2] = [1,1,0,1] # Pequena semente de glider

# --- Pré-simulação (para o glider se formar e se separar) ---
for _ in range(T_setup):
    state = step_eca(state)

# --- Simulação de Medição ---
history = [state]
# Encontra a posição inicial do glider (vamos usar o 'centro de massa' dos bits 1)
pos_inicial = np.mean(np.where(state == 1))

for _ in range(T_sim):
    state = step_eca(state)
    history.append(state)

# Encontra a posição final do glider
pos_final = np.mean(np.where(state == 1))

# --- Cálculo da Velocidade ---
deslocamento = pos_final - pos_inicial
velocidade = deslocamento / T_sim

print(f"--- Medição de Velocidade via Código ---")
print(f"Passos de tempo analisados: {T_sim}")
print(f"Deslocamento horizontal: {deslocamento:.2f} células")
print(f"Velocidade calculada: {velocidade:.4f} células/passo")

# --- Visualização ---
plt.figure(figsize=(10, 6))
plt.imshow(history, cmap='binary', aspect='auto', interpolation='nearest')
plt.title(f'Propagação de Glider na Regra 110\nVelocidade Estimada: {velocidade:.3f}', fontsize=12)
plt.xlabel('Espaço (Células)', fontsize=10)
plt.ylabel('Tempo (t)', fontsize=10)
plt.show()