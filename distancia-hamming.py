import numpy as np
import matplotlib.pyplot as plt

def step_eca(state, rule_table):
    """
    Avança o Autômato Celular em 1 geração com condição de contorno periódica (anel).
    """
    # np.roll desloca o array. 
    # shift=1 pega o vizinho da esquerda, shift=-1 pega o da direita.
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    
    # Calcula o valor binário da vizinhança (índice de 0 a 7)
    idx = 4 * left + 2 * state + right
    
    # Retorna o novo estado baseado na tabela de regras
    return rule_table[idx]

# --- Parâmetros da Simulação ---
L = 100            # Tamanho do anel
T = 50             # Número de passos de tempo
rule_number = 30   # Regra do ECA

# Converte o número da regra (30) para uma tabela de busca (lookup table)
# 30 em binário é '00011110'. 
# Lemos invertido para mapear do índice 0 (000) ao 7 (111): [0, 1, 1, 1, 1, 0, 0, 0]
rule_string = np.binary_repr(rule_number, width=8)
rule_table = np.array([int(x) for x in rule_string[::-1]])

# --- Condições Iniciais ---
# C1: Geramos uma configuração aleatória de 0s e 1s
# (Usar uma base aleatória é ideal para ver o caos da Regra 30 se propagar)
np.random.seed(42) # Semente fixa para resultados reprodutíveis
C1 = np.random.randint(2, size=L)

# C2: Cópia exata de C1, mas com 1 bit central invertido
C2 = C1.copy()
centro = L // 2
C2[centro] = 1 - C2[centro] # Inverte o bit (se 0 vira 1, se 1 vira 0)

# --- Simulação ---
state1 = C1
state2 = C2
hamming_distances = []

for t in range(T + 1): # +1 para incluir o estado inicial (t=0)
    # A Distância de Hamming é a soma dos bits que são diferentes
    h_dist = np.sum(state1 != state2)
    hamming_distances.append(h_dist)
    
    # Avança os dois universos para a próxima geração
    state1 = step_eca(state1, rule_table)
    state2 = step_eca(state2, rule_table)

# --- Plotagem ---
plt.figure(figsize=(10, 6))
plt.plot(range(T + 1), hamming_distances, marker='o', color='crimson', linestyle='-', markersize=4)
plt.title(f'Evolução da Distância de Hamming - Regra {rule_number}', fontsize=14)
plt.xlabel('Passos de Tempo (t)', fontsize=12)
plt.ylabel('Distância de Hamming $H(t)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, T)
plt.ylim(0, max(hamming_distances) + 5)
plt.show()