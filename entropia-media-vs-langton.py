import numpy as np
import matplotlib.pyplot as plt
import math

def step_ca_r2(state, rule_table):
    """Avança o Autômato Celular de raio r=2 em 1 geração (anel periódico)."""
    # Desloca os arrays para pegar os 5 vizinhos
    ll = np.roll(state, 2)   # 2 à esquerda
    l  = np.roll(state, 1)   # 1 à esquerda
    c  = state               # centro
    r  = np.roll(state, -1)  # 1 à direita
    rr = np.roll(state, -2)  # 2 à direita
    
    # Calcula o índice da tabela de regras (0 a 31)
    idx = 16 * ll + 8 * l + 4 * c + 2 * r + rr
    return rule_table[idx]

def calcular_entropia(state):
    """Calcula a Entropia de Shannon espacial (1-bloco)."""
    p1 = np.sum(state) / len(state)
    p0 = 1.0 - p1
    
    termo_0 = p0 * math.log2(p0) if p0 > 0 else 0
    termo_1 = p1 * math.log2(p1) if p1 > 0 else 0
    
    return -(termo_0 + termo_1)

# --- Parâmetros da Simulação ---
L = 500             # Tamanho da malha
T = 300             # Passos de tempo para a regra "estabilizar"
num_amostras = 20   # Quantas regras sintéticas testar por cada valor de lambda
lambdas = np.linspace(0.0, 1.0, 21) # Valores de lambda de 0 a 1 (0, 0.05, 0.10...)

entropias_medias = []

print("Simulando... Isso pode levar alguns segundos.")

for lam in lambdas:
    entropia_acumulada = 0
    
    # Quantos bits '1' precisamos na tabela de 32 bits para atingir esse lambda?
    num_ones = int(round(lam * 32))
    
    for _ in range(num_amostras):
        # 1. Cria a tabela sintética exata
        rule_table = np.zeros(32, dtype=np.int8)
        rule_table[:num_ones] = 1
        np.random.shuffle(rule_table) # Embaralha para ficar aleatório
        
        # 2. Condição Inicial (Aleatória com densidade 0.5)
        state = np.random.randint(2, size=L)
        
        # 3. Evolui o AC
        for _ in range(T):
            state = step_ca_r2(state, rule_table)
            
        # 4. Calcula e acumula a entropia final
        entropia_acumulada += calcular_entropia(state)
        
    # Calcula a média de entropia para este lambda
    entropias_medias.append(entropia_acumulada / num_amostras)

# --- Plotagem do Gráfico ---
plt.figure(figsize=(10, 6))
plt.plot(lambdas, entropias_medias, marker='o', color='teal', linestyle='-', linewidth=2)
plt.title('Entropia Média Espacial vs. Parâmetro de Langton ($\lambda$)', fontsize=14)
plt.xlabel('Parâmetro de Langton ($\lambda$)', fontsize=12)
plt.ylabel('Entropia de Shannon Média (bits)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Destacando a região de transição (Borda do Caos)
plt.axvspan(0.15, 0.35, color='orange', alpha=0.2, label='Região de Transição (Maior Complexidade)')
plt.axvspan(0.65, 0.85, color='orange', alpha=0.2) # A curva é simétrica

plt.legend()
plt.tight_layout()
plt.show()