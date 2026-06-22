import numpy as np
import math

def step_eca(state, rule_table):
    """Avança o Autômato Celular em 1 geração (anel periódico)."""
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    idx = 4 * left + 2 * state + right
    return rule_table[idx]

def calcular_entropia_shannon(p1):
    """
    Aplica a fórmula H = - sum(p_i * log2(p_i)).
    Trata o caso onde p=0 para evitar erro de log(0).
    """
    p0 = 1.0 - p1
    
    # Se a probabilidade for 0, o limite de p*log(p) é 0.
    termo_0 = p0 * math.log2(p0) if p0 > 0 else 0
    termo_1 = p1 * math.log2(p1) if p1 > 0 else 0
    
    H = -(termo_0 + termo_1)
    return H

# --- Parâmetros da Simulação ---
L = 10000          # Usamos uma grade bem grande para ter precisão estatística
T = 100            # Iterações solicitadas no exercício
densidade_inicial = 0.5

# Escolha as regras que você quer apresentar no seu trabalho aqui:
regras_para_testar = [30, 184] 

print(f"--- Cálculo da Entropia Espacial após {T} iterações ---\n")

for regra in regras_para_testar:
    # 1. Montar a tabela da regra
    rule_string = np.binary_repr(regra, width=8)
    rule_table = np.array([int(x) for x in rule_string[::-1]])
    
    # 2. Condição Inicial (Aleatória com densidade 0.5)
    np.random.seed(4) # Semente fixa para você poder reproduzir o mesmo resultado
    state = np.random.choice([0, 1], size=L, p=[1 - densidade_inicial, densidade_inicial])
    
    # 3. Evoluir o sistema por 100 passos
    for _ in range(T):
        state = step_eca(state, rule_table)
        
    # 4. Calcular p1 (probabilidade de encontrar o estado 1)
    p1 = np.sum(state) / L
    
    # 5. Calcular a Entropia
    H = calcular_entropia_shannon(p1)
    
    print(f"Regra {regra:3d} | Densidade final (p1): {p1:.4f} | Entropia (H): {H:.4f} bits")