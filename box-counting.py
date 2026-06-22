import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Simulação do Autômato Celular (Regra 90)
# ==========================================
L = 1025  # Tamanho da malha
T = 512   # Passos de tempo

# Inicializa a malha com zeros
grid = np.zeros((T, L), dtype=np.int8)

# Condição inicial: apenas a célula central é 1
centro = L // 2
grid[0, centro] = 1

# Evolução da Regra 90
for t in range(1, T):
    # A Regra 90 é um simples XOR (Ou Exclusivo) entre o vizinho da esquerda e da direita
    esquerda = np.roll(grid[t-1], 1)
    direita = np.roll(grid[t-1], -1)
    grid[t] = np.bitwise_xor(esquerda, direita)

# ==========================================
# 2. Algoritmo de Box-Counting
# ==========================================
# Para o algoritmo de caixas funcionar perfeitamente (sem sobras), 
# precisamos de uma imagem quadrada onde o lado seja potência de 2.
# Vamos colocar nossa simulação de 512x1025 dentro de uma "tela" de 1024x1024.
img_padded = np.zeros((1024, 1024), dtype=np.int8)

# A parte ativa do triângulo ocupa das colunas 1 a 1023. 
# Pegamos as primeiras 1024 colunas para caber na nossa tela.
img_padded[0:512, 0:1024] = grid[:, 0:1024]

# Definimos os tamanhos das caixas epsilon (potências de 2)
tamanhos_caixa = [2, 4, 8, 16, 32, 64, 128, 256, 512]
contagens = []

for tamanho in tamanhos_caixa:
    # Truque do Numpy: Redimensiona a imagem 2D em 4D para agrupar os pixels em caixas
    blocos = img_padded.reshape(1024 // tamanho, tamanho, 1024 // tamanho, tamanho)
    
    # Soma todos os pixels dentro de cada caixa
    soma_caixas = blocos.sum(axis=(1, 3))
    
    # Conta quantas caixas resultaram em valor maior que 0 (possuem parte do fractal)
    caixas_com_fractal = np.count_nonzero(soma_caixas)
    contagens.append(caixas_com_fractal)

# ==========================================
# 3. Cálculo da Dimensão Fractal (Regressão)
# ==========================================
# Transformação para a escala Log-Log (Base 2)
# x = log(1 / epsilon) = -log(epsilon)
x = -np.log2(tamanhos_caixa)
y = np.log2(contagens)

# Faz o ajuste linear (y = mx + c) para encontrar a inclinação da reta (m)
coeficientes = np.polyfit(x, y, 1)
dimensao_fractal = coeficientes[0] # A inclinação da reta é a dimensão fractal estimada
intercept = coeficientes[1]

# Linha de tendência teórica para o plot
linha_ajustada = dimensao_fractal * x + intercept

print(f"Dimensão Fractal Estimada (Box-Counting): {dimensao_fractal:.4f}")
print(f"Dimensão Teórica (Sierpinski): {np.log2(3):.4f}")

# ==========================================
# 4. Plotagem dos Gráficos
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: Espaço-Tempo
ax1.imshow(grid, cmap='binary', aspect='auto', interpolation='nearest')
ax1.set_title("Diagrama Espaço-Tempo - Regra 90", fontsize=14)
ax1.set_xlabel("Células", fontsize=12)
ax1.set_ylabel("Tempo (t)", fontsize=12)

# Gráfico 2: Regressão Log-Log do Box-Counting
ax2.scatter(x, y, color='black', label='Dados das Caixas $N(\epsilon)$', zorder=5)
ax2.plot(x, linha_ajustada, color='crimson', linewidth=2.5, 
         label=f'Ajuste Linear: $D_f \\approx {dimensao_fractal:.3f}$')
ax2.set_title("Análise de Box-Counting (Escala Log-Log)", fontsize=14)
ax2.set_xlabel(r"$\log_2(1/\epsilon)$", fontsize=12)
ax2.set_ylabel(r"$\log_2(N(\epsilon))$", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()