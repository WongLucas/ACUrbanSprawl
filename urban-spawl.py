import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Carregando seu arquivo (como você já fez)
caminho_arquivo = 'nova-iguacu-2010.tif'
with rasterio.open(caminho_arquivo) as src:
    matriz_2010 = src.read(1)

# 1. CRIANDO A MATRIZ SIMPLIFICADA
# Vamos criar uma cópia zerada para não estragar a original
matriz_simples = np.zeros_like(matriz_2010)

# 2. RECLASSIFICAÇÃO (A Mágica do NumPy)
# Transformando o código 24 em Classe 1 (Urbano)
matriz_simples[matriz_2010 == 24] = 1

# Transformando códigos de Floresta/Vegetação em Classe 2 (Natureza)
matriz_simples[np.isin(matriz_2010, [3, 11, 12])] = 2

# Transformando Pastagem/Agricultura em Classe 3 (Áreas Abertas/Vulneráveis)
matriz_simples[np.isin(matriz_2010, [15, 21, 25, 29, 30, 39, 41])] = 3

# Transformando Rio/Água em Classe 4 (Água)
matriz_simples[np.isin(matriz_2010, [33])] = 4

# O valor 0 (Fundo) continuará sendo 0.

# 3. VISUALIZAÇÃO DO SEU GRAFO DE ESTADOS
# Definindo cores personalizadas: Preto(Fundo), Vermelho(Urbano), Verde(Natureza), Amarelo(Campo), Azul(Água)
cores = ['black', '#e31a1c', '#33a02c', '#fdbf6f', '#1f78b4']
cmap_personalizado = ListedColormap(cores)

plt.figure(figsize=(12, 10))
# Plotamos apenas a matriz simples usando nosso mapa de cores
img = plt.imshow(matriz_simples, cmap=cmap_personalizado, interpolation='none')

# Adicionando uma legenda manual para ficar elegante
cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4], shrink=0.8)
cbar.ax.set_yticklabels(['Fundo', 'Urbano (1)', 'Vegetação (2)', 'Campo/Pastagem (3)', 'Água (4)'])

plt.title('Uso do Solo Simplificado - Nova Iguaçu (2010)', fontsize=14)
plt.axis('off') # Esconde os eixos X e Y
plt.show()