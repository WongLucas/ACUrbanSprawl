import rasterio
from rasterio import features
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt, convolve # CORREÇÃO 1: convolve importado
import osmnx as ox
import geopandas as gpd

# -------------------------------------------------------------
# 1. CARREGANDO O ARQUIVO E EXTRAINDO COORDENADAS
# -------------------------------------------------------------
caminho_arquivo = 'nova-iguacu-2010.tif'
with rasterio.open(caminho_arquivo) as src:
    matriz_2010 = src.read(1)
    
    # Capturamos as bordas geográficas reais do seu arquivo
    norte, sul, leste, oeste = src.bounds.top, src.bounds.bottom, src.bounds.right, src.bounds.left
    
    # A "régua" matemática que alinha a matriz com a Terra
    transformacao = src.transform 
    linhas, colunas = matriz_2010.shape

# -------------------------------------------------------------
# 2. RECLASSIFICAÇÃO
# -------------------------------------------------------------
matriz_simples = np.zeros_like(matriz_2010)
matriz_simples[matriz_2010 == 24] = 1 # Urbano
matriz_simples[np.isin(matriz_2010, [3, 11, 12])] = 2 # Vegetação
matriz_simples[np.isin(matriz_2010, [15, 21, 25, 29, 30, 39, 41])] = 3 # Campo/Pastagem
matriz_simples[np.isin(matriz_2010, [33])] = 4 # Água

# -------------------------------------------------------------
# 3. BAIXANDO RUAS E TRENS (A Conexão com a Realidade)
# -------------------------------------------------------------
print("Conectando ao OpenStreetMap (pode levar alguns segundos)...")

# CORREÇÃO 2: Ordem das coordenadas atualizada para (esquerda, baixo, direita, cima)
estradas_reais = ox.features_from_bbox(norte, sul, leste, oeste, tags={'highway': ['motorway', 'trunk', 'primary', 'secondary']})
print("Rodovias baixadas! Buscando malha ferroviária...")

ferrovias_reais = ox.features_from_bbox(norte, sul, leste, oeste, tags={'railway': 'rail'})
print("Ferrovias baixadas! Carimbando as vias na matriz matemática...")

# -------------------------------------------------------------
# 4. RASTERIZAÇÃO E CÁLCULO DE DISTÂNCIA
# -------------------------------------------------------------
# Matrizes cheias de 1 (vazias de ruas)
malha_autoestrada = np.ones_like(matriz_simples)
malha_ferrovia = np.ones_like(matriz_simples)

# Se existirem estradas, "queimamos" as linhas na matriz substituindo 1 por 0
if not estradas_reais.empty:
    geom_estradas = [geom for geom in estradas_reais.geometry]
    malha_autoestrada = features.rasterize(geom_estradas, out_shape=(linhas, colunas), transform=transformacao, fill=1, default_value=0, all_touched=True)

if not ferrovias_reais.empty:
    geom_ferrovias = [geom for geom in ferrovias_reais.geometry]
    malha_ferrovia = features.rasterize(geom_ferrovias, out_shape=(linhas, colunas), transform=transformacao, fill=1, default_value=0, all_touched=True)

# Calcula a distância euclidiana de cada pixel para o '0' mais próximo
distancia_autoestrada = distance_transform_edt(malha_autoestrada)
distancia_ferrovia = distance_transform_edt(malha_ferrovia)

# Inverte o valor para gerar o PESO (onde 1.0 é colado na rua e 0.0 é muito longe)
peso_autoestrada = 1.0 - (distancia_autoestrada / np.max(distancia_autoestrada)) if np.max(distancia_autoestrada) > 0 else np.zeros_like(matriz_simples)
peso_ferrovia = 1.0 - (distancia_ferrovia / np.max(distancia_ferrovia)) if np.max(distancia_ferrovia) > 0 else np.zeros_like(matriz_simples)


# -------------------------------------------------------------
# 5. O MOTOR DO TEMPO: LAÇO DO AUTÔMATO CELULAR (2010 -> 2014)
# -------------------------------------------------------------
print("Iniciando a simulação temporal de expansão urbana...")

# Criamos uma cópia da matriz de 2010 para ser a nossa matriz dinâmica
matriz_simulada = np.copy(matriz_simples)

# Fator de crescimento calibra a agressividade do espalhamento urbano
fator_crescimento = 0.15 
anos_simulacao = 4

# CORREÇÃO 3: O kernel precisa ser definido antes do laço iniciar
tamanho_filtro = 5
kernel = np.ones((tamanho_filtro, tamanho_filtro))
centro = tamanho_filtro // 2
kernel[centro, centro] = 0 

kernel_imediato = np.ones((3, 3))
kernel_imediato[1, 1] = 0

for ano in range(1, anos_simulacao + 1):
    # 1. Isola a mancha urbana atual
    matriz_urbana_atual = np.where(matriz_simulada == 1, 1, 0)
    
    # 2. Recalcula a vizinhança (pois a cidade cresceu no ano anterior)
    densidade_atual = convolve(matriz_urbana_atual, kernel, mode='constant', cval=0)
    max_vizinhos = np.max(densidade_atual)
    peso_vizinhanca_atual = densidade_atual / max_vizinhos if max_vizinhos > 0 else densidade_atual
    
    # 3. Atualiza o Mapa de Aptidão (Vizinhança Dinâmica + Vias Estáticas)
    mapa_aptidao_atual = (peso_vizinhanca_atual * 0.5) + (peso_autoestrada * 0.25) + (peso_ferrovia * 0.25)
    
    # 4. A REGRA DE TRANSIÇÃO (A "Rolagem de Dados")
    probabilidade_virar_cidade = mapa_aptidao_atual * fator_crescimento
    dados_aleatorios = np.random.rand(linhas, colunas)
    celulas_que_mudaram = probabilidade_virar_cidade > dados_aleatorios

    # 5. APLICAÇÃO DAS REGRAS RESTRITIVAS (As "Travas" de Segurança)
    
    # Trava 1: Limite do Município e da Água
    # A expansão SÓ pode ocorrer em cima de Vegetação (2) ou Campo/Pastagem (3).
    # Isso resolve o vazamento para fora do mapa (0) e para dentro d'água (4).
    solo_permitido = (matriz_simulada == 2) | (matriz_simulada == 3)
    
    # Trava 2: Contiguidade Forte (Resolve os pixels "espalhados")
    # Exige que a célula tenha pelo menos 3 pixels urbanos ao redor para se urbanizar.
    toque_cidade = convolve(matriz_urbana_atual, kernel_imediato, mode='constant', cval=0)
    vizinhanca_forte = toque_cidade >= 1    

    # A máscara final une a rolagem dos dados estocásticos com as nossas travas lógicas
    mascara_transicao = celulas_que_mudaram & solo_permitido & vizinhanca_forte    
    # Atualiza o mapa com as novas áreas urbanas
    matriz_simulada[mascara_transicao] = 1
    
    print(f"Ano 201{ano} concluído. Novas áreas urbanizadas: {np.sum(mascara_transicao)} pixels.")

print("Simulação finalizada!")

# -------------------------------------------------------------
# 6. VISUALIZAÇÃO FINAL: ANTES E DEPOIS
# -------------------------------------------------------------
cores = ['black', '#e31a1c', '#33a02c', '#fdbf6f', '#1f78b4']
cmap_personalizado = ListedColormap(cores)

fig, eixos = plt.subplots(1, 2, figsize=(16, 8))

# Estado Inicial
im1 = eixos[0].imshow(matriz_simples, cmap=cmap_personalizado, interpolation='none')
eixos[0].set_title('Estado Inicial Real (2010)', fontsize=14)
eixos[0].axis('off')

# Resultado da Simulação
im2 = eixos[1].imshow(matriz_simulada, cmap=cmap_personalizado, interpolation='none')
eixos[1].set_title('Previsão do Autômato Celular (Simulado 2014)', fontsize=14)
eixos[1].axis('off')

# Adicionando a legenda global
cbar = plt.colorbar(im2, ax=eixos, shrink=0.5, ticks=[0, 1, 2, 3, 4], orientation='vertical')
cbar.ax.set_yticklabels(['Fora do Município', 'Urbano (1)', 'Vegetação (2)', 'Campo/Pastagem (3)', 'Água (4)'])

plt.show()