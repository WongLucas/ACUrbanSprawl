import rasterio
from rasterio import features
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt, convolve
import osmnx as ox
import geopandas as gpd
from sklearn.metrics import accuracy_score, cohen_kappa_score

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

# Mantemos a ordem (norte, sul, leste, oeste) que funcionou perfeitamente no seu ambiente
estradas_reais = ox.features_from_bbox(norte, sul, leste, oeste, tags={'highway': ['motorway', 'trunk', 'primary', 'secondary']})
print("Rodovias baixadas! Buscando malha ferroviária...")

ferrovias_reais = ox.features_from_bbox(norte, sul, leste, oeste, tags={'railway': 'rail'})
print("Ferrovias baixadas! Carimbando as vias na matriz matemática...")

# -------------------------------------------------------------
# 4. RASTERIZAÇÃO E CÁLCULO DE DISTÂNCIA
# -------------------------------------------------------------
malha_autoestrada = np.ones_like(matriz_simples)
malha_ferrovia = np.ones_like(matriz_simples)

if not estradas_reais.empty:
    geom_estradas = [geom for geom in estradas_reais.geometry]
    malha_autoestrada = features.rasterize(geom_estradas, out_shape=(linhas, colunas), transform=transformacao, fill=1, default_value=0, all_touched=True)

if not ferrovias_reais.empty:
    geom_ferrovias = [geom for geom in ferrovias_reais.geometry]
    malha_ferrovia = features.rasterize(geom_ferrovias, out_shape=(linhas, colunas), transform=transformacao, fill=1, default_value=0, all_touched=True)

distancia_autoestrada = distance_transform_edt(malha_autoestrada)
distancia_ferrovia = distance_transform_edt(malha_ferrovia)

peso_autoestrada = 1.0 - (distancia_autoestrada / np.max(distancia_autoestrada)) if np.max(distancia_autoestrada) > 0 else np.zeros_like(matriz_simples)
peso_ferrovia = 1.0 - (distancia_ferrovia / np.max(distancia_ferrovia)) if np.max(distancia_ferrovia) > 0 else np.zeros_like(matriz_simples)


# -------------------------------------------------------------
# 5. O MOTOR DO TEMPO: LAÇO DO AUTÔMATO CELULAR (2010 -> 2014)
# -------------------------------------------------------------
print("Iniciando a simulação temporal de expansão urbana...")

matriz_simulada = np.copy(matriz_simples)
fator_crescimento = 0.35 
anos_simulacao = 4

# Filtro 5x5 (Calcula a Pressão/Aptidão da Região)
tamanho_filtro = 5
kernel = np.ones((tamanho_filtro, tamanho_filtro))
centro = tamanho_filtro // 2
kernel[centro, centro] = 0 

# Filtro 3x3 (A Trava de Contiguidade Imediata - Evita saltos/pixels isolados)
kernel_imediato = np.ones((3, 3))
kernel_imediato[1, 1] = 0

for ano in range(1, anos_simulacao + 1):
    matriz_urbana_atual = np.where(matriz_simulada == 1, 1, 0)
    
    # Pressão global (Filtro 5x5)
    densidade_atual = convolve(matriz_urbana_atual, kernel, mode='constant', cval=0)
    max_vizinhos = np.max(densidade_atual)
    peso_vizinhanca_atual = densidade_atual / max_vizinhos if max_vizinhos > 0 else densidade_atual
    
    # Mapa de Aptidão MCE
    mapa_aptidao_atual = (peso_vizinhanca_atual * 0.5) + (peso_autoestrada * 0.25) + (peso_ferrovia * 0.25)
    
    # Regra Estocástica (Sorteio)
    probabilidade_virar_cidade = mapa_aptidao_atual * fator_crescimento
    dados_aleatorios = np.random.rand(linhas, colunas)
    celulas_que_mudaram = probabilidade_virar_cidade > dados_aleatorios
    
    # Aplicação das Travas Restritivas
    solo_permitido = (matriz_simulada == 2) | (matriz_simulada == 3) # Só cresce no mato ou pasto
    toque_cidade = convolve(matriz_urbana_atual, kernel_imediato, mode='constant', cval=0)
    vizinhanca_forte = toque_cidade >= 1 # Tem que ter pelo menos 1 pixel vizinho colado
    
    mascara_transicao = celulas_que_mudaram & solo_permitido & vizinhanca_forte
    matriz_simulada[mascara_transicao] = 1
    
    print(f"Ano 201{ano} concluído. Novas áreas urbanizadas: {np.sum(mascara_transicao)} pixels.")

print("Simulação finalizada!")

# -------------------------------------------------------------
# 6. VALIDAÇÃO MATEMÁTICA: MÉTRICAS PURAS DE ESPALHAMENTO
# -------------------------------------------------------------
print("Iniciando a validação contra o mapa real de 2014...")

caminho_gabarito = 'nova-iguacu-2014.tif'
try:
    with rasterio.open(caminho_gabarito) as src:
        matriz_2014_bruta = src.read(1)
        
    matriz_gabarito = np.zeros_like(matriz_2014_bruta)
    matriz_gabarito[matriz_2014_bruta == 24] = 1 
    matriz_gabarito[np.isin(matriz_2014_bruta, [3, 11, 12])] = 2
    matriz_gabarito[np.isin(matriz_2014_bruta, [15, 21, 25, 29, 30, 39, 41])] = 3
    matriz_gabarito[np.isin(matriz_2014_bruta, [33])] = 4

    # 1. Definindo o que podia mudar
    mascara_vulneravel = (matriz_simples == 2) | (matriz_simples == 3)

    # 2. O que MUDOU DE VERDADE (Era mato em 2010, virou cidade no Gabarito 2014)
    crescimento_real = mascara_vulneravel & (matriz_gabarito == 1)
    
    # 3. O que MUDOU NO MODELO (Era mato em 2010, virou cidade na Simulação)
    crescimento_simulado = mascara_vulneravel & (matriz_simulada == 1)

    # 4. A INTERSEÇÃO (Onde o modelo cravou o pixel exato do espalhamento)
    acertos_na_mosca = crescimento_real & crescimento_simulado

    # Somando os pixels
    total_real = np.sum(crescimento_real)
    total_simulado = np.sum(crescimento_simulado)
    total_acertos = np.sum(acertos_na_mosca)

    # Evitando divisão por zero caso o modelo não tenha gerado nada
    taxa_acerto_realidade = (total_acertos / total_real * 100) if total_real > 0 else 0
    taxa_acerto_modelo = (total_acertos / total_simulado * 100) if total_simulado > 0 else 0

    print("-" * 50)
    print("RESULTADOS DA EXPANSÃO URBANA (MÉTRICAS REAIS)")
    print("-" * 50)
    print(f"-> A cidade cresceu REALMENTE: {total_real} pixels.")
    print(f"-> O Autômato previu o crescimento de: {total_simulado} pixels.")
    print(f"-> Acertos exatos (Pixel sobre Pixel): {total_acertos} pixels.")
    print("-" * 50)
    print(f"Sensibilidade: Do que realmente cresceu, previmos {taxa_acerto_realidade:.2f}%")
    print(f"Precisão: De tudo que nosso modelo urbanizou, {taxa_acerto_modelo:.2f}% estava correto")
    print("-" * 50)
    
except FileNotFoundError:
    print("Arquivo de 2014 não encontrado. Pulando validação.")
    matriz_gabarito = np.zeros_like(matriz_simulada)

# -------------------------------------------------------------
# 7. VISUALIZAÇÃO DE ERROS E ACERTOS (MAPA DE CONCORDÂNCIA)
# -------------------------------------------------------------
# Mapeando os erros para o gráfico final
mapa_concordancia = np.zeros_like(matriz_simulada)
mapa_concordancia[(matriz_gabarito == 1) & (matriz_simulada == 1)] = 1 # Acerto (Verde)
mapa_concordancia[(matriz_gabarito != 1) & (matriz_simulada == 1)] = 2 # Falso Positivo (Azul)
mapa_concordancia[(matriz_gabarito == 1) & (matriz_simulada != 1)] = 3 # Falso Negativo (Vermelho)

# DEFINIÇÃO DAS CORES (A Correção Final)
cores_solo = ['black', '#e31a1c', '#33a02c', '#fdbf6f', '#1f78b4']
cmap_personalizado = ListedColormap(cores_solo)

cores_erro = ['black', '#2ca02c', '#1f77b4', '#d62728'] 
cmap_erro = ListedColormap(cores_erro)

# Construindo a imagem final
fig, eixos = plt.subplots(1, 3, figsize=(18, 6))

eixos[0].imshow(matriz_simulada, cmap=cmap_personalizado, interpolation='none')
eixos[0].set_title('Nossa Previsão (Simulado 2014)', fontsize=12)
eixos[0].axis('off')

eixos[1].imshow(matriz_gabarito, cmap=cmap_personalizado, interpolation='none')
eixos[1].set_title('Gabarito (Realidade 2014)', fontsize=12)
eixos[1].axis('off')

im3 = eixos[2].imshow(mapa_concordancia, cmap=cmap_erro, interpolation='none')
eixos[2].set_title('Mapa de Concordância (Verde=Acerto, Azul/Vermelho=Erros)', fontsize=12)
eixos[2].axis('off')

plt.tight_layout()
plt.show()