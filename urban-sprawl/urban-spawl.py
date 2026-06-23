import os
import json
import rasterio
from rasterio import features
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation # NOVO: Biblioteca para gerar o GIF
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt, convolve
import osmnx as ox
import geopandas as gpd

# -------------------------------------------------------------
# 1. CARREGANDO O ARQUIVO E EXTRAINDO COORDENADAS
# -------------------------------------------------------------
caminho_arquivo = 'nova-iguacu-2010.tif'
with rasterio.open(caminho_arquivo) as src:
    matriz_2010 = src.read(1)
    
    norte, sul, leste, oeste = src.bounds.top, src.bounds.bottom, src.bounds.right, src.bounds.left
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
# 3. SISTEMA DE CACHE: VIAS REAIS (OFFLINE/ONLINE)
# -------------------------------------------------------------
ficheiro_estradas = 'cache_estradas_ni.geojson'
ficheiro_ferrovias = 'cache_ferrovias_ni.geojson'

if os.path.exists(ficheiro_estradas) and os.path.exists(ficheiro_ferrovias):
    print("Arquivos locais encontrados! Carregando vias da memória (bypass fiona)...")
    
    with open(ficheiro_estradas, 'r', encoding='utf-8') as f:
        dados_estradas = json.load(f)
        estradas_reais = gpd.GeoDataFrame.from_features(dados_estradas["features"])
        
    with open(ficheiro_ferrovias, 'r', encoding='utf-8') as f:
        dados_ferrovias = json.load(f)
        ferrovias_reais = gpd.GeoDataFrame.from_features(dados_ferrovias["features"])
else:
    print("Baixando dados do OpenStreetMap pela primeira vez (isso requer internet)...")
    estradas_reais = ox.features_from_bbox(norte, sul, leste, oeste, tags={'highway': ['motorway', 'trunk', 'primary', 'secondary']})
    ferrovias_reais = ox.features_from_bbox(norte, sul, leste, oeste, tags={'railway': 'rail'})
    
    print("Salvando dados em disco para uso futuro...")
    if not estradas_reais.empty:
        estradas_reais[['geometry']].to_file(ficheiro_estradas, driver='GeoJSON')
    if not ferrovias_reais.empty:
        ferrovias_reais[['geometry']].to_file(ficheiro_ferrovias, driver='GeoJSON')

# -------------------------------------------------------------
# 4. RASTERIZAÇÃO E CÁLCULO DE DISTÂNCIA
# -------------------------------------------------------------
print("Calculando o mapa de atração viária...")
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
# 5. O MOTOR DO TEMPO E GERAÇÃO DO GIF (ANIMAÇÃO)
# -------------------------------------------------------------
print("Iniciando a simulação temporal de expansão urbana...")

matriz_simulada = np.copy(matriz_simples)
fator_crescimento = 0.08 
anos_simulacao = 4

tamanho_filtro = 5
kernel = np.ones((tamanho_filtro, tamanho_filtro))
centro = tamanho_filtro // 2
kernel[centro, centro] = 0 

kernel_imediato = np.ones((3, 3))
kernel_imediato[1, 1] = 0

# A MUDANÇA DA ESTÉTICA: Cinza no lugar do Preto
cores_solo = ['gray', '#e31a1c', '#33a02c', '#fdbf6f', '#1f78b4']
cmap_personalizado = ListedColormap(cores_solo)

fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
ax_anim.axis('off')
frames_animacao = [] # Lista que vai guardar as "fotos" de cada ano

# Tira a foto do Ano Base (2010)
im = ax_anim.imshow(matriz_simulada, cmap=cmap_personalizado, animated=True, interpolation='none')
texto_titulo = ax_anim.text(0.5, 1.02, f"Simulação CA-Markov | Ano: 2010\nFator de Crescimento: {fator_crescimento} | Visão: {tamanho_filtro}x{tamanho_filtro}", 
                            transform=ax_anim.transAxes, ha="center", fontsize=14, fontweight='bold')
frames_animacao.append([im, texto_titulo])

# Roda o simulador
for ano in range(1, anos_simulacao + 1):
    matriz_urbana_atual = np.where(matriz_simulada == 1, 1, 0)
    
    densidade_atual = convolve(matriz_urbana_atual, kernel, mode='constant', cval=0)
    max_vizinhos = np.max(densidade_atual)
    peso_vizinhanca_atual = densidade_atual / max_vizinhos if max_vizinhos > 0 else densidade_atual
    
    mapa_aptidao_atual = (peso_vizinhanca_atual * 0.5) + (peso_autoestrada * 0.25) + (peso_ferrovia * 0.25)
    
    probabilidade_virar_cidade = mapa_aptidao_atual * fator_crescimento
    dados_aleatorios = np.random.rand(linhas, colunas)
    celulas_que_mudaram = probabilidade_virar_cidade > dados_aleatorios
    
    solo_permitido = (matriz_simulada == 2) | (matriz_simulada == 3) 
    toque_cidade = convolve(matriz_urbana_atual, kernel_imediato, mode='constant', cval=0)
    vizinhanca_forte = toque_cidade >= 1 
    
    mascara_transicao = celulas_que_mudaram & solo_permitido & vizinhanca_forte
    matriz_simulada[mascara_transicao] = 1
    
    # Tira a foto do novo ano e adiciona ao GIF
    im = ax_anim.imshow(matriz_simulada, cmap=cmap_personalizado, animated=True, interpolation='none')
    texto_titulo = ax_anim.text(0.5, 1.02, f"Simulação CA-Markov | Ano: 201{ano}\nFator de Crescimento: {fator_crescimento} | Visão: {tamanho_filtro}x{tamanho_filtro}", 
                                transform=ax_anim.transAxes, ha="center", fontsize=14, fontweight='bold')
    frames_animacao.append([im, texto_titulo])
    
    print(f"Ano 201{ano} concluído. Novas áreas urbanizadas: {np.sum(mascara_transicao)} pixels.")

# Renderizando o GIF e exibindo
print("Salvando o GIF da simulação (isso pode levar alguns segundos)...")
ani = animation.ArtistAnimation(fig_anim, frames_animacao, interval=1000, blit=False, repeat_delay=2000)
ani.save('evolucao_urbana.gif', writer='pillow')
print("GIF salvo com sucesso como 'evolucao_urbana.gif'!")

print(">>> ABRINDO ANIMAÇÃO. Feche a janela da imagem para continuar com a validação matemática! <<<")
plt.show(block=True) # Trava o código aqui até você fechar a janela

# -------------------------------------------------------------
# 6. VALIDAÇÃO MATEMÁTICA (Focada nos pixels de mudança real)
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

    mascara_vulneravel = (matriz_simples == 2) | (matriz_simples == 3)
    crescimento_real = mascara_vulneravel & (matriz_gabarito == 1)
    crescimento_simulado = mascara_vulneravel & (matriz_simulada == 1)
    acertos_na_mosca = crescimento_real & crescimento_simulado

    total_real = np.sum(crescimento_real)
    total_simulado = np.sum(crescimento_simulado)
    total_acertos = np.sum(acertos_na_mosca)

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
    print(f"Precisão: De tudo que o nosso modelo urbanizou, {taxa_acerto_modelo:.2f}% estava correto")
    print("-" * 50)
    
except FileNotFoundError:
    print("Arquivo de 2014 não encontrado. Pulando validação.")
    matriz_gabarito = np.zeros_like(matriz_simulada)

# -------------------------------------------------------------
# 7. VISUALIZAÇÃO GRÁFICA FINAL
# -------------------------------------------------------------
mapa_concordancia = np.zeros_like(matriz_simulada)
mapa_concordancia[(matriz_gabarito == 1) & (matriz_simulada == 1)] = 1 
mapa_concordancia[(matriz_gabarito != 1) & (matriz_simulada == 1)] = 2 
mapa_concordancia[(matriz_gabarito == 1) & (matriz_simulada != 1)] = 3 

# O cinza elegante sendo aplicado nos erros também
cores_erro = ['gray', '#2ca02c', '#1f77b4', '#d62728'] 
cmap_erro = ListedColormap(cores_erro)

fig, eixos = plt.subplots(1, 3, figsize=(18, 6))

eixos[0].imshow(matriz_simulada, cmap=cmap_personalizado, interpolation='none')
eixos[0].set_title('A Nossa Previsão (Simulado 2014)', fontsize=12)
eixos[0].axis('off')

eixos[1].imshow(matriz_gabarito, cmap=cmap_personalizado, interpolation='none')
eixos[1].set_title('Gabarito (Realidade 2014)', fontsize=12)
eixos[1].axis('off')

eixos[2].imshow(mapa_concordancia, cmap=cmap_erro, interpolation='none')
eixos[2].set_title('Concordância (Verde=Acerto, Azul/Vermelho=Erros)', fontsize=12)
eixos[2].axis('off')

plt.tight_layout()
plt.show()