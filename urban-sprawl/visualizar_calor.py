import os
import json
import rasterio
from rasterio import features
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import osmnx as ox
import geopandas as gpd

print("Iniciando o Gerador de Mapas de Calor de Nova Iguaçu...")

# -------------------------------------------------------------
# 1. CARREGANDO A GEOMETRIA BASE (TIF)
# -------------------------------------------------------------
caminho_arquivo = 'nova-iguacu-2010.tif'
with rasterio.open(caminho_arquivo) as src:
    transformacao = src.transform 
    linhas, colunas = src.shape
    # Pegamos as bordas para alinhar a imagem com o mapa vetorial (Lat/Lon)
    norte, sul, leste, oeste = src.bounds.top, src.bounds.bottom, src.bounds.right, src.bounds.left
    limites_extensao = [oeste, leste, sul, norte]

# -------------------------------------------------------------
# 2. BAIXANDO O CONTORNO OFICIAL DO MUNICÍPIO
# -------------------------------------------------------------
print("Baixando a fronteira oficial do município...")
# O OSMnx vai buscar o polígono exato de Nova Iguaçu no OpenStreetMap
contorno_ni = ox.geocode_to_gdf('Nova Iguaçu, Rio de Janeiro, Brazil')

# -------------------------------------------------------------
# 3. CARREGANDO O CACHE DE RUAS (Bypass Fiona)
# -------------------------------------------------------------
ficheiro_estradas = 'cache_estradas_ni.geojson'
ficheiro_ferrovias = 'cache_ferrovias_ni.geojson'

print("Carregando vias do cache local...")
with open(ficheiro_estradas, 'r', encoding='utf-8') as f:
    estradas_reais = gpd.GeoDataFrame.from_features(json.load(f)["features"])
    
with open(ficheiro_ferrovias, 'r', encoding='utf-8') as f:
    ferrovias_reais = gpd.GeoDataFrame.from_features(json.load(f)["features"])

# -------------------------------------------------------------
# 4. RECALCULANDO OS MAPAS DE CALOR (DISTÂNCIA)
# -------------------------------------------------------------
print("Calculando irradiação espacial (Mapa de Aptidão)...")
malha_autoestrada = np.ones((linhas, colunas))
malha_ferrovia = np.ones((linhas, colunas))

if not estradas_reais.empty:
    geom_estradas = [geom for geom in estradas_reais.geometry]
    malha_autoestrada = features.rasterize(geom_estradas, out_shape=(linhas, colunas), transform=transformacao, fill=1, default_value=0, all_touched=True)

if not ferrovias_reais.empty:
    geom_ferrovias = [geom for geom in ferrovias_reais.geometry]
    malha_ferrovia = features.rasterize(geom_ferrovias, out_shape=(linhas, colunas), transform=transformacao, fill=1, default_value=0, all_touched=True)

# Calcula a distância e inverte para virar "Atração" (1.0 = colado na via, 0.0 = longe)
dist_autoestrada = distance_transform_edt(malha_autoestrada)
peso_autoestrada = 1.0 - (dist_autoestrada / np.max(dist_autoestrada)) if np.max(dist_autoestrada) > 0 else np.zeros((linhas, colunas))

dist_ferrovia = distance_transform_edt(malha_ferrovia)
peso_ferrovia = 1.0 - (dist_ferrovia / np.max(dist_ferrovia)) if np.max(dist_ferrovia) > 0 else np.zeros((linhas, colunas))

# -------------------------------------------------------------
# 5. RENDERIZANDO A ARTE FINAL (GIS)
# -------------------------------------------------------------
print("Montando o mapa final...")
# Usamos um fundo escuro elegante (estilo painel de comando/GIS)
plt.style.use('dark_background')
fig, eixos = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Influência de Infraestrutura na Expansão Urbana - Nova Iguaçu', fontsize=20, fontweight='bold', color='white')

# --- MAPA 1: RODOVIAS ---
# 1. Plota o Mapa de Calor (extent garante que a imagem caia perfeitamente nas coordenadas geográficas)
im1 = eixos[0].imshow(peso_autoestrada, extent=limites_extensao, cmap='inferno', alpha=0.9)
# 2. Plota o contorno da cidade por cima
contorno_ni.boundary.plot(ax=eixos[0], color='#00ffff', linewidth=2, linestyle='--')
# 3. Plota o traçado exato das ruas
estradas_reais.plot(ax=eixos[0], color='white', linewidth=1.5)

eixos[0].set_title('Aptidão por Proximidade Rodoviária', fontsize=14)
eixos[0].axis('off')
plt.colorbar(im1, ax=eixos[0], shrink=0.7, label='Nível de Atração (0 a 1)')

# --- MAPA 2: FERROVIAS ---
im2 = eixos[1].imshow(peso_ferrovia, extent=limites_extensao, cmap='magma', alpha=0.9)
contorno_ni.boundary.plot(ax=eixos[1], color='#00ffff', linewidth=2, linestyle='--')
ferrovias_reais.plot(ax=eixos[1], color='white', linewidth=1.5)

eixos[1].set_title('Aptidão por Malha Ferroviária (Ramal Japeri)', fontsize=14)
eixos[1].axis('off')
plt.colorbar(im2, ax=eixos[1], shrink=0.7, label='Nível de Atração (0 a 1)')

plt.tight_layout()
plt.show()