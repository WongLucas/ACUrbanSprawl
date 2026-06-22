import pygame
import numpy as np
from random import seed
from random import randint
import math
import matplotlib.pyplot as plt
import pandas as pd
def add_estatistica(step, CA_matriz_0, CA_Y, CA_X) -> np.array([]):
    densidade  = 0.0
    vivas      = 0.0
    mortas     = 0.0
    entropia   = 0.0
    total      = float(CA_Y * CA_X)

    for j in range(0, CA_Y):
        for i in range(0, CA_X):
            C = CA_matriz_0[j][i]
            if C == 1:
                vivas = vivas + 1
            else:
                mortas = mortas + 1
            #if C == 0:
            #    morta = morta + 1
            #else:
            #    viva = viva + 1

    vivas = vivas / total
    mortas = mortas / total
    densidade =  vivas
    entropia  = ((vivas * math.log2(vivas)) + (mortas * math.log2(mortas))) * -1.0
    vetor = np.array([step, densidade, entropia])

    return vetor



#condição de contorno circular
def funcao_transicao_circular(CA_matriz_0, CA_matriz_1, CA_Y, CA_X):
   for j in range(0, CA_Y):
        for i in range(0, CA_X):
            N = 0
            S = 0
            L = 0
            O = 0
            NO = 0
            NE = 0
            SO = 0
            SE = 0
            C = CA_matriz_0[j][i]
            
            im1 = i - 1
            iM1 = i + 1
            
            jm1 = j - 1
            jM1 = j + 1
            
            if im1 < 0:
                im1 = CA_X - 1

            if iM1 == CA_X:
                iM1 = 0

            if jm1 < 0:
                jm1 = CA_Y - 1

            if jM1 == CA_Y:
                jM1 = 0

            N = CA_matriz_0[jm1][i]
            NO = CA_matriz_0[jm1][im1]
            NE = CA_matriz_0[jm1][iM1]
            

            S = CA_matriz_0[jM1][i]
            SO = CA_matriz_0[jM1][im1]
            SE = CA_matriz_0[jM1][iM1]

            O = CA_matriz_0[j][im1]
            L = CA_matriz_0[j][iM1]
        
            
            soma = N + NO + NE + S + SO + SE + O + L                
            #regra
            if C == 0 and soma == 3:
                CA_matriz_1[j][i] = 1
            elif C == 1 and soma >= 2 and soma <= 3:
                CA_matriz_1[j][i] = 1
            else:
                CA_matriz_1[j][i] = 0



def condicao_inicial(CA_Y, CA_X, prob):
    CA_matriz_0 =  np.zeros((CA_Y, CA_X))
    CA_matriz_1 =  np.zeros((CA_Y, CA_X))
    rng = np.random.default_rng()
    for j in range(0, CA_Y):
        for i in range(0, CA_X):

            value = rng.random()
            #p = j * CA_X + i + 1
            #if p % 2 == 0:
            #    CA_matriz_0[j][i] = 1
            if value < prob:
                CA_matriz_0[j][i] = 1
            
    return CA_matriz_0, CA_matriz_1
    
'''
Função principal - primeira a ser chamada
'''
if __name__ == '__main__':
    width  = 1024
    height = 768
    FPS    = 60
    CA_X   = 150
    CA_Y   = 150
    max_time_step = 3000
    prob = 0.5
    plot_file_name = f"GOL-{CA_X}-{CA_Y}-{max_time_step}-{prob}.png"

    CA_matriz_0, CA_matriz_1 = condicao_inicial(CA_Y, CA_X, prob)
    vetor_estatistica = []

    pygame.init()
    gameDisplay = pygame.display.set_mode((width,height))
    pygame.display.set_caption('Autômato Celular')
    gameDisplay.fill((0, 0, 0))

    btn    = pygame.Rect(0, 0, 100, 30)
    rect1  = pygame.Rect(0, 30, 100, 100)
    clock  = pygame.time.Clock()

    running = True
    update  = False


    deltaX =  (width / CA_X)
    deltaY =  (height / CA_Y)
    tempo = 0
    estatistica = add_estatistica(tempo, CA_matriz_0, CA_Y, CA_X)
    vetor_estatistica.append(estatistica)
    while running:
        clock.tick(FPS)
        gameDisplay.fill((0, 0, 0))


        for j in range(0, CA_Y):
            for i in range(0, CA_X):
                s = CA_matriz_0[j][i]
                if s == 1:
                    x = i  * deltaX
                    y = j  * deltaY
                    pygame.draw.rect(gameDisplay, (255, 255, 255), (x + 1, y + 1, deltaX - 2, deltaY - 2), 0)

        #for j in range(1, CA_Y):
            #pygame.draw.line(gameDisplay, (255, 0, 0), (0, j * deltaY), (width, j * deltaY))

        #for i in range(1, CA_X):
            #pygame.draw.line(gameDisplay, (255, 0, 0), (i * deltaX, 0), (i * deltaX, height))
            #for i in range(0, CA_X - 1):

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                #if e.key == pygame.K_r:
                    #ca.random()
                if e.key == pygame.K_u:
                    update = not update
        if update:
            tempo = tempo + 1
            if tempo == max_time_step:
                running = False
            funcao_transicao_circular(CA_matriz_0, CA_matriz_1, CA_Y, CA_X)
            #funcao_transicao_constante(CA_matriz_0, CA_matriz_1, CA_Y, CA_X)
            aux = CA_matriz_1
            CA_matriz_1 = CA_matriz_0
            CA_matriz_0 = aux
            estatistica = add_estatistica(tempo, CA_matriz_0, CA_Y, CA_X)
            vetor_estatistica.append(estatistica)

            #if tempo % 10 == 0:
            #    imprimir_estatistica(CA_matriz_0, CA_Y, CA_X)
                
        pygame.display.flip()

    '''
        
            if e.type == pygame.MOUSEBUTTONDOWN:
                (mouseX, mouseY) = pygame.mouse.get_pos()
                if(btn.collidepoint((mouseX, mouseY))):
                    move_it = not move_it

            if e.type == pygame.QUIT:
                running = False
        #end event handling

        if move_it:
            rect1.move_ip(move_direction * 5, 0)
            if not gameDisplay.get_rect().contains(rect1):
                move_direction = move_direction * -1
                rect1.move_ip(move_direction * 5, 0)

        pygame.draw.rect(gameDisplay, (255, 0, 255), rect1, 1)
        pygame.draw.rect(gameDisplay, (255, 0, 0) if move_it else (0, 255, 255), btn)
    '''


    #end main loop
    pygame.quit()
    df = pd.DataFrame(vetor_estatistica, columns=['timestep', 'densidade', 'entropia'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(df['timestep'], df['entropia'], color='blue', marker='o')
    ax1.set_title(f'Game of Life - Entropia - Condição inicial: {prob}')
    ax1.set_xlabel('Instante de tempo')
    ax1.set_ylabel('Entropia')
    ax1.set_ylim(0, 1)
    ax1.grid(True)

    ax2.scatter(df['timestep'], df['densidade'], color='green', marker='o') # Mudei a cor para diferenciar
    ax2.set_title(f'Game of Life - Densidade - Condição inicial: {prob}')
    ax2.set_xlabel('Instante de tempo')
    ax2.set_ylabel('Densidade')
    ax2.set_ylim(0, 1)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plot_file_name, dpi=300)
    print(f"\t - Arquivo salvo: {plot_file_name}")
    plt.show()
    print("End the game")