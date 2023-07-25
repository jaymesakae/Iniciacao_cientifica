#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np

def threshold_otsu(img):
    '''Calcula o limiar de Otsu utilizando o histograma da imagem'''
    
    bins = range(0, 257)
    hist, _ = np.histogram(img, bins)
    
    num_pixels = img.shape[0]*img.shape[1] 
    sum_img = np.sum(img)
    m_G = sum_img/num_pixels
    max_sigma_I = -1
    
    sum_back = 0
    num_back = 0
    for threshold in range(0, 256):
        num_back = num_back + hist[threshold]  # Número de pixels com valor menor que threshold
        sum_back = sum_back + threshold*hist[threshold]  # Soma dos valores de pixel background

        num_fore = num_pixels - num_back
        sum_fore = sum_img - sum_back
        
        if num_back == 0 or num_fore == 0:
            continue
        
        P_back = num_back/num_pixels
        P_fore = num_fore/num_pixels
        m_back = sum_back/num_back   
        m_fore = sum_fore/num_fore    
        
        sigma_I = P_back*(m_back-m_G)**2 + P_fore*(m_fore-m_G)**2
        
        if sigma_I > max_sigma_I:
            max_sigma_I = sigma_I
            best_threshold = threshold
    return best_threshold

def threshold_local(img, radius, threshold, threshold_global=0):
    '''Aplica limiarização local em uma imagem. 'radius' define o tamanho da vizinhança
       que será considerada. Pixels possuindo valor maior que a média da vizinhança
       mais 'threshold' são considerados foreground. Opcionalmente, podemos considerar
       uma condição adicional de que o pixel pode ser foreground apenas se o seu valor
       na imagem for maior que 'threshold_global' '''     
        
    num_rows, num_cols = img.shape
    img_bin = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            # Limites da vizinhança do pixel, tomando cuidado para
            # não ultrapassar as bordas da imagem
            first_row = max([row-radius, 0])
            first_col = max([col-radius, 0])
            last_row = min([row+radius, num_rows])
            last_col = min([col+radius, num_cols])
                
            # Obtém vizinhança do pixel (row, col)
            img_patch = img[first_row:last_row, first_col:last_col]
            
            mean_intensity_patch = np.mean(img_patch)
            img_corr = img[row, col] - mean_intensity_patch
            if img_corr>=threshold and img[row, col]>=threshold_global:
                img_bin[row, col] = 1
    
    return img_bin

