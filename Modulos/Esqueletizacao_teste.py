#!/usr/bin/env python
# coding: utf-8

# # Bibliotecas que serao usadas

# In[1]:


#tempo de execucao total do programa todo 612.76912 segundos
#bibliotecas usadas
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology


#codigos do Cesar Comin
import segmentation as seg
import image as im
import skeleton

#modulos das funcoes usadas
import suavizacao
import Limiarizacao as lim


# # Importando a imagem

# In[21]:


img = plt.imread("amostra.tif")
plt.imshow(img, 'gray')

img_result = plt.imread("Comparacao.png")
img_result = im.Image(img_result)

#calculando o histograma da imagem amostra
plt.figure()
_ = plt.hist(img.flatten(), 255)
plt.xlim((0, 60))
print(img.dtype)


# In[46]:


sigma = 15
img_noise = np.random.normal(0, sigma, size = img.shape)


img_corrupt = img.astype(float) + img_noise
img_corrupt = 255 * (img_corrupt - img_corrupt.min())/(img_corrupt.max() - img_corrupt.min())
img_corrupt = img_corrupt.astype(np.uint8)
plt.figure(figsize = [10,10])
plt.imshow(img_corrupt, 'gray')


# # Processando o limear da imagem e binarizando-a

# In[3]:


#tempo de execucao: 47.0591 segundos
T = lim.threshold_otsu(img)
img_bin = lim.threshold_local(img, 5, 5, T)

plt.figure(figsize = [10,10])
plt.imshow(img, 'gray')


plt.figure(figsize = [10,10])
plt.imshow(img_bin, 'gray')


# # Suavizacao de Imagem

# In[3]:


#tempo de execucao: 372.45818 segundos
w = suavizacao.gaussian_filter_2d(20)
img_filtered = suavizacao.convolution(img, w)
plt.figure(figsize = [10,10])
plt.imshow(img_filtered, 'gray')


# In[5]:


#histograma da imagem suavizada
_ = plt.hist(img_filtered.flatten(), 50)
plt.xlim((0, 60))


# # Algoritmo de segmentacao do professor Cesar Comin

# In[32]:


#tempo de execucao: 0.383918 segundos

img_suave = seg.vessel_segmentation(im.Image(img_filtered), 0)
img_suave = img_suave.data

plt.figure(figsize = [10,10])
plt.title("IMAGEM SUAVIZADA")
plt.imshow(img_filtered.data, 'gray')

plt.figure(figsize = [10,10])
plt.title("IMAGEM SEGMENTADA")
plt.imshow(img_suave, 'gray')


# # Algoritmo de skeletizacao do Professor Cesar Comin

# In[33]:


#tempo de execucao: 147.6471 segundos

#pode-se usar a variavel img_result para fazer a analise da imagem segmentada pelo professor

img_t = skeleton.skeletonize(im.Image(img_suave))

plt.figure(figsize = [10,10])
plt.imshow(img_t.data, 'gray')


# # Esqueletizacao 1: algoritmo skeletonize da biblioteca skimage

# In[6]:


#tempo de execucao: 0.0557

img_skeleton = skimage.morphology.skeletonize(img_suave)

plt.figure(figsize = [10,10])
plt.imshow(img_suave, 'gray')
plt.title("Original")

plt.figure(figsize = [10,10])
plt.imshow(img_skeleton, 'gray')
plt.title("Esqueleto")


# # Esqueletizacao 2: Algoritmo Medial_axis da biblioteca skimage

# In[7]:


#tempo de execucao: 0.6443

teste = skimage.morphology.medial_axis(img_result.data)
teste2 = skimage.morphology.medial_axis(img_suave)

plt.figure(figsize = [10,10])
plt.imshow(teste, 'gray')
plt.title("Imagem segmentada pelo professor")

plt.figure(figsize = [10,10])
plt.imshow(teste2, 'gray')
plt.title("Imagem binarizada por mim")


# In[8]:


if (img_skeleton.data == teste.data):
    print("deu o mesmo: ", img_skelet)
else:
    print("Deu diferente")


# In[40]:



def merge_image(img1, img2):
    """esta funcao recebe a imagem original binaria (img1) e o esqueleto binario da mesma (img2), tranforma elas em 
    uma imagem de saida
"""
    img_saida = np.zeros((img1.shape[0], img1.shape[1], 3), dtype = np.uint8)
    
    imgA = img1 * 255 #nonbinary(img1)
    imgB = img2 * 255 #nonbinary(img2)
    
    img_saida[:,:,0] = imgA.copy()
    img_saida[:,:,1] = imgA.copy() 
    img_saida[:,:,2] = imgA.copy() 
    rows, cols = np.nonzero(img2)
    img_saida[rows,cols] = (255, 0, 0)
    
    return img_saida


# In[41]:


img_teste = merge_image(img_suave, teste2)

plt.figure(figsize = [10,10])
plt.imshow(img_teste)

#plt.imsave("esqueleto-Medial_axis.tiff", img_teste)


# In[42]:


img_teste2 = np.zeros((teste2.shape[0], teste2.shape[1], 3), dtype = np.uint8)
img_teste2[:,:,0] = nonbinary(teste2).copy()
img_teste2[:,:,1] = nonbinary(teste2).copy()
img_teste2[:,:,2] = nonbinary(teste2).copy()


#mostrando as imagem, o esqueleto mais forte doq o corpo
plt.figure(figsize = (10,10))
plt.imshow(nonbinary(img_suave))
plt.imshow(img_teste2, alpha=0.8)


# In[ ]:




