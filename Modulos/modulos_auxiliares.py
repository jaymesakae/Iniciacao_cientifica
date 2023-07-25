#bibliotecas usadas
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu 
from scipy.ndimage import distance_transform_edt as distance

import random

#codigos do Cesar Comin
import Modulos.segmentation as seg
import Modulos.image as im
import Modulos.skeleton as skn
import Modulos.util as util

#modulos das funcoes usadas
import Modulos.Limiarizacao as lim

#Modulo de funções de teste
import Modulos.cldice as clDice

## Graph module
import Modulos.creation as creation
import networkx as nx
import Modulos.adjustment as ad


def merge_image(img1, img2, color = ([255, 0, 0], [0, 255, 0])):
    """
    Function to plot two images in one
    
    Input:
        img1: [int] background image or image with noise
        img2: [int] foreground image or image original
        
    Output:
        img_out: [int] merging of the two images, being green of image 1 and red of image 2
    """
    
    img_saida = np.zeros((img1.shape[0], img1.shape[1], 3), dtype = (np.uint8))
    
    #get range of img1
    Range = im.Image.get_range( im.Image(img1) )
    
    #transform img1 in non binary image
    if Range == (0, 1):
        imga = img1 * 255
    else:
        imga = img1

    #get range of img2
    Range = im.Image.get_range(im.Image(img2))
    
    #transform img2 in non binary image
    if Range == (0, 1):
        imgb = img2 * 255
    
    else:
        imgb = img2

        
    imga = transform_rgb_to_gray(imga)
    
    imgb = transform_rgb_to_gray(imgb)
    
    
    rows, cols = np.nonzero(imga)
    img_saida[rows, cols] = color[0]

    rows, cols = np.nonzero(imgb)
    img_saida[rows, cols] = color[1]
    
    rows, cols = np.nonzero(imga & imgb)
    img_saida[rows, cols] = (255, 255, 255)
    
    return img_saida

def merge_modified(img1, img2, color = ([255, 0, 0], [0, 0, 255]), background = None):
    """
    This function can be substitue below function
    Function to plot two images in one
    
    Input:
        img1: [int] background image or image with noise
        img2: [int] foreground image or image original
        color: [tuple of list] list 1 is range of color rgb to img1 and list 2 to img2
        background: [image] segmentated image to put in background
        
    Output:
        img_out: [int] merging of the two images, being green of image 1 and red of image 2
    """
    
    img_saida = np.zeros((img1.shape[0], img2.shape[1], 3), dtype = np.uint8)
    
    #get range of img1
    Range = im.Image.get_range( im.Image(img1) )
    
    #transform img1 in non binary image
    if Range == (0, 1):
        imga = img1 * 255
    else:
        imga = img1

    #get range of img2
    Range = im.Image.get_range( im.Image(img2) )
    
    #transform img2 in non binary image
    if Range == (0, 1):
        imgb = img2 * 255
    
    else:
        imgb = img2
        
    if background is not None:
        if im.Image.get_range( im.Image(background) ) == (0, 1):
            background = background * 255
        background = transform_rgb_to_gray(background)
        rows, cols = np.nonzero(background)
        img_saida[rows, cols] = (150,150,150)

    imga = transform_rgb_to_gray(imga)
    
    imgb = transform_rgb_to_gray(imgb)
    
    rows, cols = np.nonzero(imga)
    img_saida[rows, cols] = color[0]

    rows, cols = np.nonzero(imgb)
    img_saida[rows, cols] = color[1]
    
    for row in range(img1.shape[0]):
        for col in range(img1.shape[1]):
            if img1[row, col] == img2[row, col] and img1[row, col] == True:
                img_saida[row, col] = (255, 255, 0)
                
    return img_saida
    


def transform_rgb_to_gray(img):
    """
    Function to transform RGB image in a image in shades of gray
    
    Input:
        img: [int] RGB image
        
    Output:
        img_out: [int] image in shades of gray
    
    """
    img_out = np.zeros((img.shape[0], img.shape[1], 2), dtype = np.uint8)
    if len(img.shape) == 3:
        for i in range(len(img.shape)):
            img_out = img[:, :, i]
    
    else:
        return img
    return img_out


def transform_nonbinary(img):
    """
    Function to transform a image with 255 value in a binary image
    
    Input:
        img: should be RGB image or image in shades of gray
        
    Output:
        img_out: [bool with 2 channels] Binary image
        or
        img_out: [bool with 3 channels] Binary image
    """
    
    # if the image have 3 channels
    img_out = transform_rgb_to_gray(img)
    rows, cols = np.nonzero(img_out)
    img_out[:,:] = False
    img_out[rows, cols] = True

    return img_out



def mostra_img(img, nome = None):
    """
    Function for display a image list
    
    Input:
        img: [list] A image list for plot
    
    Output:
        nothing
    """
    for mostra in range(len(img)):
        plt.figure(figsize = [10,10])
        if nome is not None:
            plt.title(nome[mostra])
        plt.imshow(img[mostra], 'gray')
    return


def plot_graph(values, label, ylabel, title, Range, color, save = False):
    """
    Function to show a graph of IOU, TPR and PPV operations
    
    Input:
        values: [List of list] contain values to plot (iou, tpr, etc)
        label: [list] name of each list above
        ylabel: [string] y-axis name
        title: [string] title of graph
        Range: [list] list with noise value
        size: [tuple] tuple with size value to plot graph
    
    Output:
        Nothing
    """

    if len(values) == len(label):
        plt.title(title)
        for i in range(len(label)):
            plt.plot(Range, values[i], color[i], label = label[i])
        plt.xlabel("noise")
        plt.ylabel(ylabel)
        plt.legend()
        if save == True:
            plt.savefig(title+'.png')
        plt.show()
        
        return
    else:
        print("tamanhos errados")
        return
    
        
## Comparation methods
def confusion_matrix(img, img_skeleton):
    """
    This function returns the iou (intersection over union), TPR (True positive rate) and PPV (Precision) values
    from a noisy skeleton
    (tp) true positive = pixel identifield in the original skeleton and in the noisy skeleton
    (fp) false positive = pixel identifield in the noisy skeleton but not identifield in the skeleton original
    (fn) false negative = pixel identifield in skeleton original but not identifield in skeleton noisy
    input:
        img: [bool] image with skeleton original
        img_skeleton: [bool] image with noise skeleton
    outpút:
        The IOU calculation TP/(TP + FP + FN)
        The TPR calculation TP/(TP + FN)
        The PPV calculation TP/(TP + FP)
    """
    num_row, num_col = img.shape[0], img.shape[1]
    tp = 0 #true positive
    fp = 0 #false positive
    fn = 0 #false negative
    for row in range(num_row):
        for col in range(num_col):
            if img[row, col] == True and img_skeleton[row, col] == True:
                tp += 1
            elif img[row, col] == False and img_skeleton[row, col] == True:
                fp += 1
            elif img[row, col] == True and img_skeleton[row, col] == False:
                fn += 1
    
    iou = tp/(tp + fp + fn)
    tpr = tp/(tp + fn)
    ppv = tp/(tp + fp)
    return iou, tpr, ppv 

def confusion_matrix_new(img_skel_ref, img_skel_alg, sigma = 1):
    """
    This function returns the iou (intersection over union), TPR (True positive rate or sensibility) 
    and PPV (Precision) values from a noisy skeleton
    (tp) true positive = pixel identifield in the original skeleton and in the noisy skeleton
    (fp) false positive = pixel identifield in the noisy skeleton but not identifield in the skeleton original
    (fn) false negative = pixel identifield in skeleton original but not identifield in skeleton noisy
    input:
        img: [bool] image with skeleton original
        img_skeleton: [bool] image with noise skeleton
        sigma: [int] threshold for Elclidean distance calculate
    output:
        The IOU calculation TP/(TP + FP + FN)
        The TPR calculation TP/(TP + FN)
        The PPV calculation TP/(TP + FP)    
    """
    img_skel_ref_dil = (distance(1 - img_skel_ref) <= sigma).astype(np.uint8)
    img_skel_alg_dil = (distance(1 - img_skel_alg) <= sigma).astype(np.uint8)
    
    # Sensibility calculate
    tp_tpr = np.sum((img_skel_ref) & (img_skel_alg_dil))
    fn_tpr = np.sum((img_skel_ref) & (img_skel_alg_dil == False))
    
    tpr = tp_tpr/(tp_tpr+fn_tpr)
    
    # Precision calculate
    tp_ppv = np.sum((img_skel_ref_dil) & (img_skel_alg))
    fp_ppv = np.sum((img_skel_ref_dil == False) & (img_skel_alg))
    
    ppv = tp_ppv/(tp_ppv+fp_ppv)
    
    
    
    #print(tp_iou, fp_iou, fn_iou)
    tp_iou = (tp_tpr + tp_ppv)/2
    iou = tp_iou/(tp_iou + fp_ppv + fn_tpr)
    
    #print(tp_tpr-tp_ppv)
    
    #error = (tp_tpr-tp_ppv)/tp_iou
    print('tp precision: ', tp_ppv, 'tp sensibility: ',tp_tpr, 
          '\nfp precision: ', fp_ppv, 'fn sensibility: ', fn_tpr, '\n')
          #'error tp: ', error)
    
    return iou, tpr, ppv


def compare_skeleton(skel_original, segmentation, noise_aplication, noise_treatment, method = 'palagyi'):
    iou = []
    tpr = []
    ppv = []
    
    iou_graph = []
    tpr_graph = []
    ppv_graph = []
    
    
    for noise in noise_aplication:
        img_noise = random_noise(segmentation, noise)
        img_noise = 1 - util.remove_small_comp(1 - img_noise, 2) # será q o valor default da conta?
        img_noise = util.remove_small_comp(img_noise)
        
        if method == 'palagyi':
            skel_algo = (skn.skeletonize( im.Image(img_noise) )).data
            
        elif method == 'zhang':
            skel_algo = skimage.morphology.skeletonize(img_noise, method='zhang')
            
        elif method == 'lee':
            skel_algo = skimage.morphology.skeletonize(img_noise, method='lee')
            
        elif method == 'medial_axis':
            skel_algo = skimage.morphology.medial_axis(img_noise)
        else:
            print('Error, method not found')
            return
        
        #remove pad
        skel_algo = skel_algo[10:266,10:266]
        
        # quality values
        iou_tpr_ppv = confusion_matrix_new(skel_original, skel_algo)
        iou.append(iou_tpr_ppv[0])
        tpr.append(iou_tpr_ppv[1])
        ppv.append(iou_tpr_ppv[2])
        
        #plt.figure()
        #plt.subplot(121)
        #plt.title('skeleton ori')
        #plt.imshow(skel_original)
        #plt.subplot(122)
        #plt.title('skeleton algo')
        # plt.imshow(skel_algo)
        
        # create graph for algorithm skeleton
        graph_skel_algo = creation.create_graph(im.Image(skel_algo))
        
        aux_iou_graph = []
        aux_tpr_graph = []
        aux_ppv_graph = []
        
        for remove_noise in noise_treatment:
            graph_aux = ad.adjust_graph(graph_skel_algo, remove_noise)
            
            #transform the graph in image
            graph_aux = util.graph_to_img(graph_aux)
            graph_aux = transform_nonbinary(graph_aux)
            
            # calculate iou, tpr and ppv
            iou_tpr_ppv = confusion_matrix_new(skel_original, graph_aux)
            aux_iou_graph.append(iou_tpr_ppv[0])
            aux_tpr_graph.append(iou_tpr_ppv[1])
            aux_ppv_graph.append(iou_tpr_ppv[2])
            
        # store iou, tpr and ppv values
        iou_graph.append(aux_iou_graph)
        tpr_graph.append(aux_tpr_graph)
        ppv_graph.append(aux_ppv_graph)
        
        
        print('\n******** NOISE:', noise, ', METHOD:', method, '********')
        print('iou: ', iou[-1], 'sensibility', tpr[-1], '\nprecision:', ppv[-1], '\n\n')
        
    return [(iou, tpr, ppv), (iou_graph, tpr_graph, ppv_graph)]



## Apply noise in the image
def random_noise(img, p):
    """
    Function to aplly noise based on probability
    input: 
    img: [bool] Binary image
    p: [float] Range for probability
    
    output:
    img_out: [bool] Image binary with noise
    """
    img_out = np.zeros(img.shape, dtype = 'uint8')
    num_rows, num_cols = img.shape
    for row in range(num_rows):
        for col in range(num_cols):
            img_out[row, col] = img[row, col]
            if img[row,col] == False: 
                chance = random.random()
                if chance < p:
                    img_out[row,col] = True
    return img_out

## Function for calculate clDice value

## analise this function (peculiar behavior) and write a best description
def clDice_adaptative_2D(v_p, v_1, v_p_skeleton, v_1_skeleton):
    """
    this function compute the cldice metric
    
    args: 
        v_p: predicted image
        v_1: ground thurh image
        
    returns:
        float: cldice metric
    """
    tprec = clDice.cl_score(v_p, v_1_skeleton)
    tsens = clDice.cl_score(v_1, v_p_skeleton)
    
    return 2*tprec*tsens/(tprec+tsens)


## Function for generate and plot graph (in development)
def generate_plot_graph(skeleton, title):
    """
    This function generate and plot a networkx graph to image
    
    Input: 
        skeleton: [bool] skeleton image ## THIS IS A LIST 
        title: [string] name of graph
        
    Output:
        graph: [networkx.MultiGraph] skeleton graph
    """
    # Gerando o grafo do esqueleto
    graph = []
    positions_ = []

    for count in range(len(skeleton)):
        graph.append( creation.create_graph( im.Image(skeleton[count] ) ) )

        #definindo as posições do grafo em imagem
        positions = graph[-1].nodes(data = 'center')
        positions_.append( {k:v[::-1] for k, v in dict(positions).items()} )

        plt.figure(figsize = [20,20])
        _ = title + " " + str(count)
        plt.title(_)
        plt.imshow(skeleton[count], cmap = "gray" )

        # Plotando o grafo
        nx.draw_networkx_edges(graph[count], pos = positions_[count], edge_color = 'red')
        nx.draw_networkx_nodes(graph[count], pos = positions_[count], node_size = 10)
    return graph

def plot_network_graph(graph, title):
    """
    This function plot a list of networkx graph
    
    Input:
        graph: [list] a list with skeleton binary image graphs
        title: [string] title of graph
    
    Output:
        nothing
    """
    positions_ = []

    for count in range(len(graph)):
        
        #definindo as posições do grafo em imagem
        positions = graph[count].nodes(data = 'center')
        positions_.append( {k:v[::-1] for k, v in dict(positions).items()} )

        plt.figure(figsize = [20,20])
        _ = title + " " + str(count)
        plt.title(_)
        plt.imshow(graph[count], cmap = "gray" )

        # Plotando o grafo
        nx.draw_networkx_edges(graph[count], pos = positions_[count], edge_color = 'red')
        nx.draw_networkx_nodes(graph[count], pos = positions_[count], node_size = 10)
    return
