# Iniciação Científica

Este repositório contém minha iniciação científica em parceria com o Dr. Prof. Cesar Henrique Comin. Este repositório está dividido em diferentes notebooks python.

Atualmente estou no desenvolvimento para replicar a métologia aplicada no artigo [1], onde é demonstrado uma métrica de remoção de arestas terminais de estruturas tubulares que são consideradas espurias, ou ruidos de segmentação.

Os diferentes notebooks que estão aqui foram desenvolvidos com o tempo e contém alguns testes que foram feitos para conseguir chegar a replicação da métrica, atualmente os códigos que estão com o matérial mais recente é o notebook bulge_size_code2.ipynb e bulge_size_class.py, onde o .py está com a classe que estamos desenvolvendo e o .ipynb está os testes que estão sendo feitos com as imagens Vaso_amao.png, varia_raio.png e varia_comprimento.png.

Dependencias e versões:
* Numpy (1.23.5)
* Matplotlib (3.6.2)
* Scikit-learn 1.2.0
* Scipy (1.10.0)
* Jupyter-notebook (6.5.1)
* Scikit-image (0.19.3)
* Networkx (2.8.7)
* Igraph (0.7.1)

Referencias:
[1] Dominik Drees, Aaron Scherzinger, René Hägerling,Friedemann Kiefer, and Xiaoyi Jiang. Scalable robust graph and feature extraction for arbitrary vessel networks in large volumetric datasets. BMC bioinformatics,22(1):1–28, 2021.
