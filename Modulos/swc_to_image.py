import numpy as np
import networkx as nx
import skimage.draw as draw

def swc_to_graph(filename):
    """Generates a networkx graph from a swc file."""
    
    fd = open(filename, 'r')
    for line in fd:
        if line[0]!='#':
            break

    graph = nx.Graph()
    for line in fd:
        line_s = line.strip()
        if len(line_s)>0:
            data  =line_s.split()
            node_id = int(data[0])
            seg_type = int(data[1])
            posz = int(float(data[2]))
            posx = int(float(data[3]))
            posy = int(float(data[4]))
            radius = float(data[5])
            parent = int(data[6])

            graph.add_node(node_id, type=seg_type, pos=(posz, posx, posy))
            if parent!=-1:
                graph.add_edge(parent, node_id, radius=radius)
                
    return graph

def create_image(graph, img_shape):
    """Generates an image from a graph."""

    img = np.zeros(img_shape, dtype=np.uint8)
    for node1, node2 in graph.edges:
        pos1 = graph.nodes[node1]['pos']
        pos2 = graph.nodes[node2]['pos']
        coords = draw.line_nd(pos1, pos2, endpoint=True)
        img[coords] = 255

    return img

def swc_to_image(filename, img_shape):
    """Creates an image using the data from a swc file."""

    graph = swc_to_graph(filename)
    img = create_image(graph, img_shape)

    return img

if __name__=='__main__':
    import matplotlib.pyplot as plt

    filename = '1.swc'
    img_shape = (45, 1104, 1376)
    img = swc_to_image(filename, img_shape)

    plt.figure(figsize=[15,15])
    plt.imshow(np.max(img, axis=0), 'gray')
    plt.show()