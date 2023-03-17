from io import BytesIO
from PIL import ImageDraw
from skimage.morphology import skeletonize, dilation, square
import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import app.utils.config as config
import app.utils.models as models

rng = np.random.default_rng()

class Node:

    def __init__(self, point2D):
        # self.point2D = point2D
        self.x = point2D[0]
        self.y = point2D[1]
        self.neighbors = None
        self.visited = False
        self.distance = -1
        self.dist_from_prev = 1.0

    
    def norm(self, n2):
        if isinstance(n2, Node):
            dis = ((self.x - n2.x)**2 + (self.y - n2.y)**2)**(0.5)
            return dis
        return f'{n2} must be a Node!'


    def __repr__(self) -> str:
        return f'Node({self.x}, {self.y})'
    


class ImageGraph:

    def __init__(self, image, r):
        self.image = image
        self.r = r
        self.skeleton = np.where(skeletonize(image), 1, 0)
        self.adj = self.populate_adj_dict()
        self.endnodes = {}
        self.root = self.get_root()
    

    def get_root(self):

        kernel = np.ones(shape=(2*self.r+1, 2*self.r+1))
        padded_img = np.pad(self.image, self.r, constant_values=0)
        gen1 = (_ for _ in np.argwhere(self.image))
        # gen2 = ((_, np.sum(np.multiply(self.image[_[0]-self.r:_[0]+self.r+1, _[1]-self.r:_[1]+self.r+1], kernel))) for _ in gen1)
        gen2 = ((_, np.sum(np.multiply(padded_img[_[0]:_[0]+2*self.r+1, _[1]:_[1]+2*self.r+1], kernel))) for _ in gen1)
        root_of_image = max(gen2, key=lambda x: x[1])

        gen3 = (_ for _ in np.argwhere(self.skeleton))

        root_skeleton = min(gen3, key=lambda x: np.linalg.norm(x-root_of_image[0]))

        return root_skeleton


    def populate_adj_dict(self):
        adj = {}
        padded_img = np.pad(self.skeleton, 1, constant_values=0)
        for vertex in (tuple(_) for _ in np.argwhere(self.skeleton)):
        # for vertex in self.vertices:
            adj[vertex] = Node(vertex)
            x = vertex[0]
            y = vertex[1]
            patch = padded_img[x:x+3, y:y+3].copy()
            patch[1, 1] = 0
            neighs = np.argwhere(patch) + np.array([x-1, y-1])
            adj[vertex].neighbors = sorted([tuple(n) for n in neighs], key=lambda x: np.linalg.norm(np.asarray(x)-np.asarray(vertex)))
        return adj
    


    def color_nodes(self, neigh_list, color):

        length = len(neigh_list)
            
        self.skeleton[neigh_list[0]] = color

        if length == 1:
            return

        elif length == 2:
            if self.adj[neigh_list[0]].dist_from_prev == 1.0:
                if any(not self.adj[neigh].visited for neigh in self.adj[neigh_list[0]].neighbors):
                    self.skeleton[neigh_list[1]] = rng.choice(range(2,256))
                    return
                else:
                    self.skeleton[neigh_list[1]] = color
                    return     
            self.skeleton[neigh_list[1]] = rng.choice(range(2,256))
            return

        elif length == 3:
            self.skeleton[neigh_list[2]] = rng.choice(range(2,256))
            if all(self.adj[neigh].dist_from_prev == 1 for neigh in neigh_list):
                self.skeleton[neigh_list[1]] = rng.choice(range(2,256))
                return
            if any(not self.adj[neigh].visited and np.linalg.norm(np.asarray(neigh_list[0]) - np.asarray(neigh)) >= 1.0 for neigh in self.adj[neigh_list[0]].neighbors):
                self.skeleton[neigh_list[1]] = color
                return
            else:
                self.skeleton[neigh_list[1]] = rng.choice(range(2,256))
            return
        
        else:
            self.skeleton[neigh_list[1]] = rng.choice(range(2,256))
            self.skeleton[neigh_list[2]] = rng.choice(range(2,256))
            self.skeleton[neigh_list[3]] = rng.choice(range(2,256))


    def get_veins(self):

        root = tuple(self.root)
        self.endnodes = {}
        # self.populate_adj_dict()

        self.adj[root].distance = 0
        queue = deque()
        queue.append(root)
        self.adj[root].visited = True
        self.skeleton[root] = rng.choice(range(2,256))

        while queue:
            
            front = queue.popleft()
            neigh_cnt = len(self.adj[front].neighbors)
            not_vstd_neighs = []
            # sorted_neighbors = sorted(self.adj[front].neighbors, key=lambda x: self.adj[front].norm(x))
            for neighbor in self.adj[front].neighbors:
                if not self.adj[neighbor].visited:
                    self.adj[neighbor].visited = True
                    self.adj[neighbor].distance = self.adj[front].distance + 1
                    self.adj[neighbor].dist_from_prev = np.linalg.norm(np.asarray(front)-np.asarray(neighbor))
                    not_vstd_neighs.append(neighbor)
                    queue.append(neighbor)
            if not_vstd_neighs:
                self.color_nodes(not_vstd_neighs, self.skeleton[front])
            elif neigh_cnt == 1:
                self.endnodes[front] = {
                    'color' : (int(self.skeleton[front]), rng.choice(256), rng.choice(256)),
                    'length' : self.adj[front].distance
                }
        self.skeleton = np.where(self.skeleton != 1, self.skeleton, 0)
        return self.skeleton


    # def get_lateral_veins(self, radius):

    #     x = self.root[0]
    #     y = self.root[1]
    #     patch = self.skeleton[x-radius:x+radius+1, y-radius:y+radius+1]
    #     patch[1:-1, 1:-1] = 0
    #     neighs = np.argwhere(patch) + np.array([x-radius, y-radius])

    #     for neigh in neighs:
    #         neigh = tuple(neigh)
    #         prev = min(self.adj[neigh].neighbors, key=lambda x: self.adj[x].distance)
    #         print(f'{neigh} -> {prev}')

    #         tang1 = tang(prev, neigh)
    #         endnode = min(self.endnodes.keys(), key=lambda x: np.abs(tang(x, neigh) - tang1))
    #         self.endnodes[endnode]['is_lateral'] = True 


def process_image(upload_file, img_size, service:str):

    resized = upload_file.resize(size=img_size)
    array = img_to_array(resized)
    if array.shape[-1] == 4:
        array = array[:,:,:-1]
    if service in ['black_rot', 'sem_seg']:
        array /= 255.
    elif service == 'pests':
        array = vgg16.preprocess_input(array)
    elif service == 'classification':
        array = mobilenet_v2.preprocess_input(array)

    expanded_array = tf.expand_dims(array, axis=0)

    return expanded_array


def get_phenotypic_chars(an_array, model):

    preds = model.predict(an_array)
    mask = np.argmax(preds, axis=-1)
    mask = np.squeeze(mask)
    blade_pixels = np.sum(mask==2)
    vein_pixels = np.sum(mask==1)
    mask = np.where(mask == 1,mask, 0).astype('uint8')
    leafGraph = ImageGraph(mask, 5)
    skeleton = leafGraph.get_veins()

    return skeleton, leafGraph.__dict__['endnodes'], blade_pixels, vein_pixels


def draw_veins(an_image, resize_factor, size=config.cfg.phenot.img_size):
    lateral = []
    start1 = 0
    start2 = 0
    img_array = process_image(an_image, size, service='sem_seg')
    sk, endnodes, blade_pixels, vein_pixels = get_phenotypic_chars(img_array, models.sem_seg_model)
    

    for i in range(int(size[0]/3), size[0], int(size[0]/3)):
        for j in range(int(size[0]/3), size[0], int(size[0]/3)):
            _ = {key:value for (key, value) in endnodes.items() if key[0] in range(start1, i) and key[1] in range(start2, j)}
            if _:
                lateral.append(max(_.items(), key=lambda x: x[1]['length']))
            else:
                lateral.append(None)
            start2 = j
        start2 = 0
        start1 = i
    _ = lateral.pop(1)
    _ = lateral.pop(3)


    colored_veins = np.zeros(shape=size+(3,))
    dilated = dilation(sk, footprint=square(3))

    for v in endnodes.values():
        ch_1 = np.where(dilated == v['color'][0], dilated, 0)
        ch_2 = np.where(dilated == v['color'][0], v['color'][1], 0)
        ch_3 = np.where(dilated == v['color'][0], v['color'][2], 0)
        stacked_mask = np.stack((ch_1, ch_2, ch_3), axis=2)
        colored_veins += stacked_mask

    colored_veins = array_to_img(colored_veins)
    final_img = ImageDraw.Draw(an_image)


    for item in endnodes.items():
        coords = item[0]
        distance = round(item[1]['length'] * resize_factor, 2)
        final_img.text(
            xy=(coords[1] - 5, coords[0] - 5),  # coords[0] -15
            text=f'{distance}cm',
            fill=(0, 0, 0),
            font=config.cfg.phenot.font
            )
    
    with BytesIO() as output:
        an_image.save(output, 'BMP')
        data1 = output.getvalue()

    with BytesIO() as output:
        colored_veins.save(output, 'BMP')
        data2 = output.getvalue()
    
    return data1, data2, blade_pixels, vein_pixels

