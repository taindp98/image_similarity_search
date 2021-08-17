import numpy as np
from torchvision import models, transforms
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image
import faiss
import multiprocessing as mp

import timeit

def recreate_image(codebook, labels, w, h):
    # d = codebook.shape[1]
    image = np.zeros((w, h,3))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
        # transforms.ToTensor()
        ]
    )

    img = transform(img)
    # print(img)
    return img

def percent_color(centroids,labels,white_color_idx):
    # print('centroids compute percent',centroids)
    list_percent = []
    dict_percent = {}
    for c in range(centroids.shape[0]):
        if c not in white_color_idx:
            count = labels.tolist().count(c)
            p = count/labels.shape[0]
            # list_percent.append(p)
            dict_percent[c] = p
    
    list_centroids = centroids.tolist()
    # print('list_centroids',list_centroids)
    # print('dict_percent',dict_percent)
    # dict_percent = {idx:item for idx, item in enumerate(list_percent)}
    dict_sort_percent = {k: v for k, v in sorted(dict_percent.items(), key=lambda item: item[1])}
    
    
    i = -1
    major_percent_idx = list(dict_sort_percent.keys())[i]
    while major_percent_idx in white_color_idx:
        i -= 1
        major_percent_idx = list(dict_sort_percent.keys())[i]
    # print("dict_sort_percent",dict_sort_percent)
    # print('centroids',centroids)
    major_color = list_centroids[major_percent_idx]
    # print('percent',list_percent)
    # print('='*50)
    # print('major_color',major_color)
    return list_percent,major_color

def check_possible_white_color(centroids,threshold_white = 0.9):
    list_centroids = centroids.tolist()
    list_possible_idx =  []
    for idx, centroid in enumerate(list_centroids):
        if centroid[0] >= threshold_white and centroid[1] >= threshold_white and centroid[2] >= threshold_white:
            list_possible_idx.append(idx)
    return list_possible_idx

def multi_single_color_reg(centroids,labels):
    """
    regconize multi-color or single-color
    """
    ## remove possible white color 
    # print('old centroids',centroids)
    # white_color_idx = np.argmax(centroids,axis=0)
    white_color_idx = check_possible_white_color(centroids)

    # print('white idx',white_color_idx)
    # centroids = np.delete(centroids, white_color_idx,axis=0)
    # print('new centroids',centroids)

    # percentage
    percent,_ = percent_color(centroids,labels,white_color_idx)
    list_idx_color_percent_gte_5 = []
    for idx, p in enumerate(percent):
        if p > 0.05:
            list_idx_color_percent_gte_5.append(idx)

    # print('percent',percent)
    # list_centroids = centroids.tolist()
    # print('list_idx_color_percent_gte_5',list_idx_color_percent_gte_5)
    # print('list_centroids',len(list_centroids))
    # pop item has percentage <= 5%
    # for idx in list_idx_color_percent_lte_5:
        # list_centroids.pop(idx)
    
    list_centroids = list(centroids[list_idx_color_percent_gte_5])

    list_d = []
    for c1 in list_centroids:
        sublist_d = []
        for c2 in list_centroids:
            d = distance.euclidean(c1, c2)
            sublist_d.append(d)
        list_d.append(sublist_d)

    array_d = np.array(list_d)
    triu_d = np.array(array_d)[np.triu_indices(array_d.shape[0])]

    # print('array_d',array_d)
    # print('shape',triu_d.shape)

    mean_triu_d = np.mean(triu_d)
    # sum_triu_d = sum(triu_d)
    # print('sum',sum_triu_d)
    # sum_triu_d_norm = 1/(1+np.exp(-sum_triu_d))
    return mean_triu_d,centroids
    
def show_color_range(centroids):
    # image = np.zeros((3,3,3))
    image = np.zeros((1,len(centroids),3))
    label_idx = 0
    # for i in range(3):
    for i in range(len(centroids)):
        image[0][i] = centroids[label_idx]
        label_idx += 1

    plt.imshow((image * 255).astype(np.uint8))

def get_major_color(centroids,labels):
    white_color_idx = check_possible_white_color(centroids)
    # centroids_new = np.delete(centroids, white_color_idx,axis=0)

    # print('white_idx',white_color_idx)

    # print('current_centroids',centroids_new)
    _,major_color = percent_color(centroids,labels,white_color_idx)
    
    return major_color



def compare_product_color(pair,tuple_result_multiprocess,threshold_color=0.25):
    """
    - remove possible white color
    - major percentage color after remove
    """
    max_iter = 100

    item_1 = pair[0].split('/')[-1].replace('.jpg','')
    item_2 = pair[1].split('/')[-1].replace('.jpg','')

    # labels_1,centroids_1 = get_centroids(item_1)
    # labels_2,centroids_2 = get_centroids(item_2)

    ## multiprocessing

    # pool = mp.Pool(mp.cpu_count())
    # tup_res_pool = pool.map(get_centroids_faiss, pair)
    labels_1,centroids_1 = tuple_result_multiprocess[item_1]
    labels_2,centroids_2 = tuple_result_multiprocess[item_2]


    # labels_1,centroids_1 = get_centroids_faiss(item_1,max_iter)
    # print(labels_1,centroids_1)
    # labels_2,centroids_2 = get_centroids_faiss(item_2,max_iter)
    # print(labels_2,centroids_2)
    major_color_1 = get_major_color(centroids_1,labels_1)
    major_color_2 = get_major_color(centroids_2,labels_2)

    # print(major_color_1,major_color_2)
    # show_color_range([major_color_1,major_color_2])

    dist_2_colors = distance.euclidean(major_color_1, major_color_2)
    # print('dist_2_colors',dist_2_colors)
    if dist_2_colors > threshold_color:
        ## create new dir for product
        return True
    else:
        return False

def get_centroids_faiss(img,max_iter=100):
    # print(mp.current_process())
    start = timeit.default_timer()
    n_colors = 10

    img_preprocess = np.array(preprocess(Image.open(img)))
    # print('='*50)
    # plt.imshow(img_preprocess)
    img_preprocess_norm = np.array(img_preprocess, dtype=np.float32) / 255
    # img_preprocess = cv2.imread(img)

    ## check color channel
    # d, w, h= original_shape = tuple(img_preprocess.shape)
    w, h, d= original_shape = tuple(img_preprocess_norm.shape)
    assert d == 3

    image_array = np.reshape(img_preprocess_norm, (w * h, d)) # a x d
    n_init = 10
    kmeans = faiss.Kmeans(d=image_array.shape[1], k=n_colors, niter=max_iter, nredo=n_init)
    kmeans.train(image_array)

    centroids = np.array(kmeans.centroids,dtype=np.float32)

    search = kmeans.index.search(image_array, 1)[1].tolist()
    labels = np.array([item[0] for item in search],dtype=np.int32)
    # print('labels',labels)
    # print('centroids',centroids)
    # labels = kmeans.labels_
    # labels = kmeans.predict(image_array)
    # centroids = kmeans.cluster_centers_
    end = timeit.default_timer()

    # print('time',end-start)
    return labels,centroids

def get_centroids(img):
    start = timeit.default_timer()

    n_colors = 10

    img_preprocess = np.array(preprocess(Image.open(img)))
    # print('='*50)
    # plt.imshow(img_preprocess)
    img_preprocess_norm = np.array(img_preprocess, dtype=np.float64) / 255
    # img_preprocess = cv2.imread(img)

    ## check color channel
    # d, w, h= original_shape = tuple(img_preprocess.shape)
    w, h, d= original_shape = tuple(img_preprocess_norm.shape)
    assert d == 3

    image_array = np.reshape(img_preprocess_norm, (w * h, d)) # a x d

    # 10 x d
    # image_array_shuffle = shuffle(image_array, random_state=0)
    kmeans = KMeans(n_clusters = n_colors ,random_state=0).fit(image_array)
    # labels = kmeans.labels_
    labels = kmeans.predict(image_array)
    centroids = kmeans.cluster_centers_
    end = timeit.default_timer()

    # print('time',end-start)
    return labels,centroids

def multi_process_product(product):
    """
    input: list products
    return: list centroids
    """
    product = sorted(product)
    track_key = [item.split('/')[-1].replace('.jpg','') for item in product]
#     pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(2)
    tup_res_pool = pool.map(get_centroids_faiss, product)

    dict_res_pool = dict(zip(track_key,tup_res_pool))
    return dict_res_pool

def check_sublist(lst1, lst2):
    ls1 = [element for element in lst1 if element in lst2]
    ls2 = [element for element in lst2 if element in lst1]
    
    if ls1 == ls2 and ls1:
        return True

def remove_existing(list_pattern):
    """
    0) remove empty sublist
    1) find element > 1 in flatten list
    2) find sublist longer
    """
    list_flatten = list(sum(list_pattern, []))
    
    unique_elements = list(set(list_flatten))
                        
    list_final_idx = []
    
    for item in list_pattern:
        if len(item) == 0:
            list_pattern.remove(item)
#     print('list_flatten',list_flatten,list_pattern)
    for ele in unique_elements:
        count_ele = list_flatten.count(ele)
#         print(count_ele)
        if count_ele > 1:
            max_len = 0
            true_idx = None
            for idx, l1 in enumerate(list_pattern):
#                 print(ele)
                if ele in l1 and len(l1) > max_len:
                    max_len = len(l1)
                    true_idx = idx
            if true_idx not in list_final_idx:
                
                list_final_idx.append(true_idx)
        else:
            for idx, l2 in enumerate(list_pattern):
                if ele in l2 and idx not in list_final_idx:
                    list_final_idx.append(idx)
    list_rm_dup = [list_pattern[idx] for idx in list_final_idx]
    return list_rm_dup