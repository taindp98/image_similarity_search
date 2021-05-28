import matplotlib.pyplot as plt
from PIL import Image
import pickle

def visualize_group(list_imgs,step):
    list_imgs = sorted(list_imgs)
    fig=plt.figure(figsize=(20, 7))
    columns = len(list_imgs)
    rows = 1
    plt.xlabel('step {}'.format(step))
    for i in range(0, columns*rows):
        img = list_imgs[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(Image.open(img))
        plt.title(str('/'.join(list_imgs[i].split('/')[-2:])))
        
def export_pickle_file(data_dict,path):
    with open(path, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path, 'rb') as handle:
        valid = pickle.load(handle)
    if data_dict == valid:
        print('export pickle file done')