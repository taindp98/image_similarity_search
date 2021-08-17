import matplotlib.pyplot as plt
from PIL import Image
def visualize_group(list_imgs):
    list_imgs = sorted(list_imgs)
    fig=plt.figure(figsize=(20, 20))
    columns = len(list_imgs)
    rows = 1
    for i in range(0, columns*rows):
        img = list_imgs[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(Image.open(img))
        plt.title(i)