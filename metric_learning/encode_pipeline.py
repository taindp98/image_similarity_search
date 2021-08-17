from custom_dataloader import EncodeDataFactory
from model_class import RecommendNet
from triplet_loss import TripletLoss
import torch
from datetime import datetime,date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
from tqdm import tqdm
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
today_time = date.today().strftime("%m_%d_%Y") + '_' + datetime.now().strftime("%H_%M")

size_triplet = 10
cluster_product_path = '../../resource/full_cluster_product.csv'

def feedward_per_ite(dataloader, idx, model):
    """
    encoding
    """
    # for step, vector in enumerate(dataloader):
        ## put tensor to device
    # vector = tensor.to(device)
    _id,path,tensor = dataloader

    vector = tensor.to(device)
    vector_enc = model(vector)

    # print('_id',_id)
    # print('='*50)
    # print('vector_enc',vector_enc[0].shape)
    # for _id,path,tensor in dataloader:
    

    file_id = os.path.join('../../encode',str(idx)+'_id_'+'.h5')
    file_path = os.path.join('../../encode',str(idx)+'_path_'+'.h5')
    file_vector = os.path.join('../../encode',str(idx)+'_vector_'+'.h5')
    
    torch.save(_id,file_id)
    torch.save(path,file_path)
    torch.save(vector_enc,file_vector)

def feedward(model):

    print('Encoding on: ', device)
    model.cuda()
    model.eval()

    encode_data = EncodeDataFactory(cluster_product_path)
    dataloaders = encode_data.get_dataloader()
    
    for idx,item in tqdm(enumerate(dataloaders),total=len(dataloaders)):

        feedward_per_ite(item, idx, model)
