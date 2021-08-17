from custom_dataloader import TripletDataFactory
from model_class import RecommendNet
from triplet_loss import TripletLoss
import torch
from datetime import datetime,date
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
today_time = date.today().strftime("%m_%d_%Y") + '_' + datetime.now().strftime("%H_%M")

size_triplet = 10
cluster_product_path = '../../resource/full_cluster_product.csv'

def train_step(dataloader, model, loss,opt):
    """
    train 1 epoch
    """
    mean_loss = 0
    d_ap = []
    d_an = []
    for step, (anchor,pos,neg) in enumerate(dataloader):
        ## put tensor to device
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor_enc = model(anchor)
        pos_enc = model(pos)
        neg_enc = model(neg)

        triplet_loss, ap, an = loss(anchor_enc,pos_enc,neg_enc,average_loss=True)

        d_ap.append(ap.cpu().detach().numpy())
        d_an.append(an.cpu().detach().numpy())
        mean_loss += triplet_loss

        ## backpropagation

        triplet_loss.backward()

        opt.step()

    mean_loss = mean_loss/len(dataloader)
    cur_lr = opt.param_groups[0]['lr']

    
    ax = plt.figure(figsize=(10,6))
    ax = sns.distplot(np.array(d_ap), label='positive_score')
    ax = sns.distplot(np.array(d_an), label='negative_score')
    ax.legend(labels=['positive_score','negative_score'])
    ax.set_xlim(0, 10)
    plt.show()
    print('='*20+'Training'+'='*20)
    print(f'triplet_loss: {mean_loss.cpu().detach().numpy():.4f} d_A_P: {(np.mean(d_ap)):.4f} d_A_N: {(np.mean(d_an)):.4f} current_lr: {cur_lr: .5f}')    
    print("=====================================")
    return mean_loss, d_ap, d_an

def val_step(dataloader, model, loss):
    """
    validate 1 epoch
    """
    mean_loss = 0
    d_ap = []
    d_an = []
    with torch.no_grad():
        for step, (anchor,pos,neg) in enumerate(dataloader):
            ## put tensor to device
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchor_enc = model(anchor)
            pos_enc = model(pos)
            neg_enc = model(neg)

            triplet_loss, ap, an = loss(anchor_enc,pos_enc,neg_enc,average_loss=True)

            d_ap.append(ap.cpu().detach().numpy())
            d_an.append(an.cpu().detach().numpy())
            mean_loss += triplet_loss

            # triplet_loss.backward()

        

    mean_loss = mean_loss/len(dataloader)

    
    ax = plt.figure(figsize=(10,6))
    ax = sns.distplot(np.array(d_ap), label='positive_score')
    ax = sns.distplot(np.array(d_an), label='negative_score')
    ax.legend(labels=['positive_score','negative_score'])
    ax.set_xlim(0, 10)
    plt.show()
    print('='*20+'Validation'+'='*20)
    print(f'triplet_loss: {mean_loss.cpu().detach().numpy():.4f} d_A_P: {(np.mean(d_ap)):.4f} d_A_N: {(np.mean(d_an)):.4f} ')    
    print("=====================================")
    return mean_loss, d_ap, d_an

def train(model, epochs, loss_fn, optimize, scheduler, seed, verbose=True):

    losses = []
    d_anchor_pos = []
    d_anchor_neg = []
    torch.manual_seed(seed)
    # torch.cuda.current_device()
    print('Training on: ', device)
    if str(device) != 'cpu':
        model.cuda()
    model.train()

    df_product = pd.read_csv(cluster_product_path)
    unique_prod = list(set(list(df_product['product'])))
    train_prod,test_prod = train_test_split(unique_prod,test_size=0.25,random_state=0, shuffle=True)
    df_product_train = df_product.loc[df_product['product'].isin(train_prod)]
    df_product_train.to_csv('../../resource/df_train.csv',header=True,index=False)
    df_product_val = df_product.loc[df_product['product'].isin(test_prod)]
    df_product_val.to_csv('../../resource/df_val.csv',header=True,index=False)

    for epoch in range(epochs+1):
        print(f"Epoch {epoch+1}")
        # idxs_a, idxs_p, idxs_n = dataset.get_triplets_batch(size_triplet)
        triplet_data_train = TripletDataFactory(df_product_train,size_triplet)
        dataloader_train = triplet_data_train.get_dataloader()

        triplet_data_val = TripletDataFactory(df_product_val,size_triplet)
        dataloader_val = triplet_data_val.get_dataloader()
        
        scheduler.step()
        optimize.zero_grad()
        
        mean_loss, d_ap, d_an = train_step(dataloader_train, model, loss_fn,optimize)
        
        mean_loss_val, d_ap_val, d_an_val = val_step(dataloader_val, model, loss_fn)
        
        d_ap = np.mean(d_ap)
        d_an = np.mean(d_an)
        losses.append(mean_loss)
        d_anchor_pos.append(d_ap)
        d_anchor_neg.append(d_an)
        
        
        if losses[-1] <= min(losses):
            best_model= copy.deepcopy(model)
            best_scheduler = copy.deepcopy(scheduler)
            best_epoch= epoch + 1
            best_accuracy = min(losses)
            best_optimizer = copy.deepcopy(optimize)
            
            checkpoint = {
            'model': best_model,
            'epoch':epoch+1,
            'model_state_dict':best_model.state_dict(),
            'optimizer_state_dict':best_optimizer.state_dict(),
            'scheduler_state_dict':best_scheduler.state_dict()
            }
                
            path_checkpoint =  '../../model/recommend_checkpoint' + today_time + '.pth'
            torch.save(checkpoint, path_checkpoint)
    
    return model