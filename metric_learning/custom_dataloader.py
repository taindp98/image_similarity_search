import pandas as pd
import random
from custom_dataset import CustomImageDataset,FeedwardDataset
from torch.utils.data import DataLoader
from custom_transforms import transform,transform_test
import torch
class TripletDataFactory():
    """
    anchor: random 1 product
    positive: random same product & same group w anchor
    negative: random different product or different group w anchor 
    """
    # def __init__(self,cluster_product_path,size):
    def __init__(self,df_full_cluster,size):
        # self.cluster_product_path = cluster_product_path
        self.df_full_cluster = df_full_cluster
        self.size = size
        self.triplet_tups = []

        random.seed(0)

        for i in range(self.size):
            # df_full_cluster = pd.read_csv(self.cluster_product_path)

            list_unique_product = list(set(list(df_full_cluster['product'])))


            ##  random choice anchor
            positive_id = random.choice(list_unique_product)
            df_current_positive_product = df_full_cluster[df_full_cluster['product']==positive_id]
            list_unique_positive_group = list(set(list(df_current_positive_product['group'])))
            
            positive_group = random.choice(list_unique_positive_group)

            if len(list_unique_positive_group) > 1:
                ## 2 options select negative
                option_neg = random.randint(0,1)
                if option_neg == 0:
                    ## get negative from group
                    negative_id = positive_id
                    negative_group =  sorted(set(list_unique_positive_group)-set([positive_group]))[0]
                else:
                    ## get negative from different product
                    negative_id = sorted(set(list_unique_product)-set([positive_id]))[0]
                    df_current_negative_product = df_full_cluster[df_full_cluster['product']==negative_id]
                    list_unique_negative_group = list(set(list(df_current_negative_product['group'])))
                    negative_group = random.choice(list_unique_negative_group)
            else:
                ## 1 option select negative
                negative_id = sorted(set(list_unique_product)-set([positive_id]))[0]
                df_current_negative_product = df_full_cluster[df_full_cluster['product']==negative_id]
                list_unique_negative_group = list(set(list(df_current_negative_product['group'])))
                negative_group = random.choice(list_unique_negative_group)
            
            df_positive_product = df_full_cluster[(df_full_cluster['product']==positive_id) & (df_full_cluster['group']==positive_group)]
            # print(negative_group,negative_id)
            df_negative_product = df_full_cluster[(df_full_cluster['product']==negative_id) & (df_full_cluster['group']==negative_group)]

            list_positive_product = list(df_positive_product['path'])
            list_negative_product = list(df_negative_product['path'])

            anchor = random.choice(list_positive_product)
            positive = random.choice(list_positive_product)
            negative = random.choice(list_negative_product)

            tup = tuple([anchor,positive,negative])
            self.triplet_tups.append(tup)
    
    def get_dataset(self):
        return CustomImageDataset(self.triplet_tups,transform)

    def get_dataloader(self):
        params = {
            'batch_size': 16,
            'shuffle' : False, 
            'num_workers' : 4,
        }
        dataset = self.get_dataset()
        data_loader = DataLoader(dataset, **params)
        return data_loader

class EncodeDataFactory():
    def __init__(self,cluster_product_path):
        """
        
        """
        self.cluster_product_path = cluster_product_path
        self.list_unique_product = pd.read_csv(self.cluster_product_path)
        self.products = list(self.list_unique_product['path'])

    def get_dataset(self):
        return FeedwardDataset(self.products,transform_test)

    def get_dataloader(self):
        params = {
            'batch_size': 16,
            'shuffle' : False, 
            'num_workers' : 4,
        }
        dataset = self.get_dataset()

        return DataLoader(dataset, **params)
        # list_id = []
        # list_path = []
        # list_tensor = []
        # for item in dataset:
            ## item is tuple _id,path,tensor
            # _id,path,tensor = item
            # list_id.append(_id)
            # list_path.append(path)
            # list_tensor.append(tensor)
        # tensors = torch.stack(list_tensor)
        # data_loader_tensors = DataLoader(tensors, **params)
        # data_loader_ids = DataLoader(list_id, **params)
        # data_loader_path = DataLoader(list_path, **params)
        # return data_loader_ids,data_loader_path,data_loader_tensors