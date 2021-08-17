from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
class CustomImageDataset(Dataset):
    """
    usage:
        dataset = CustomImageDataset()
        train_loader = DataLoader(
                                dataset=dataset,
                                batch_size = bs,
                                shuffle = True,
                                num_workers = 2
                                )

        for epoch in range(num_epoch_train):
            for i,data in enumerate(train_loader,0):
                inputs, labels = data

                y_pred = model(inputs)

                loss = criterion(y_pred,labels)

        ref: input is a list of tuple of triplet batch

        return: 1 tuple triplet batch
    """

    def __init__(self,tups,transform=None):
        """
        instantiating the Dataset object
        initialize the directory containing the images

        """
        self.transform = transform
        self.tups = tups
    
    def __len__(self):
        """
        returns the number of samples in dataset.
        """
        return len(self.tups)
    
    def __getitem__(self,idx):
        """
        loads and returns a sample from the dataset at the given index idx
        transform in here
        """
        current_tuple_triplet = self.tups[idx]
        # if self.transform:
        # print('current_tuple_triplet',current_tuple_triplet)
        anchor = self.transform(Image.open(current_tuple_triplet[0]))
        pos = self.transform(Image.open(current_tuple_triplet[1]))
        neg = self.transform(Image.open(current_tuple_triplet[2]))

        trans_tup = tuple([anchor,pos,neg])
        return trans_tup
class FeedwardDataset(Dataset):
    """
    return
    """

    def __init__(self,products,transform=None):
        """
        instantiating the Dataset object
        initialize the directory containing the images

        """
        ## transform test
        self.transform = transform
        # self.tups = tups
        self.products = products
    
    def __len__(self):
        """
        returns the number of samples in dataset.
        """
        return len(self.products)
    
    def __getitem__(self,idx):
        """
        loads and returns a sample from the dataset at the given index idx
        transform in here
        """
        current_product = self.products[idx]
        # if self.transform:
        # print('current_tuple_triplet',current_tuple_triplet)
        tensor = self.transform(Image.open(current_product))

        tuple_product = tuple([idx,current_product,tensor])
        return tuple_product
