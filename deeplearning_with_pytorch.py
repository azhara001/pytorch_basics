import torch 
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset

print('pytorch modules imported!')

import pandas as pd

# importing fashion mnist dataset from torchvision    
train = torchvision.datasets.FashionMNIST(
    root='FashionMNIST/raw/train-images-idx3-ubyte',
    train=True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
    )

test = torchvision.datasets.FashionMNIST(
    root='FashionMNIST/raw/train-images-idx3-ubyte',
    train=False,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
    )


train_loader = DataLoader(train,batch_size=10)
test_loader = DataLoader(test,batch_size=10)

# importing fashion mnist dataset from local directory (data downloaded from: https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download)
class dataset_fashionMNIST(Dataset):

    def __init__(self,path_dir='fashion-mnist_train.csv',transform=None,labels_transform=None):
        self.path = "FashionMNIST_local/"+path_dir
        self.data = pd.read_csv(self.path).to_numpy()
        self.X = torch.tensor(self.data[:,1:]).reshape(-1,1,28,28)
        self.Y = torch.tensor(self.data[:,0])
        self.transform = transform
        self.labels_transform = labels_transform

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self,index):
        return self.X[index],self.Y[index]
        

train_loc = dataset_fashionMNIST()
test_loc = dataset_fashionMNIST(path_dir='fashion-mnist_test.csv')

train_loader_loc = DataLoader(train_loc,batch_size=10)
test_loader_loc = DataLoader(test_loc,batch_size=10)

