from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
import numpy as np


class MangoDataset(Dataset):
    def __init__(self, root, train = True):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        self.root = root
        
        self.train=train
        self.transform = transforms.Compose(
            [transforms.Resize((245,245)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomRotation(15, resample=False, expand=False, center=None),
#              transforms.RandomRotation(30, resample=False, expand=False, center=None),
#              transforms.RandomRotation(30, resample=False, expand=False, center=None),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])
        self.transform_test = transforms.Compose(
            [transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])

        # load image path and annotations
        if self.train=="train":
            df = pd.read_csv(root+"train.csv")
        elif self.train=="val":
            df = pd.read_csv(root+"dev.csv")
        elif self.train=="test":
            df = pd.read_csv(root+"test_example.csv")
        self.imgs = df["image_id"].to_numpy()
        label = df["label"].to_numpy()
        label = np.where(label=="A", 0, label)
        label = np.where(label=="B", 1, label)
        numerical_lbls = np.where(label=="C", 2, label)
        self.lbls = numerical_lbls
        self.ps = np.zeros(len(self.imgs))
        self.conf = np.zeros(len(self.imgs))
        assert len(self.imgs) == len(self.lbls), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        imgpath = self.imgs[index]
        if self.train=="train":
        
            img = Image.open(self.root+"C1-P1_Train/"+imgpath).convert('RGB')
            img = self.transform(img)
        elif self.train=="val":
            img = Image.open(self.root+"C1-P1_Dev/"+imgpath).convert('RGB')
            img = self.transform_test(img)
        elif self.train=="test":
            img = Image.open(self.root+"C1-P1_Test/"+imgpath).convert('RGB')
            img = self.transform_test(img)
            return img
        lbl = int(self.lbls[index])
        return img, lbl,self.conf[index],index,self.ps[index]


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.imgs)