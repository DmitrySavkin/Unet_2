import segmentation_models_pytorch as smp
#hide
#!pip install -q git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
from tqdm.auto import tqdm
from typing import Any, Callable, List, Optional, Tuple
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
# %matplotlib inline

ROOT_PATH = Path("../coco_unet_segmentation/coco_dataset/")
BATCH_SIZE = 64
IMAGE_SIZE = (128, 128)
#  ../coco_unet_segmentation/coco_dataset/

train_annotations = COCO(ROOT_PATH / "annotations/instances_train2017.json")
valid_annotations = COCO(ROOT_PATH / "annotations/instances_val2017.json")

cat_ids = train_annotations.getCatIds(supNms=["person", "vehicle"])
train_img_ids = []
for cat in cat_ids:
    train_img_ids.extend(train_annotations.getImgIds(catIds=cat))
    
train_img_ids = list(set(train_img_ids))
print(f"Number of training images: {len(train_img_ids)}")

valid_img_ids = []
for cat in cat_ids:
    valid_img_ids.extend(valid_annotations.getImgIds(catIds=cat))
    
valid_img_ids = list(set(valid_img_ids))
print(f"Number of validation images: {len(valid_img_ids)}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv64 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv128 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv256 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv512 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.upconv1024 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dconv1024 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upconv512 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dconv512 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv256 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dconv256 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv128 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv128 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 183, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = F.relu(self.conv64(x / 255.))
        x2 = F.relu(self.conv128(self.pool(x1)))
        x3 = F.relu(self.conv256(self.pool(x2)))
        x4 = F.relu(self.conv512(self.pool(x3)))
        x5 = F.relu(self.conv1024(self.pool(x4)))
        ux5 = self.upconv1024(x5)
        cc5 = torch.cat([ux5, x4], 1)
        dx4 = F.relu(self.dconv1024(cc5))
        ux4 = self.upconv512(dx4)
        cc4 = torch.cat([ux4, x3], 1)
        dx3 = F.relu(self.dconv512(cc4))
        ux3 = self.upconv256(dx3)
        cc3 = torch.cat([ux3, x2], 1)
        dx2 = F.relu(self.dconv256(cc3))
        ux2 = self.upconv128(dx2)
        cc2 = torch.cat([ux2, x1], 1)
        dx1 = F.relu(self.dconv128(cc2))  # no relu?
        last = self.conv1(dx1)
        return last  # sigmoid if classes arent mutually exclusv


class ImageData(Dataset):
    def __init__(
        self, 
        annotations: COCO, 
        img_ids: List[int], 
        cat_ids: List[int], 
        root_path: Path, 
        transform: Optional[Callable]=None
    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.img_data = annotations.loadImgs(img_ids)
        self.cat_ids = cat_ids
        self.files = [str(root_path / img["file_name"]) for img in self.img_data]
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[i]['id'], 
            catIds=self.cat_ids, 
            iscrowd=None
        )
        anns = self.annotations.loadAnns(ann_ids)
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] 
                                                 for ann in anns]), axis=0)).unsqueeze(0)
        
        img = io.read_image(self.files[i])
        if img.shape[0] == 1:
            img = torch.cat([img]*3)
        
        if self.transform is not None:
            return self.transform(img, mask)
        
        return img, mask


def train_transform(
    img1: torch.LongTensor, 
    img2: torch.LongTensor
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    params = transforms.RandomResizedCrop.get_params(img1, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    
    img1 = TF.resized_crop(img1, *params, size=IMAGE_SIZE)
    img2 = TF.resized_crop(img2, *params, size=IMAGE_SIZE)
    
    # Random horizontal flipping
    if random.random() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
        
    return img1, img2



train_data = ImageData(train_annotations, train_img_ids, cat_ids, ROOT_PATH / "images/train2017", train_transform)
valid_data = ImageData(valid_annotations, valid_img_ids, cat_ids, ROOT_PATH / "images/val2017", train_transform)

train_dl = DataLoader(
    train_data,
    BATCH_SIZE, 
    shuffle=True, 
    drop_last=True, 
    num_workers=4,
    pin_memory=True,
)

valid_dl = DataLoader(
    valid_data,
    BATCH_SIZE, 
    shuffle=False, 
    drop_last=False, 
    num_workers=4,
    pin_memory=True,
)

model = Net()
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
#nn.BCEWithLogitsLoss()

n_iter = 2
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.reshape(-1, 128, 128))
        writer.add_scalar("loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % n_iter == n_iter-1 :    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / n_iter:.3f}')
            running_loss = 0.0

print('Finished Training')
# writer.flush()