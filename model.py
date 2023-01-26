
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

SQ = lambda x: torch.squeeze(x)
USQ = lambda x: torch.unsqueeze(x, 0)


class FixedData(torch.utils.data.DataLoader):
    def __init__(self, cal, skip=1):
        self.imgs = []
        self.labels = []
        
        reshaper = torchvision.transforms.Resize((256, 256))
        
        for item in tqdm(range(0, len(cal), skip)):
            img, y = cal[item]
            
            tens = torchvision.transforms.functional.to_tensor(img)
            
            if tens.shape[0] == 1:
                continue
            
            min_dim = min(tens.shape[1], tens.shape[2])
            cropper = torchvision.transforms.CenterCrop((min_dim, min_dim))
            tens = reshaper(cropper(tens))
            
            tens = (tens * 255).to(torch.int8)
            self.imgs.append(tens.to("cuda"))
            self.labels.append(y)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, item):
        return self.imgs[item].to(torch.float32)/255, self.labels[item]


class Img2Vec(pl.LightningModule):
    def __init__(self, word_vecs=None, lr: float=1e-5):
        super().__init__()

        if word_vecs is None:
            word_vecs = torch.load("./data/256_words.pt")

        self.n_classes = word_vecs.shape[0]
        self.vec_len = word_vecs.shape[1]

        self.model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Identity()
        self.buffer = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.vec_len)
        )

        class_weights = word_vecs.clone()
        for i in range(self.n_classes):
            class_weights[i] = nn.functional.normalize(class_weights[i], dim=0)

        self.class_matrix = nn.Linear(self.vec_len, self.n_classes, bias=False)
        self.class_matrix.weight = nn.Parameter(class_weights)
        self.class_matrix.weight.requires_grad = False

        self.softy = nn.Softmax(dim=-1)

        self.lr = lr

        self.loss_func = nn.CrossEntropyLoss()
        

    def configure_optimizers(self):
        # create optimizer with our lr
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        return optimizer


    def forward(self, x):
        feats = self.model(x)
        vec = self.buffer(feats)
        vec = nn.functional.normalize(vec, dim=-1)
        return self.class_matrix(vec)


    def infer(self, x, word_vecs):
        
        class_weights = word_vecs.clone()
        for i in range(word_vecs.shape[0]):
            class_weights[i] = nn.functional.normalize(class_weights[i], dim=0)

        feats = self.model(x)
        vec = self.buffer(feats)
        vec = nn.functional.normalize(vec, dim=-1)
        
        curr_matrix = nn.Linear(self.vec_len, word_vecs.shape[0], bias=False)
        curr_matrix.weight = nn.Parameter(class_weights)

        output = curr_matrix(vec)

        return self.softy(5*word_vecs.shape[0]*output)


    def _step(self, batch, prefix):

        x, y = batch

        pred = self.forward(x)

        loss = self.loss_func(pred, y)

        self.log(prefix+'_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        rank_sum = 0
        for b in range(x.shape[0]):
            rank_sum += torch.sum(torch.where(pred[b] >= pred[b][int(y[b])], 1, 0))

        self.log('avg_rank', rank_sum/x.shape[0], on_epoch=True, on_step=False, prog_bar=True)
        
        self.log('monitor', self.class_matrix.weight[0][0].item(),  on_epoch=False, on_step=True, prog_bar=True)

        return loss


    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")
        
    
    # def validation_step(self, batch, batch_idx):
    #     return self._step(batch, "val")