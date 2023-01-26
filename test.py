
import torch
import torchvision
import pytorch_lightning as pl

from model import Img2Vec, FixedData
from pytorch_lightning.loggers import CSVLogger

from gensim.models import KeyedVectors

from PIL import Image
import os
import matplotlib.pyplot as plt
import random


MODEL_STATE = r'local_data/epoch=10-step=6908.ckpt'

INPUT_DIR = r"C:/Users/adam3/Downloads/vec_test"

WORD_DIR = r'D:\Repos\img2vec\local_data\word2vec'

DEVICE = 'cpu'


def main():
    print("")
    
    model = Img2Vec()
    if MODEL_STATE[-5:] == '.ckpt':
        model.load_state_dict(torch.load(MODEL_STATE, map_location=DEVICE)["state_dict"])
    else:
        model.load_state_dict(torch.load(MODEL_STATE, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    word_lib = KeyedVectors.load(os.path.join(WORD_DIR, 'word2vec.kv'))
    
    names = []
    vec_mat = torch.zeros(2, 100)
    
    i = 0
    while i <= 1:
        n = input(" >>> Class "+str(i)+": ").lower().strip()
        
        if n in word_lib:
            names.append(n)
            vec_mat[i] = torch.tensor(word_lib[n])
            i += 1
        else:
            print("Error: invalid word.")
    
    print("")
    vec_mat = vec_mat.to(DEVICE)
    
    reshaper = torchvision.transforms.Resize((256, 256))
    
    image_list = list(os.listdir(INPUT_DIR))
    random.shuffle(image_list)
    for img_path in image_list:
        if img_path[-4:].lower() not in ['.jpg', 'jpeg', '.png']:
            continue
          
        img = Image.open(os.path.join(INPUT_DIR, img_path))
        tens = torchvision.transforms.functional.to_tensor(img)
        
        if tens.shape[0] == 1:
            continue
        
        min_dim = min(tens.shape[1], tens.shape[2])
        cropper = torchvision.transforms.CenterCrop((min_dim, min_dim))
        tens = reshaper(cropper(tens))
        
        pred = torch.squeeze(model.infer(torch.unsqueeze(tens.to(DEVICE), 0), vec_mat))
        for i in [0, 1]:
            print(names[i]+':', round(pred[i].item(), 3))
        
        plt.imshow(tens.permute(1, 2, 0).numpy())
        plt.savefig("./figs/test_out.png")
        
        input("...")
        print("")
        

if __name__ == '__main__':
    main()