
import torch
import torchvision
import pytorch_lightning as pl

from model import Img2Vec, FixedData
from pytorch_lightning.loggers import CSVLogger

# from gensim.models import KeyedVectors


START_MODEL = r"local_data/epoch=10-step=6908.ckpt"


def main():
    print("")

    data = torchvision.datasets.Caltech256("./local_data/cifar-10-batches-py", download=True)
    data = FixedData(data, skip=3)

    class_vecs = torch.load("./data/256_words.pt")

    # cats = []
    # for i in range(len(data.categories)):
    #     name = data.categories[i][4:]
        
    #     if name[-4:] == "-101":
    #         name = name[:-4]
    #     while '-' in name:
    #         name = name[name.find('-')+1:]
            
    #     cats.append(name)
    
    # word_vecs = KeyedVectors.load('./word2vec.kv')
    # for i in range(len(cats)):
    #     c = cats[i]
    #     if c == 'breadmaker':
    #         c = 'stove'
    #     class_vecs[i] = torch.tensor(word_vecs[c])
    # torch.save(class_vecs, "./data/256_words.pt")
    
    model = None
    if START_MODEL is None:
        model = Img2Vec(class_vecs, lr=1e-5)
    else:
        model = Img2Vec.load_from_checkpoint(START_MODEL)
    
    model.cuda()
    model.train()

    loader = torch.utils.data.DataLoader(
        data, shuffle=True, batch_size=16, num_workers=0
    )

    logger = CSVLogger(
            save_dir='.',
            flush_logs_every_n_steps=100000
        )

    # callback for best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss", mode="min", save_top_k=1
    )

    # init trainer to run forever
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=-1,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # train forever...
    trainer.fit(model, loader)
    

if __name__ == '__main__':
    main()