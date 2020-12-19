import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from pprint import pprint

from dataset import DatasetLSTM
from models import FakenewsLSTM
import gensim
import torch.nn.functional as F
from tqdm.auto import tqdm
from pytorch_lightning.metrics import F1

fpath = "../data/fake.token"
tpath = "../data/true.token"
glove = "../data/glove.6B.100d.word2vec.bin"


def get_sent_vect(mx=-1):
    def readfile(p, m):
        l = []
        with open(p, "r") as file:
            sentence = []
            sid = 0
            for token in file:
                if m > 0 and len(l) >= (m // 2): return l
                t = token.split("\t")
                csid = int(t[2]) #current sid
                if csid != sid:
                    l.append(sentence)
                    sentence = [t[4]]
                    sid = csid
                else:
                    sentence.append(t[4])
        return l
    
    false = readfile(fpath, mx)
    true = readfile(tpath, mx)

    return false, true

def get_art_vect(mx=-1):
    def readfile(p, m):
        l = []
        with open(p, "r") as file:
            article = []
            aid = 0
            for token in file:
                if m > 0 and len(l) >= (m // 2): return l
                t = token.split("\t")
                caid = int(t[1]) #current article id
                if caid != aid:
                    l.append(article)
                    article = [t[4]]
                    aid = caid
                else:
                    article.append(t[4])
        return l
    
    false = readfile(fpath, mx)
    true = readfile(tpath, mx)

    return false, true



# lista di frasi #con glove no punct
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def train(fake, true, batch_size, embeddings_file, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    word_vec = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True
        )
    # Load data
    data = DatasetLSTM(fake, true, word_vec)
    # Parameters
    params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 0,
        "collate_fn": partial(DatasetLSTM.generate_batch, vocab = data.vocab ),
    }

    # split data
    train_len = int((80 * len(data)) // 100.0)
    val_len = len(data) - train_len
    print("Train size:", train_len)
    print("Val size:", val_len)
    train_set, val_set = torch.utils.data.random_split(data, [train_len, val_len])
    # load generators
    train_generator = torch.utils.data.DataLoader(train_set, **params)
    val_generator = torch.utils.data.DataLoader(val_set, **params)
    # train step
    model = FakenewsLSTM(word_vec).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss().to(device)
    f1 = F1(num_classes=1).to(device)
    for epoch in range(epochs):
        # train step
        train_loop = tqdm(train_generator)
        train_loop.set_description("Epoch {}/{}".format(epoch + 1, epochs))
        train_loss, train_acc = train_step(criterion, device, train_loop, model, optimizer)
        # val step
        val_loop = tqdm(val_generator)
        val_loop.set_description("Epoch {}/{}".format(epoch + 1, epochs))
        val_loss, val_f1 = val_step(model, criterion,f1, device, train_loop)
        
        # print total loss
        epoch_loss = train_loss / len(train_generator)
        epoch_acc =  train_acc / len(train_generator)

        val_epoch_loss = val_loss / len(val_generator)
        val_epoch_f1 = val_f1 / len(val_generator)

        print("train loss: {:.3}".format(epoch_loss))
        print("train acc: {:.3}".format(epoch_acc))

        print("val loss: {:.3}".format(val_epoch_loss))
        print("val f1: {:.3}".format(val_epoch_f1))


        


def train_step(criterion, device, loop, model, optimizer):
    # Training
    train_loss = train_acc =  0.0
    for i, batch in enumerate(loop):
        # Transfer to GPU
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(x)
        loss = criterion(outputs, y.float().unsqueeze(1))
        acc = binary_acc(outputs, y.float().unsqueeze(1))

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc.item()
        loop.set_postfix(loss=loss.item())
        loop.set_postfix(acc=acc.item())
    return train_loss, train_acc

def val_step(model, criterion, f1, device, loop):
    val_loss = val_f1 = 0.0
    for i, batch in enumerate(loop):
        # Transfer to GPU
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        # evaluate
        with torch.no_grad():
            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y.float().unsqueeze(1))
            f1_ = f1(outputs, y.float().unsqueeze(1))
        
        val_loss += loss.item()
        val_f1 += f1_.item()
        loop.set_postfix(loss=loss.item())
        loop.set_postfix(f1=f1_.item())
    return val_loss, val_f1


fake, true = get_art_vect(20000)
train(fake, true, 32 , glove, epochs=5)
