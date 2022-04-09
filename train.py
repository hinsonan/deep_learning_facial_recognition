import torch
import numpy as np
from data_process import FaceDataset
from torchvision import transforms
from torch.utils.data import random_split,DataLoader
from emotional_classification_model import EmotionalModel
from sklearn.metrics import accuracy_score,precision_score,recall_score
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
BATCH_SIZE = 32
EPOCHS = 10

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # note that this may not be the most accurate way to normalize
        transforms.Normalize(mean=[0.5],std=[0.5])
    ])
    dataset = FaceDataset(transform)
    train_size = int(len(dataset)*.6)
    val_size = int(len(dataset)*.2) + 1
    test_size = int(len(dataset)*.2) + 1
    train_set, val_set, test_set = random_split(dataset,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42))

    # now that we have our split data we need to create data loaders in order to use them
    training_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)

    validation_loader = DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True)

    testing_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

    return training_loader,validation_loader,testing_loader

def train_model():
    train_loader,val_loader,test_loader = load_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'You are using: {device}')

    model = EmotionalModel().to(device=device)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1,EPOCHS+1):
        train_loss = 0.0
        running_train_loss = 0.0
        for i,data in enumerate(train_loader):
            inputs,labels = data
            
            inputs = inputs.to(device=device)
            labels = labels.to(device=device).to(torch.int64)

            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = loss_fn(outputs,labels)
            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()
            # print info after certain amount of batch sizes
            if i % 10 == 0 and i != 0:
                train_loss = running_train_loss / 10
                # copy back to cpu to get the metrics
                copy_labels, copy_outputs = labels.cpu().detach().numpy().astype(int),outputs.cpu().detach().numpy()
                copy_outputs = np.argmax(copy_outputs,axis=1)
                train_acc = accuracy_score(copy_labels, copy_outputs)
                train_recall = recall_score(copy_labels, copy_outputs,average='macro')
                train_prec = precision_score(copy_labels, copy_outputs,average='macro')
                print(f'batch: {i} Loss: {train_loss} Acc: {train_acc}, Precision: {train_prec}, Recall: {train_recall}',end='\r')
                running_train_loss = 0.0
        
        # run through val set at the end of every epoch
        running_vloss = 0.0
        running_vaccuracy = 0.0
        running_vprecision = 0.0
        running_vrecall = 0.0
        for i, val_data in enumerate(val_loader):
            vinputs, vlabels = val_data
            vinputs = vinputs.to(device=device)
            vlabels = vlabels.to(device=device).to(torch.int64)
            with torch.no_grad():
                voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            copy_vlabels, copy_voutputs = vlabels.cpu().detach().numpy().astype(int),voutputs.cpu().detach().numpy()
            copy_voutputs = np.argmax(copy_voutputs,axis=1)
            vacc = accuracy_score(copy_vlabels, copy_voutputs)
            vprec = precision_score(copy_vlabels, copy_voutputs,average='macro')
            vrecall = recall_score(copy_vlabels, copy_voutputs,average='macro')
            running_vloss += vloss
            running_vaccuracy += vacc
            running_vprecision += vprec
            running_vrecall += vrecall

        avg_vloss = running_vloss / (i+1)
        avg_vacc = running_vaccuracy / (i+1)
        avg_vprec = running_vprecision / (i+1)
        avg_recall = running_vrecall / (i+1)
        print(f'EPOCH: {epoch} LOSS train {train_loss:.3f} valid {avg_vloss:.3f}, '
              f'Acc: train {train_acc:.3f} val {avg_vacc:.3f}, '
              f'Precision: train {train_prec:.3f} val {avg_vprec:.3f}, '
              f'Recall: train {train_recall:.3f} val {avg_recall:.3f}')
                    

if __name__ == '__main__':
    train_model()
