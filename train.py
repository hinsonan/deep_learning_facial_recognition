import torch
from data_process import FaceDataset
from torchvision import transforms
from torch.utils.data import random_split,DataLoader
from emotional_classification_model import EmotionalModel

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

    for epoch in range(EPOCHS):
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
            if i % 10 == 0:
                train_loss = running_train_loss / 10
                print(f'\r batch: {i} loss: {train_loss}')
                running_train_loss = 0.0
        
        # run through val set at the end of every epoch
        for i, val_data in enumerate(val_loader):
            running_vloss = 0.0
            vinputs, vlabels = val_data
            vinputs = vinputs.to(device=device)
            vlabels = vlabels.to(device=device).to(torch.int64)
            with torch.no_grad():
                voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'EPOCH: {epoch} LOSS train {train_loss} valid {avg_vloss}')
                    

if __name__ == '__main__':
    train_model()
