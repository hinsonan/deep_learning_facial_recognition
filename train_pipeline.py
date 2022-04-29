import json
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split,DataLoader
import yaml
from emotional_classification_model import EmotionalModel
from sklearn.metrics import accuracy_score,precision_score,recall_score
import warnings
from data_process import FaceDataset
from plot_helpers import plot_learning_curve
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

class TrainingPipeline():

    def __init__(self,config_path:str) -> None:
        with open(config_path,'r') as f:
            config = yaml.load(f)
        
        # experiment general information 
        self.experiment_name = config['experiment_name']
        self.model_path = config['model']['save_path']
        self.do_training = config['model']['train']
        self.do_eval = config['model']['eval']
        self.model_type = config['model']['model_type']

        # use the augmented data or not 
        self.use_augmented_data = config['use_augmented_data']

        # read in hyper params
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.learning_rate = config['lr']

        # load data
        self.train_loader,self.val_loader,self.test_loader = self.load_data()

    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize(size=(128,128)),
            transforms.ToTensor(),
            # note that this may not be the most accurate way to normalize
            transforms.Normalize(mean=[0.5],std=[0.5])
        ])
        dataset = FaceDataset(transform,self.use_augmented_data)
        train_size = int(len(dataset)*.6)
        val_size = int(len(dataset)*.2) + 1
        test_size = int(len(dataset)*.2)
        train_set, val_set, test_set = random_split(dataset,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42))

        # now that we have our split data we need to create data loaders in order to use them
        training_loader = DataLoader(train_set,batch_size=self.batch_size,shuffle=True)

        validation_loader = DataLoader(val_set,batch_size=self.batch_size,shuffle=True)

        testing_loader = DataLoader(test_set,batch_size=self.batch_size,shuffle=True)

        return training_loader,validation_loader,testing_loader

    def validate(self,model,device,loss_fn):
        # run through val set at the end of every epoch
        running_vloss = 0.0
        running_vaccuracy = 0.0
        running_vprecision = 0.0
        running_vrecall = 0.0
        for i, val_data in enumerate(self.val_loader):
            vinputs, vlabels = val_data
            vinputs = vinputs.to(device=device)
            vlabels = vlabels.to(device=device).to(torch.int64)
            with torch.no_grad():
                voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            torch.cuda.synchronize()            
            copy_vlabels, copy_voutputs = vlabels.cpu().detach().numpy().astype(int),voutputs.cpu().detach().numpy()
            copy_voutputs = np.argmax(copy_voutputs,axis=1)
            vacc = accuracy_score(copy_vlabels, copy_voutputs)
            vprec = precision_score(copy_vlabels, copy_voutputs,average='macro')
            vrecall = recall_score(copy_vlabels, copy_voutputs,average='macro')
            running_vloss += vloss.cpu().detach().numpy()
            running_vaccuracy += vacc
            running_vprecision += vprec
            running_vrecall += vrecall

        avg_vloss = running_vloss / (i+1)
        avg_vacc = running_vaccuracy / (i+1)
        avg_vprec = running_vprecision / (i+1)
        avg_recall = running_vrecall / (i+1)
        return avg_vloss,avg_vacc,avg_vprec,avg_recall

    def train_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'You are using: {device}')

        model = EmotionalModel(self.model_type).to(device=device)

        optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate)

        loss_fn = torch.nn.CrossEntropyLoss()

        metrics = {'train_loss':[],'train_acc':[],'train_precision':[],'train_recall':[],'val_loss':[],'val_acc':[],'val_precision':[],'val_recall':[]}
        for epoch in range(1,self.epochs+1):
            train_loss = 0.0
            running_train_loss = 0.0
            for i,data in enumerate(self.train_loader):
                inputs,labels = data
                
                inputs = inputs.to(device=device)
                labels = labels.to(device=device).to(torch.int64)

                optimizer.zero_grad()

                # optimizer.param_groups[0]["lr"] = 0.0001
                
                outputs = model(inputs)

                loss = loss_fn(outputs,labels)
                loss.backward()

                optimizer.step()

                running_train_loss += loss.item()
                # print info after certain amount of batch sizes
                if i % 10 == 0 and i != 0:
                    train_loss = running_train_loss / 10
                    # copy back to cpu to get the metrics
                    torch.cuda.synchronize()
                    copy_labels, copy_outputs = labels.cpu().detach().numpy().astype(int),outputs.cpu().detach().numpy()
                    copy_outputs = np.argmax(copy_outputs,axis=1)
                    train_acc = accuracy_score(copy_labels, copy_outputs)
                    train_recall = recall_score(copy_labels, copy_outputs,average='macro')
                    train_prec = precision_score(copy_labels, copy_outputs,average='macro')
                    print(f'batch: {i} Loss: {train_loss} Acc: {train_acc}, Precision: {train_prec}, Recall: {train_recall}',end='\r')
                    running_train_loss = 0.0

            # end of epoch metrics
            avg_vloss,avg_vacc,avg_vprec,avg_recall = self.validate(model,device,loss_fn)
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['train_precision'].append(train_prec)
            metrics['train_recall'].append(train_recall)
            metrics['val_loss'].append(avg_vloss)
            metrics['val_acc'].append(avg_vacc)
            metrics['val_precision'].append(avg_vprec)
            metrics['val_recall'].append(avg_recall)
            print(f'EPOCH: {epoch} LOSS train {train_loss:.3f} valid {avg_vloss:.3f}, '
                f'Acc: train {train_acc:.3f} val {avg_vacc:.3f}, '
                f'Precision: train {train_prec:.3f} val {avg_vprec:.3f}, '
                f'Recall: train {train_recall:.3f} val {avg_recall:.3f}')
            
        # training is done return the metrics
        self.plot_metrics(metrics)

        # save the model
        torch.save(model,self.model_path)

    def plot_metrics(self,metrics:dict):
        if not os.path.isdir(f'experiment_results/{self.experiment_name}'):
            os.mkdir(f'experiment_results/{self.experiment_name}')
        plot_learning_curve(metrics['train_loss'],metrics['val_loss'],'Loss',f'experiment_results{os.sep}{self.experiment_name}{os.sep}loss.png')
        plot_learning_curve(metrics['train_acc'],metrics['val_acc'],'Accuracy',f'experiment_results{os.sep}{self.experiment_name}{os.sep}accuracy.png')
        plot_learning_curve(metrics['train_precision'],metrics['val_precision'],'Precision',f'experiment_results{os.sep}{self.experiment_name}{os.sep}precision.png')
        plot_learning_curve(metrics['train_recall'],metrics['val_recall'],'Recall',f'experiment_results{os.sep}{self.experiment_name}{os.sep}recall.png')

        # write out metrics
        with open(f'experiment_results{os.sep}{self.experiment_name}{os.sep}metrics.json','w') as f:
            json.dump(metrics,f,indent=4)

    def evaluate(self):
        model = torch.load(self.model_path,map_location='cpu')
        model.eval()
        running_test_acc = 0
        running_test_prec = 0
        running_test_recall = 0
        for i,data in enumerate(self.test_loader):
            inputs, labels = data
            out = model(inputs)
            out = np.argmax(out.detach().numpy(),axis=1)
            acc = accuracy_score(labels, out)
            prec = precision_score(labels, out,average='macro')
            recall = recall_score(labels, out,average='macro')
            running_test_acc += acc
            running_test_prec += prec
            running_test_recall += recall

        avg_vacc = running_test_acc / (i+1)
        avg_vprec = running_test_prec / (i+1)
        avg_recall = running_test_recall / (i+1)
        results =  {'Test Accuracy':avg_vacc,'Test Precision':avg_vprec,'Test Recall':avg_recall}

        # write out metrics
        with open(f'experiment_results{os.sep}{self.experiment_name}{os.sep}evaluation.json','w') as f:
            json.dump(results,f,indent=4)

    def run_pipeline(self):
        if self.do_training:
            print('Begin Training')
            self.train_model()
            print('Done')
        if self.do_eval:
            print('Begin Evaluation')
            self.evaluate()
            print('Done')

                    

if __name__ == '__main__':
    pipeline = TrainingPipeline(f'config{os.sep}experiment3.yaml')
    pipeline.run_pipeline()
    
