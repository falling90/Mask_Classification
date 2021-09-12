from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from dataset import *
from model import *
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import wandb
import tqdm
import torch.optim as optim
import argparse
from dotenv import load_dotenv
import os

# kfold cross validation dataset
def create_kfold_datasets(df, fold_type = 'stratified_kfold', n_splits = 0):
    '''
        (fold_type) = 'kfold' or 'stratified_kfold' -> (string)
        (n_splits) = 0 -> (int)
    '''
    if fold_type == 'kfold':
        kfold = KFold(n_splits=n_splits, shuffle=True)
        def fold_dataset():
            for train_index, val_index in kfold.split(df):
                train_dataset = MaskDataset(df, train_index, train=True)
                val_dataset = MaskDataset(df, val_index, train=False)
                yield train_dataset, val_dataset
        fold_datasets = fold_dataset()
        return fold_datasets
    elif fold_type == 'stratified_kfold':
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        kfold_label = df['NewClass']
        def fold_dataset():
            for train_index, val_index in kfold.split(df, kfold_label):
                train_dataset = MaskDataset(df, train_index, train=True)
                val_dataset = MaskDataset(df, val_index, train=False)
                yield train_dataset, val_dataset
        fold_datasets = fold_dataset()
        return fold_datasets
        
    else : 
        print("Fold type error : Use 'kfold' or 'stratified_kfold' ")

def train_model(model, criterion, optimizer,fold_datasets, num_epochs=1,):
    '''
    
    ## train_model

        ### input:
            model : 
                custom model

            criterion :

            optimizer:

            fold_datasets:

            num_epochs:


        ### description:
            train model

        ### output:

            traiend model    
    '''

    for i, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f'k-fold : {i+1}')
        print('-' * 10)
        image_datasets = {'train':train_dataset,'validation':val_dataset}
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            shuffle=True
        )
        dataloaders = {'train':train_loader, 'validation':val_loader}
        mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(mydevice)
                    labels = labels.to(mydevice)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])
                wandb.log({f"{phase}_acc":epoch_acc, f"{phase}_loss":epoch_loss})
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
    return model

def test_model(model,test_dataset,test_loader):
    """
    ## train_model

        ### input:
            model : 
                custom model

            train_dataset

            train_loader


        ### description:
            eval model
    """
    mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(mydevice)
        labels = labels.to(mydevice)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_dataset)
    test_acc = running_corrects.double() / len(test_dataset)
    print('test loss: {:.4f}, acc: {:.4f}'.format(test_loss,test_acc))
    wandb.log({f"test_acc":test_acc, f"test_loss":test_loss})
    print('test_done')

# if __name__ == '__main__':
parser = argparse.ArgumentParser()

load_dotenv(verbose=True)

# Data and model checkpoints directories
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--batchsize', type=int, default=16, help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
parser.add_argument('--model', type=str, default='resnet152', help='model type (default: resnet152)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum weight (default: 0.9)')
parser.add_argument('--kfoldnum', type=int, default=3, help='k-fold num (default: 3)')

args = parser.parse_args()
print(args)

# if possible convert to GPU
mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# TRAIN_MASK_PATH = {'label':'../input/data/train/csvs/train.csv','images':'../input/data/train/images','new':'../input/data/train/csvs/new_train.csv'}
TRAIN_MASK_PATH = {'label':'../input/data/train/csvs/train.csv',
                   'images':'../input/data/train/images',
                   'new':'../input/data/train/new_train.csv'}
TEST_MASK_PATH = '../input/data/eval'
img_size_x=[512, 256, 299]
img_size_y=[384, 192, 224]
# wandb login YOUR-API-KEY --relogin
# wandb.init(project='', entity='',
#             config = {'learning_rate':args.lr,
#                     'batch_size':args.batchsize,
#                     'epoch':args.epochs,
#                     'model': args.model,
#                     'momentum':args.momentum,
#                     'kfold_num':args.kfoldnum,
#                     'img_x':img_size_x[2],
#                     'img_y':img_size_y[2],
#                     }
# )
# config = wandb.config
config = {'learning_rate':args.lr,
        'batch_size':args.batchsize,
        'epoch':args.epochs,
        'model': args.model,
        'momentum':args.momentum,
        'kfold_num':args.kfoldnum,
        'img_x':img_size_x[2],
        'img_y':img_size_y[2],
}


# data import & split data
df = pd.read_csv(TRAIN_MASK_PATH['new'])
test_length = len(df) - int(len(df)*0.2)
test_df = df.iloc[test_length:]
train_df = df.iloc[:test_length]
# fold_datasets = create_kfold_datasets(train_df, 'stratified_kfold', config['kfold_num'])
#
# model_list = ['resnet152', 'custom_model', 'vit_base_patch16_224']

# train data settings
# train_dataset = MaskDataset(train_df)
# train_loader = DataLoader(
#     train_dataset,
#     shuffle=True
# )

# test data settings
test_dataset = MaskTestDataset(test_df)
test_loader = DataLoader(
    test_dataset,
    shuffle=True
)

model_list = ['resnet152', 'custom_model', 'vit_base_patch16_224']

for model_name in model_list:
    # wandb.init(project='', entity='')

    model = config_model(model_name)
    #Hyper parameter 가져오기
    lr = config['learning_rate']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config['momentum'])

    #모델 학습
    fold_datasets = create_kfold_datasets(train_df, 'stratified_kfold', config['kfold_num'])
    model = train_model(model, criterion, optimizer, fold_datasets, num_epochs=config['epoch'])
    model.eval()

    print('---------------------------------------------------------------------------')
    print(f'testing model: {model_name}')
    test_model(model,test_dataset,test_loader)
    print('---------------------------------------------------------------------------')

    print('saving model')
    torch.save(model.state_dict(), f'/opt/ml/code/model/{model_name}.pth')
    print('model saved')

    print('done')