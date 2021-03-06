import myaugmentation
import mydataset
import mymodel
import myloss

import argparse
import os
import numpy as np
import random
import torch
import wandb
import yaml

import multiprocessing
from tqdm import tqdm
from importlib import import_module
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def seed_setting(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(model, optimizer_name="Adam", lr=1e-3, momentum=0.9, weight_decay=5e-4):
    if optimizer_name.lower()=="adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower()=="momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower()=="sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

INDEX=[0]
def train():
    INDEX[0] += 1
    # run_name = sweep_config['run_name'] + f"_{INDEX[0]}"
    run_name = yaml_sweep_config['run_name'] + f"_{INDEX[0]}"
    
    with wandb.init(name=run_name) as run:
        args = wandb.config
        # initial settings
        seed_setting(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # dataset
        dataset_module = getattr(import_module("mydataset"), args.dataset)
        dataset = dataset_module(data_dir=args.main_dir, val_ratio=0.2)
        
        # augmentation
        transform_module = getattr(import_module("myaugmentation"), args.augmentation)
        transform = transform_module(
            train=True,
            img_size=args.resize,
            # mean=dataset.mean,
            # std=dataset.std,
        )
        dataset.set_transform(transform)
        train_set, val_set = dataset.split_dataset()
        
        # data_loader
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=False,
            drop_last=True,
        )

        # model
        model_module = getattr(import_module("mymodel"), "config_model")  # default: BaseModel
        model = model_module(model_name=args.model, num_classes=18).to(device)
        model = torch.nn.DataParallel(model)

        # criterion & optimizer & learning_rate_decay_step
        criterion_module = getattr(import_module("myloss"), "get_criterion")  # default: BaseModel
        criterion = criterion_module(args.criterion)
        optimizer = get_optimizer(model, optimizer_name=args.optimizer, lr=args.lr)
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in tqdm(enumerate(train_loader)):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)
                loss.backward()
                optimizer.step()
                
                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = args.lr
                    print(
                        f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )

                    loss_value = 0
                    matches = 0
            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in tqdm(val_loader):
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"args.model_dir = {args.model_dir}")
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{args.model_dir}/{run_name}_best{val_acc:4.2%}.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{args.model_dir}/{run_name}_last{val_acc:4.2%}.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                print()

            wandb.log({'Train_loss': train_loss, 'Train_Acc': train_acc, 'Eval_loss': val_loss, 'Eval_Acc': val_acc})
        run.finish()

sweep_config = {
    "program": "train.py",
    "method": "grid",
    # "method": "bayes",
    # "metric":
    # {
    #     "name": "val_loss"
    #     "goal": "minimize"
    # }
    # "parameters":
    # {
    #     "learning_rate":
    #     {
    #         "min": 0.0001
    #         "max": 0.1
    #     }
    #     "optimizer":
    #         "values": ["adam", "sgd"]
    # }    
    "run_name": "Grid_Sweep",
    "parameters":
    {
        "seed":
            {"value": 42},
        "main_dir":
            {"value": './input/data/train'},
        "dataset":
            {"value": 'MyDataset'},
        "augmentation":
            {"value": 'my_transform'},
        "resize":
            {"value": (256, 192)},
        "optimizer":
            {"value": 'Adam'},
        "model":
            {"values": ['resnet34', 'resnet152']},
        "batch_size":
            {"values": [64, 80, 96, 112]},
        "lr":
            {"values": [0.001, 0.005, 0.0005]},
        "lr_decay_step":
            {"value": 10},
        "criterion":
            {"value": 'cross_entropy'},
        "epochs":
            {"value": 5},
        "log_interval":
            {"value": 20},
        "model_dir":
            {"value": './model'},
        "name":
            {"value": 'exp'},
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--main_dir', type=str, default='./input/data/train', help='Main Directory Path (default: ./input/data/train)')
    parser.add_argument('--dataset', type=str, default='MyDataset', help='Dataset Type (default : MyDataset)')
    parser.add_argument('--augmentation', type=str, default='my_transform', help='Augmentation Type (default: my_transform)')
    parser.add_argument("--resize", nargs="+", type=list, default=[256, 192], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='resnet152', help='model type (default: resnet152)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    args = parser.parse_args()
    # args_dict = vars(args)
    
    # yaml_sweep_config = dict()

    with open('mysweep.yaml') as f:
        yaml_sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(yaml_sweep_config)

    # stream = open("mysweep.yaml", 'r')
    # yaml_data = yaml.load_all(stream)
    # # print(f"yaml_data = {yaml_data}\n\n")
    # for data in yaml_data:
    #     for key, value in data.items():
    #         # print(key, value)
    #         yaml_sweep_config[key] = value
    
    # print(sweep_config)
    # print("\n\n")
    # print(yaml_sweep_config)

    # print(sweep_config)
    # print(yaml_sweep_config)
    
    sweep_id = wandb.sweep(yaml_sweep_config, entity="falling90", project="Test-Project")
    wandb.agent(sweep_id, function=train, count=24)

    # train(args)