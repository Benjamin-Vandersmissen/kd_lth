import torch
from torch.cuda.amp import GradScaler
import tqdm
import wandb
import math
import torch.nn.functional as F
import os
import copy

from recipe import TrainingRecipe
from utils import evaluate, Step

def train_tna(model, twin_model, recipe: TrainingRecipe, train_loader : torch.utils.data.DataLoader, eval_loader : torch.utils.data.DataLoader, lamb: float, amp: bool = False, device=torch.device('cuda'), late_matching=0):
    """
    Train a network using the Twin Network Augmentation strategy.
    Params:
    -------
        model : The base model.
        twin_model : The twin model. In the normal TNA operations, this should be the same architecture with different initialization as the base model, but the ablations allow for different variations.
        recipe : A Recipe object used for training configuration
        train_loader : A dataloader for the training data.
        eval_loader : A dataloader for the validation data.
        lamb : A trade-off parameter for the Logit Matching Loss.
        amp : Whether to use Automatic Mixed Precision to speed-up training.
        freeze_twin : Whether to use a frozen twin network during TNA. Typically used as an ablation with a pretrained twin network.
        twin_weights : A dictionary containing pretrained twin network weights. Typically used as an ablation in conjunction with freezing the weights.
        use_crossentropy : Whether to use the Cross-Entropy loss in the total loss calculation. For a knowledge distillation like approach, this is set to False.
        device : Which device to run the calculations on.
    """
    amp &= device.type == 'cuda'  # Safety check.
    train_losses = []
    accuracies = []

    device1 = torch.device('cuda:0') if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device
    device2 = torch.device('cuda:1') if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device

    model = model.to(device1)
    twin_model = twin_model.to(device2)
    
    optim = recipe.optimizer(model)
    scheduler = recipe.lr_scheduler(optim)
    criterion = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss(reduction='sum')

    if amp:
        scaler = GradScaler()
    
    for ep in tqdm.trange(recipe.train_epochs()):
        logging = {}
        model.train()
        epoch_totalloss = 0
        epoch_matchloss = 0
        for imgs, targets in train_loader:
            with torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
                out = model(imgs.to(device1))
                twin_out = twin_model(imgs.to(device2))
                
                loss = criterion(out, targets.to(device1))
                twin_loss = criterion(twin_out, targets.to(device2))
                matchloss = lamb*mseloss(out.to(dtype=torch.float32), twin_out.to(dtype=torch.float32)).cpu()/out.shape[0]  # Get the avg sum of square errors (sum over logits, mean over batch)

            optim.zero_grad()
            twin_optim.zero_grad()
            total_loss =  loss.to(device1) + twin_loss.to(device1)

            if ep >= late_matching:
                total_loss += matchloss.to(device1)
            if amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            epoch_totalloss += total_loss.item()
            # print(matchloss.item())
            epoch_matchloss += matchloss.item()
            # print(matchloss.item())
            
            if amp:
                scaler.step(optim)
                scaler.step(twin_optim)
                scaler.update()
            else:
                optim.step()
                twin_optim.step()
        
        model.eval()
        twin_model.eval()
        # print("Train Loss:", epoch_loss)
        train_losses.append(epoch_totalloss)
        logging |= {'Total Loss': epoch_totalloss, 'Match Loss': epoch_matchloss}
        scheduler.step()
        if not freeze_twin:
            twin_scheduler.step()
        
        acc = evaluate(model, eval_loader, amp)
        twin_acc = evaluate(twin_model, eval_loader, amp)
        accuracies.append((acc, twin_acc, (acc+twin_acc)/2))
        
        logging |= {'Accuracy': acc, 'Twin Accuracy': twin_acc}

        if use_wandb:
            wandb.log(logging)

    if model_dir is not None:
        torch.save(model.state_dict(), os.path.join(model_dir, 'base.pth'))
        torch.save(twin_model.state_dict(), os.path.join(model_dir, 'twin.pth'))



def train(model, recipe, train_loader, eval_loader, amp=False, device=torch.device('cuda'), use_wandb=True, start=None, end=None):
    """
        Train a model following a specified recipe with the train_loader and evaluate each epoch with eval_loader.
        model: torch.Module - Which model to train.
        recipe: Recipe - Which recipe to follow for training.
        train_loader: torch.utils.data.DataLoader - A dataloader for the training epochs
        eval_loader: torch.utils.data.DataLoader - A dataloader for the evaluation after each epoch.
        device: torch.device - Which device to train on.
        use_wandb: bool - Whether to log the Accuracies directly to WandB directly or not.
        start: Step - Which iteration to start the training process from (useful when stopping and starting training). Defaults to 0
        end: Step - Which iteration to end the training process. 
    """
    train_losses = []
    accuracies = []
    model = model.to(device)
    model.train()

    if start is None:
        start = Step(len(train_loader))

    if end is None:
        end = Step(len(train_loader), it=len(train_loader)*recipe.train_epochs())
    
    optim = recipe.optimizer(model)
    scheduler = recipe.lr_scheduler(optim, cur_ep=start.ep())
    criterion = torch.nn.CrossEntropyLoss()
    amp &= device.type == 'cuda'

    if amp:
        scaler = GradScaler()

    pbar = tqdm.tqdm(total=end.it, initial=start.it)
    while start.it != end.it:
        logging = {}
        epoch_loss = 0
        for imgs, targets in train_loader:
            with torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
                out = model(imgs.to(device))
                loss = criterion(out, targets.to(device))
            optim.zero_grad()

            if amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            start.update()
            pbar.update()
            
            epoch_loss += loss.item()

            if start.it == end.it:
                break

        torch.save(model.state_dict(), f'r18_cifar10_dense_train/model_{start.ep()}.pth')
        scheduler.step()  # Epoch-wise scheduler.
        train_losses.append(epoch_loss)
        logging |= {'Total Loss': epoch_loss}

        acc = evaluate(model, eval_loader, amp)
        accuracies.append(acc)
        logging |= {'Accuracy': acc}
        if use_wandb:
            wandb.log(logging)

    return accuracies

def train_mutual(model, twin_model, recipe: TrainingRecipe, train_loader : torch.utils.data.DataLoader, eval_loader : torch.utils.data.DataLoader, amp: bool = False, device=torch.device('cuda'), model_dir=None):
    """
    Train a network using the Twin Network Augmentation strategy.
    Params:
    -------
        model : The base model.
        twin_model : The twin model. In the normal TNA operations, this should be the same architecture with different initialization as the base model, but the ablations allow for different variations.
        recipe : A Recipe object used for training configuration
        train_loader : A dataloader for the training data.
        eval_loader : A dataloader for the validation data.
        amp : Whether to use Automatic Mixed Precision to speed-up training.
        device : Which device to run the calculations on.
    """
    amp = amp and device.type == 'cuda'  # Safety check.
    train_losses = []
    accuracies = []

    device1 = torch.device('cuda:0') if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device
    device2 = torch.device('cuda:1') if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device

    model = model.to(device1)
    twin_model = twin_model.to(device2)
    
    optim = recipe.optimizer(model)
    twin_optim = recipe.optimizer(twin_model)

    scheduler = recipe.lr_scheduler(optim)
    twin_scheduler = recipe.lr_scheduler(twin_optim)
    criterion = torch.nn.CrossEntropyLoss()
    kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

    if amp:
        scaler = GradScaler()
    
    for ep in tqdm.trange(recipe.train_epochs()): 
        logging = {}
        model.train()
        twin_model.train()
        epoch_totalloss = 0
        epoch_klloss = 0
        # with torch.autograd.detect_anomaly():
        for imgs, targets in train_loader:
            with torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
                out = model(imgs.to(device1))
                twin_out = twin_model(imgs.to(device2)).to(device1)
                
                loss = criterion(out, targets.to(device1))
                twin_loss = criterion(twin_out, targets.to(device1))
                
                loss_kl = kl_criterion(F.log_softmax(out, dim=1), F.softmax(twin_out, dim=1)) + kl_criterion(F.log_softmax(twin_out, dim=1), F.softmax(out, dim=1))
                
            optim.zero_grad()
            twin_optim.zero_grad()
            total_loss = loss + twin_loss + loss_kl
            if amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            epoch_totalloss += total_loss.item()
            epoch_klloss += loss_kl.item()
            
            if amp:
                scaler.step(optim)
                scaler.step(twin_optim)
                scaler.update()
            else:
                optim.step()
                twin_optim.step()
        
        model.eval()
        twin_model.eval()
        # print("Train Loss:", epoch_loss)
        train_losses.append(epoch_totalloss)
        logging |= {'Total Loss': epoch_totalloss, 'KL Loss': epoch_klloss}
        scheduler.step()
        twin_scheduler.step()
        
        acc = 0
        twin_acc = 0
        for imgs, targets in eval_loader:
            with torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
                preds = torch.argmax(model(imgs.to(device1)), dim=1).cpu()
                twin_preds = torch.argmax(twin_model(imgs.to(device2)), dim=1).cpu()
    
                acc += torch.sum(preds == targets)
                twin_acc += torch.sum(twin_preds == targets)

        acc = acc.item() / len(eval_loader.dataset)
        twin_acc = twin_acc.item() / len(eval_loader.dataset)
        logging |= {'Accuracy': acc, 'Twin Accuracy': twin_acc}
        wandb.log(logging)
        accuracies.append((acc, twin_acc, (acc+twin_acc)/2))

    if model_dir is not None:
        torch.save(model.state_dict(), os.path.join(model_dir, 'base.pth'))
        torch.save(twin_model.state_dict(), os.path.join(model_dir, 'mutual.pth'))
