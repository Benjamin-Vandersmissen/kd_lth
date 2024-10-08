import torch
import wandb
import argparse
import tqdm
import os

import recipe
from models import PrunableModule
from utils import Step, evaluate, coreset
from torch.cuda.amp import GradScaler

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=str, required=True)
parser.add_argument('--ipc', type=int, nargs='+', required=True)

args = parser.parse_args()
print(args)

wandkey = open('wandAPI').read().strip()
wandb.login(key=wandkey)

api = wandb.Api()

config = api.run(f'Bvandersmissen/lth_kd/{args.run}').config
name = api.run(f'Bvandersmissen/lth_kd/{args.run}').name
run = wandb.init(project='lth_kd', id=args.run, resume=True)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
amp = True
mapping = None

train_loader, eval_loader, model_fn, recipe, num_classes = recipe.training_recipe(config['dataset'], config['network'], config)
full_train = train_loader.dataset

wandb.define_metric("Pruning Iterations")
# define which metrics will be plotted against it
for ipc in args.ipc:
    wandb.define_metric(f"Ticket@{ipc}IPC", step_metric="Pruning Iterations")
for i in tqdm.trange(config['iterations']+1):
    log = {}
    for ipc in args.ipc:
        train, _ = coreset(full_train, ipc)
        train_loader = torch.utils.data.DataLoader(train, num_workers = train_loader.num_workers, batch_size=train_loader.batch_size, shuffle=True)
    
        model = PrunableModule(model_fn())
        if i != 0:
            model.load_state_dict(torch.load(os.path.join('artifacts', name, f'pretrain.pth'), weights_only=True))
            model.mask = torch.load(os.path.join('artifacts', name, f'mask_{i}.pth'), weights_only=True)
        else:
            model.load_state_dict(torch.load(os.path.join('artifacts', name, f'init.pth'), weights_only=True))
            
        model = model.to(device)
        model.train()
    
        # print("Before Sparse Training:")
        # print(evaluate(model, eval_loader, amp))
        
        start = Step(len(train_loader))
        end = Step(len(train_loader), it=len(train_loader)*recipe.train_epochs())
        
        optim = recipe.optimizer(model)
        scheduler = recipe.lr_scheduler(optim, cur_ep=start.ep())
        criterion = torch.nn.CrossEntropyLoss()
        amp &= device.type == 'cuda'
    
        if amp:
            scaler = GradScaler()
    
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
                epoch_loss += loss.item()
    
                if start.it == end.it:
                    break
            scheduler.step()  # Epoch-wise scheduler.
    
        # print("After Sparse Training")
        # print(evaluate(model, eval_loader, amp))
        # print("++++++++++++++++++")
        log |= {f'Ticket@{ipc}IPC': evaluate(model, eval_loader, amp)}
    wandb.log(log)