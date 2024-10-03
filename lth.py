from train import train
from utils import Step, evaluate
from recipe import training_recipe

import torch
import copy
import os
import time
import math

def rewind(full_network: torch.nn.Module, initial_network: dict):
    with torch.no_grad():
        full_network.load_state_dict(initial_network)
    return full_network


def lth(network, recipe, train_loader, eval_loader, pruning_rate, pruning_iterations, amp, use_wandb): 
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # Pretraining.
    start = Step(len(train_loader), 0)
    pretrain_end = Step(len(train_loader), recipe.pretrain_its)
    train(network, recipe, train_loader, eval_loader, start=start, end=pretrain_end, device=device, amp=amp, use_wandb=use_wandb)
    # torch.save(network.state_dict(), os.path.join(args.output_dir, 'pretrained.pth'))  # TODO: choose correct location, depending on whether we want to save.  
    rewind_weights = {key: copy.deepcopy(val.cpu()) for key, val in network.state_dict().items()}
    
    for it in range(pruning_iterations+1):
        ##
        # (it > 0) : Prune, rewind, reduced train with coreset
        ##
        if it > 0:
            network.prune(pruning_rate, 'full')
            network = rewind(network, rewind_weights).to(device)
            start.it = recipe.pretrain_its # Reset step to pretraining.
        
        ##
        # Mask Search training w/ full dataset. (possibly starting from a partially trained checkpoint)
        ##
        stime = time.time()
        train(network, recipe, train_loader, eval_loader, start=start, device=device, amp=amp, use_wandb=use_wandb)
        # torch.save(trained_network.state_dict(), os.path.join(args.output_dir, f'trained_ticket_{it}.pth'))
        # torch.save(trained_network.mask, os.path.join(args.output_dir, f'mask_{it}.pth'))
        duration = time.time() - stime