import torch
import torch.nn.functional as F
import torchvision
import tqdm
import wandb
import recipe
import os

from models import PrunableModule

from train import *
from lth import *

from github_autoupdate import GithubAutoUpdater
from command_parser import get_parser

args = get_parser().parse_args()
print(args)

train_loader, eval_loader, model_fn, recipe, num_classes = recipe.training_recipe(args.dataset, args.network, args)

if not args.nosync:
    updater = GithubAutoUpdater()
    updater.run()

    wandkey = open('wandAPI').read().strip()
    wandb.login(key=wandkey)
    wandb.init(project='LucasMethod', config=vars(args) | {'GIT_HASH': updater.current_commit()})
    model_dir = f'./models/{wandb.run.name}'
    os.makedirs(model_dir)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.subcommand == 'find':
    lth(PrunableModule(model_fn()), recipe, train_loader, eval_loader, args.rate, args.iterations, args.amp, not args.nosync)

elif args.subcommand == 'train':
    train(PrunableModule(model_fn()), recipe, train_loader, eval_loader, args.amp, use_wandb = not args.nosync)

elif args.subcommand == 'tna':
    train_tna(PrunableModule(model_fn()), PrunableModule(model_fn()), recipe, train_loader, eval_loader, args.lamb, args.amp, use_wandb = not args.nosync)

# if 'single' in args.typ:
#     model = model_fn()
#     train_standard(model, recipe, train_loader, eval_loader, amp=args.amp, device=device, model_dir=model_dir)
# elif 'double' in args.typ:
#     if args.twin_model is None:
#         twin_model_fn = model_fn
#     else:
#         twin_model_fn = models.get_model(args.twin_model, num_classes, tiny = args.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet'])

#     model = model_fn()
#     twin_model = twin_model_fn()
#     if args.fixed_twin_loc is not None:
#         twin_weights = torch.load(os.path.join('models', args.fixed_twin_loc, 'base.pth'))
#         twin_model.load_state_dict(twin_weights)
#     train_tna(model, twin_model, recipe, train_loader, eval_loader, lamb = args.lamb, freeze_twin=args.freeze_twin, amp = args.amp, device = device, model_dir=model_dir, late_matching=args.late_matching)
    
# elif args.typ == 'mutual':
#     if args.twin_model is None:
#         twin_model_fn = model_fn
#     else:
#         twin_model_fn = models.get_model(args.twin_model, num_classes, tiny = args.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet'])
#     model = model_fn()
#     twin_model = twin_model_fn()
#     train_mutual(model, twin_model, recipe, train_loader, eval_loader, amp=args.amp, device=device, model_dir=model_dir)
    
