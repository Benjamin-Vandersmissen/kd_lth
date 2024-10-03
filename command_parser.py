import argparse
from models import all_models
from datasets import all_datasets
import recipe
from train import train_tna, train, train_mutual

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    
    subparser_find(subparsers.add_parser('find'))
    subparser_train(subparsers.add_parser('train'))
    subparser_tna(subparsers.add_parser('tna'))
    return parser

def overlapping_arguments(parser):
    parser.add_argument('--network', type=str, required=True, choices=all_models)
    parser.add_argument('--dataset', type=str, required=True, choices=all_datasets)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--nosync', action='store_true')
    parser = recipe.TrainingRecipe.add_to_parser(parser)

def subparser_find(parser):
    parser.add_argument('--rate', type=float, default=0.2)
    parser.add_argument('--iterations', type=int, default=10)
    overlapping_arguments(parser)

def subparser_train(parser):
    overlapping_arguments(parser)

def subparser_tna(parser):
    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--twin', type=str, required=True)
    parser.add_argument('--lamb', type=float, default=0.005)
    overlapping_arguments(parser)
    # TODO: either convert to different arguments e.g. train -> standard, TNA, DML, KD and have subparsers for every command, or have overlapping arguments.
        