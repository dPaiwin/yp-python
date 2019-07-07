import argparse
import os
from os.path import normpath

from random_forest_manager import RandomForestManager


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--loadpath', default=None, type=str, help='path to a dataset to where faq and nofaq files are stored')
arg_parser.add_argument('--savepath', default='model', type=str, help='path to a model to be saved')
arg_parser.add_argument('--savepostfix', default='', type=str, help='subpath for savepath to save info stuff into')
arg_parser.add_argument('--trainfile', default='data.xlsx', type=str, help='a file with all train examples. columns are [0/1; text]')

arg_parsed = arg_parser.parse_args()
args = {
    'loadpath': arg_parsed.loadpath,
    'savepath': arg_parsed.savepath,
    'savepostfix': arg_parsed.savepostfix,
    'trainfile': arg_parsed.trainfile
}

args['savepath'] = normpath(args['savepath']) + args['savepostfix']
if not os.path.exists(args['savepath']):
    os.makedirs(args['savepath'])

rand_forest_er = RandomForestManager()
rand_forest_er.train(args['trainfile'])
rand_forest_er.pickle_save(args['savepath'])
