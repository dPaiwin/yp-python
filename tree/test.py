import random
import argparse

from random_forest_manager import RandomForestManager

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--loadpath', type=str, default='model', help='a folder with all the models')
arg_parser.add_argument('--testfile', type=str, default='data.xlsx', help='test input file')

parsed = arg_parser.parse_args()

args = {
    'loadpath': parsed.loadpath,
    'testfile': parsed.testfile,
}

rand_forest_er = RandomForestManager()
rand_forest_er.pickle_load(args['loadpath'])

data = rand_forest_er.load_dicts(args['testfile'])
result = rand_forest_er.predict(item['text'] for item in data)

cnt = 0
for d, r in zip(data, result):
    if d['type id'] == r:
        cnt += 1

print(cnt)
print(cnt/len(data))

while True:
    print(rand_forest_er.predict([input('> ')])[0])

