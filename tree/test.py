import random
import argparse

from banker import Banker

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--loadpath', type=str, default='model', help='a folder with all the models')
arg_parser.add_argument('--testfile', type=str, default='data.xlsx', help='test input file')

parsed = arg_parser.parse_args()

args = {
    'loadpath': parsed.loadpath,
    'testfile': parsed.testfile,
}

banker = Banker(args)
banker.pickle_load(args['loadpath'])

data = banker.load_dicts(args['testfile'])

random.shuffle(data)
data = data[:200]
result = banker.predict(item['text'] for item in data)

cnt = 0
for d, r in zip(data, result):
    if d['type id'] == r:
        cnt += 1

print(cnt)
print(cnt/len(data))

while True:
    print(banker.predict([input('> ')])[0])

