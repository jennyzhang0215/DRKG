import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='../../Neural-LP/datasets',
                    help='the directory name for saving the converted datasets')
args = parser.parse_args()
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

entity_map_file = os.path.join('embed', 'entities.tsv')
relation_map_file = os.path.join('embed', 'relations.tsv')
entity_map = pd.read_csv(entity_map_file, sep='\t', names=['name', 'id'])
print('entity_map', entity_map)
relation_map = pd.read_csv(relation_map_file, sep='\t', names=['name', 'id'])
print('relation_map', relation_map)

entity_name = entity_map['name'].to_csv(os.path.join(args.save_dir, 'entities.txt'),
                                        sep='\t', header=False, index=False)
relation_name = relation_map['name'].to_csv(os.path.join(args.save_dir, 'relations.txt'),
                                            sep='\t', header=False, index=False)

### process kg triplets
drkg_file = 'drkg.tsv'
df = pd.read_csv(drkg_file, sep="\t")
triples = df.values.tolist()
num_triples = len(triples)
print('#triplets: {}'.format(num_triples))

seed = np.arange(num_triples)
np.random.shuffle(seed)
fact_cnt = int(num_triples * 0.8)
train_cnt = int(num_triples * 0.1)
valid_cnt = int(num_triples * 0.05)
fact_set = seed[ : fact_cnt].tolist()
train_set = seed[fact_cnt : fact_cnt + train_cnt].tolist()
valid_set = seed[fact_cnt + train_cnt : fact_cnt + train_cnt + valid_cnt].tolist()
test_set = seed[fact_cnt + train_cnt + valid_cnt : ].tolist()

with open(os.path.join(args.save_dir, "facts.txt"), 'w+') as f:
    for idx in fact_set:
        f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

with open(os.path.join(args.save_dir, "train.txt"), 'w+') as f:
    for idx in train_set:
        f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

with open(os.path.join(args.save_dir, "valid.txt"), 'w+') as f:
    for idx in valid_set:
        f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

with open(os.path.join(args.save_dir, "test.txt"), 'w+') as f:
    for idx in test_set:
        f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))
