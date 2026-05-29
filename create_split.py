import json

train = open('CAMUS_public/database_split/subgroup_training.txt').read().strip().split('\n')
val = open('CAMUS_public/database_split/subgroup_validation.txt').read().strip().split('\n')
test = open('CAMUS_public/database_split/subgroup_testing.txt').read().strip().split('\n')
data = {'train': train, 'val': val, 'test': test}
print(f'train: {len(train)}, val: {len(val)}, test: {len(test)}')
with open('CAMUS_public/camus_split.json', 'w') as f:
    json.dump(data, f)
print('Done')
