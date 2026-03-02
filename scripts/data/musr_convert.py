import jsonlines
import json

with jsonlines.open('train.jsonl', 'r') as reader:
    data = list(reader)

newdata = []
for d in data:
    nd = {'idx':len(newdata), 'question':d['prompt'], 'steps':[],'answer':'\\boxed{' + d['answer'] + '}'}
    newdata.append(nd)

with open('musr_train.json', 'w') as f:
    f.write(json.dumps(newdata))


with jsonlines.open('test.jsonl', 'r') as reader:
    data = list(reader)

newdata = []
for d in data:
    nd = {'idx':len(newdata), 'question':d['prompt'], 'steps':[],'answer':'\\boxed{' + d['answer'] + '}'}
    newdata.append(nd)

with open('musr_test.json', 'w') as f:
    f.write(json.dumps(newdata))



