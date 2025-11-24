import jsonlines
import json

splits = ["train", "val", "test"]

# folder = 'ff_rl_data/'
input_folder = '/mnt/disk/new_nrl_ncp/rl_data/'
folder = 'new_ncp_rl_data/'

for split in splits:
    with jsonlines.open(input_folder + split + '.jsonl') as reader:
        data = list(reader)
    
    # original format is prompt: '', answer: '', 
    # we want to convert it to the following format:
    # {
    #     "question": "",
    #     "answer": "",
    #     "steps": []
    # }
    new_data = []
    for item in data:
        new_data.append({
            "question": item['prompt'][-1]['content'],
            "answer": '',
            "steps": []
        })
    with open(folder + split + '.json', 'w') as f:
        json.dump(new_data, f)
