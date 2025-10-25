import jsonlines
import json

splits = ["train", "val", "test"]

folder = 'ff_rl_data/'

for split in splits:
    with jsonlines.open(folder + split + '.jsonl') as reader:
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
            "question": item['prompt'],
            "answer": item['answer'],
            "steps": []
        })
    with open(folder + split + '.json', 'w') as f:
        json.dump(new_data, f)