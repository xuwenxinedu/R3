import fire
import pandas as pd
import json
from tqdm import tqdm
import os

def merge(input_path, output_path, nn=8):
    data = []
    for i in tqdm(range(nn)):
        with open(os.path.join(input_path, f'{i}.json'), 'r') as f:
            data.extend(json.load(f))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(merge)
