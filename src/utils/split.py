import fire
import os
import pandas as pd

def split(input_path, output_path, nn=8):
    df = pd.read_csv(input_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    for i in range(nn):
        start = i * df_len // nn
        end = (i+1) * df_len // nn
        df[start:end].to_csv(f'{output_path}/{i}.csv', index=True)

if __name__ == '__main__':
    fire.Fire(split)
