from openpyxl import load_workbook
import argparse
import pandas as pd
from tqdm import tqdm
import os
import torch
from openpyxl import load_workbook

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dir",
        action='store',
        required=True,
        dest='model_dir'
    )
    parser.add_argument(
        "-save",
        action='store',
        required=True,
        dest='save_name'
    )

    args = parser.parse_args()
    if args.model_dir[-1] != "/":
        args.model_dir += "/"

    pd.set_option('display.max_colwidth', None)

    losses = []
    models = []
    patches = []

    for model in tqdm(os.listdir(args.model_dir)):
        if not model.startswith("."):
            model_path = os.path.join(args.model_dir, model)
            print(model_path)
            curr_model = torch.load(model_path, map_location="cpu")
            models.append(model_path)
            losses.append(curr_model['validation_loss'])
            patches.append(curr_model['patches'])

    data = pd.DataFrame(
        {'model': models,
         'loss': losses,
         'patches': patches})

    data = data.sort_values(by=['loss'])
    print("Best Model :\n", data['model'].iloc[:5], "\n", data['loss'].iloc[:5], "\n", data['patches'].iloc[:5])

    if os.path.isfile(args.save_name):
        with pd.ExcelWriter(args.save_name, engine='openpyxl', mode='a') as writer:
            data.to_excel(writer, sheet_name=args.model_dir.split("/")[-2])

    else:
        with pd.ExcelWriter(args.save_name, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=args.model_dir.split("/")[-2])
