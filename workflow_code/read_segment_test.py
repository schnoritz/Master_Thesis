
import os
import re
import pandas as pd
from tqdm import tqdm


def main():

    root_dir = "/mnt/qb/baumgartner/sgutwein84/segment_results"

    test_cases = [x for x in os.listdir(root_dir) if not x.startswith(".")]

    print(test_cases)

    tests = []
    for test_case in tqdm(test_cases):
        path = os.path.join(root_dir, test_case)
        segments = [x for x in os.listdir(path) if not x.startswith(".")]

        for segment in tqdm(segments):

            segment_path = os.path.join(path, segment)
            models = [x for x in os.listdir(segment_path) if not x.startswith(".")]

            for model in models:
                gamma_path = os.path.join(segment_path, model, "gamma.txt")

                with open(gamma_path) as fin:
                    lines = fin.readlines()
                    gamma = float(lines[4])
                    size = re.findall(r'\d+', lines[8])
                    size = float(".".join(size))

                gammas = {
                    "segment": segment,
                    "model": model,
                    "gamma": gamma,
                    "size": size
                }

                tests.append(gammas)

    print(tests)
    df = pd.DataFrame(tests)
    df.to_excel("/Users/simongutwein/Desktop/tests.xlsx")
    df.to_pickle("/Users/simongutwein/Desktop/tests.pkl")


if __name__ == "__main__":
    main()
