import json
import numpy as np
import os
import pandas as pd

folder = "experiment_with_imagenet-r"
methods = os.listdir(folder)

R_matrix = {}
for method in methods:
    current_folder = folder + "/" + method
    files = sorted(os.listdir(current_folder))
    r = []
    for file in files:
        with open(current_folder + "/" + file, "r") as f:
            data = json.load(f)
            element = [d['avg_acc'] for d in data]
            r.append(element)
    R_matrix[method] = r

df = {
    "method" : [],
    "A_avg" : [],
    "A_last" : []
}

for method in R_matrix:
    acc_triangle = R_matrix[method]
    A_avg = np.mean([acc_triangle[i][i] for i in range(len(acc_triangle))])
    A_last = np.mean([acc_triangle[i][-1] for i in range(len(acc_triangle))])
    df["method"].append(method)
    df["A_avg"].append(A_avg)
    df["A_last"].append(A_last)

pd.DataFrame(df)