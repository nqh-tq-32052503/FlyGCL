import json
import numpy as np
import os
import pandas as pd

def calculate_new_metrics(R_matrix):
    T = len(R_matrix)
    # A_last = (1/T) * sum(R_{T,i}) với i = 1..T
    # tức là trung bình hàng cuối của R_matrix
    last_row = R_matrix[-1]           # [R_{T,0}, R_{T,1}, ..., R_{T,T}]
    A_last = np.mean(last_row)        # (1/T) * sum(R_{T,i})

    # A_avg = (1/T) * sum(R_{i,i}) với i = 1..T  
    # tức là trung bình đường chéo chính
    diagonal = [R_matrix[i][i] for i in range(T)]   # [R_00, R_11, ..., R_TT]
    A_avg = np.mean(diagonal)         # (1/T) * sum(R_{i,i})
    f_vals = []
    for j in range(T):
        col_j = [R_matrix[i][j] for i in range(j, T)]  # cột j, từ hàng j..T
        max_col_j = np.max(col_j)                        # max(R_j)
        R_T_j = R_matrix[-1][j]                          # R_{T,j} = hàng cuối, cột j
        f_vals.append(max_col_j - R_T_j)
    F_last = np.mean(f_vals)
    if T > 1:
        bwt_vals = []
        for i in range(T - 1): # Duyệt qua các task cũ (không tính task cuối cùng)
            bwt_vals.append(R_matrix[T-1][i] - R_matrix[i][i])
        BWT = np.mean(bwt_vals)
    else:
        BWT = 0.0  # Không thể tính BWT nếu chỉ có 1 task
    
    return A_avg, A_last, F_last, BWT

def report_all_methods(all_R_matrices):    
    df = {
        "method" : [],
        "A_avg" : [],
        "A_last" : [],
        "F_last" : [],
        "BWT" : []
    }

    for method in all_R_matrices:
        R_matrix = all_R_matrices[method]
        A_avg, A_last, F_last, BWT = calculate_new_metrics(R_matrix)
        df["method"].append(method)
        df["A_avg"].append(A_avg)
        df["A_last"].append(A_last)
        df["F_last"].append(F_last)
        df["BWT"].append(BWT)

    return pd.DataFrame(df)

folder = "experiment_with_imagenet-r"
methods = os.listdir(folder)

all_R_matrices = {}
for method in methods:
    current_folder = folder + "/" + method
    files = sorted(os.listdir(current_folder))
    r = []
    for file in files:
        with open(current_folder + "/" + file, "r") as f:
            data = json.load(f)
            element = [d['avg_acc'] for d in data]
            r.append(element)
    all_R_matrices[method] = r