import stumpy
import pickle
import numpy as np
import time
import sklearn.metrics as metrics

from pipelinetools import *
from multiprocessing import Pool
import multiprocessing

DIST_FUNC_POS = 'euclid_align_pos'
DIST_FUNC_NEG = 'euclid_align_neg'
FOLDER = '../'
TRACE_FNAMES = [FOLDER + fname for fname in ['merged_train_pos', 'merged_train_neg', 'merged_test_pos', 'merged_test_neg']]
SHAPELET_FNAME = FOLDER + 'shapelets'

def generate_names(set_type, tabs, index, num_shapelets=2):
    
    filenames_pos = make_name_list({
        'type':['pos'],
        'centroid_id':list(range(num_shapelets)),
        'tabs':[tabs],
        'dataset':[set_type],
        'index': [index]
    })
    filenames_neg = make_name_list({
        'type':['neg'],
        'centroid_id':list(range(num_shapelets)),
        'tabs':[tabs],
        'dataset':[set_type],
        'index': [index]
    })

    return filenames_pos, filenames_neg

def get_parameter_list(filenames_pos, filenames_neg, shapelets_pos, shapelets_neg, X_pos, X_neg, y):
    parameter_list = [] 
    
    for i in range(len(filenames_pos)):
        parameter_set = [
            filenames_pos[i],
            X_pos,
            y,
            shapelets_pos[i],
            DIST_FUNC_POS
        ]
        parameter_list.append(parameter_set)
    
    for i in range(len(filenames_neg)):
        parameter_set = [
            filenames_neg[i],
            X_neg,
            y,
            shapelets_neg[i],
            DIST_FUNC_NEG
        ]
        parameter_list.append(parameter_set)
    
    return parameter_list

def load_data():

    # load traces
    traces = []
    for file in TRACE_FNAMES:
        with open(file, 'rb') as f:
            traces.append(pickle.load(f))

    # load and split shapelets into pos/neg
    with open(SHAPELET_FNAME, 'rb') as f:
        shapelets = pickle.load(f)
    shapelets = {0: shapelets[0], 1:shapelets[1]}
    shapelets = (process_traces(shapelets, 'p'), process_traces(shapelets, 'n'))

    return tuple(traces) + shapelets

def chunk_traces(X_pos, X_neg, y, num_chunks, set_type):
    param_list = []
    k, m = divmod(len(X_pos), num_chunks)
    print("Approx Chunk Length: " + str(k))
    for i in range(num_chunks):
        idx = slice(i*k+min(i,m), (i+1)*k+min(i+1,m))
        fnames_pos, fnames_neg = generate_names(set_type=set_type, tabs='3', index = idx.start)
        
        chunk_pos = X_pos[idx]
        chunk_neg = X_neg[idx]
        chunk_y = y[idx]

        param_list_chunk = get_parameter_list(
            fnames_pos, fnames_neg, 
            shapelets_pos, shapelets_neg, 
            chunk_pos, chunk_neg, 
            chunk_y)
        
        param_list = param_list + param_list_chunk

    return param_list

def convert_for_stumpy(x_list):
    return [x.astype(np.float64) for x in x_list]

if __name__ == "__main__":
    #make_results_folder()
    train_pos, train_neg, test_pos, test_neg, shapelets_pos, shapelets_neg = load_data()

    X_train_pos, _ = traces_to_xy(train_pos)
    X_train_neg, y_train = traces_to_xy(train_neg)

    X_test_pos, _ = traces_to_xy(test_pos)
    X_test_neg, y_test = traces_to_xy(test_neg)

    for i in range(len(shapelets_pos)):
        shapelets_pos[i] = convert_for_stumpy(shapelets_pos[i])
        shapelets_neg[i] = convert_for_stumpy(shapelets_neg[i])

    X_train_pos = convert_for_stumpy(X_train_pos)
    X_train_neg = convert_for_stumpy(X_train_neg)

    X_test_pos = convert_for_stumpy(X_test_pos)
    X_test_neg = convert_for_stumpy(X_test_neg)

    print(X_train_pos[0].dtype)

    param_list = []

    num_cpus = multiprocessing.cpu_count()
    num_chunks = multiprocessing.cpu_count() // 4
    print("CPUs Available: " + str(num_cpus))
    print("Chunks: " + str(num_chunks))

    param_list = chunk_traces(X_train_pos, X_train_neg, y_train, num_chunks, 'train') + chunk_traces(X_test_pos, X_test_neg, y_test, num_chunks, 'test')
    print("Tasks to Complete: " + str(len(param_list)))

    with Pool(num_cpus) as p:
        p.map(compute_shapelet_distances_mp, param_list)
