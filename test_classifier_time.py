import stumpy
import pickle
import numpy as np
import time
import sklearn.metrics as metrics

from pipelinetools import *
from multiprocessing import Pool

def generate_names(set_type, tabs, num_shapelets=2):
    
    filenames_pos = make_name_list({
        'type':['pos'],
        'centroid_id':list(range(num_shapelets)),
        'tabs':[tabs],
        'dataset':[set_type]
    })
    filenames_neg = make_name_list({
        'type':['neg'],
        'centroid_id':list(range(num_shapelets)),
        'tabs':[tabs],
        'dataset':[set_type]
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
            "cbd"
        ]
        parameter_list.append(parameter_set)
    
    for i in range(len(filenames_neg)):
        parameter_set = [
            filenames_neg[i],
            X_neg,
            y,
            shapelets_neg[i],
            "cbd"
        ]
        parameter_list.append(parameter_set)
    
    return parameter_list

if __name__ == "__main__":

    train_pos_loc = '../merged_train_pos'
    train_neg_loc = '../merged_train_neg'

    test_pos_loc  = '../merged_test_pos'
    test_neg_loc  = '../merged_test_neg'

    with open(train_pos_loc, 'rb') as f:
        train_pos = pickle.load(f)
        
    with open(train_neg_loc, 'rb') as f:
        train_neg = pickle.load(f)
    
    with open(test_pos_loc, 'rb') as f:
        test_pos = pickle.load(f)

    with open(test_neg_loc, 'rb') as f:
        test_neg = pickle.load(f)

    shapelets_loc = '../shapelets'

    with open(shapelets_loc, 'rb') as f:
        shapelets = pickle.load(f)
        
    shapelets = {0: shapelets[0], 1:shapelets[1]}

    shapelets_pos = process_traces(shapelets, 'p')
    shapelets_neg = process_traces(shapelets, 'n')

    filenames_train_pos, filenames_train_neg = generate_names(set_type="train", tabs='3', num_shapelets=2)
    filenames_test_pos, filenames_test_neg = generate_names(set_type="test", tabs='3',num_shapelets=2)

    X_train_pos, y_train = traces_to_xy(train_pos)
    X_train_neg, y_train = traces_to_xy(train_neg)

    X_test_pos, y_test = traces_to_xy(test_pos)
    X_test_neg, y_test = traces_to_xy(test_neg)

    X_train_pos = X_train_pos[:1000]
    X_train_neg = X_train_neg[:1000]
    y_train = y_train[:1000]

    X_test_pos = X_test_pos[:1000]
    X_test_neg = X_test_neg[:1000]
    y_test = y_test[:1000]

    parameter_list_train = get_parameter_list(
        filenames_train_pos, filenames_train_neg, 
        shapelets_pos, shapelets_neg, 
        X_train_pos, X_train_neg, 
        y_train)

    parameter_list_test = get_parameter_list(
        filenames_test_pos, filenames_test_neg, 
        shapelets_pos, shapelets_neg, 
        X_test_pos, X_test_neg, 
        y_test)

    parameter_list = parameter_list_train + parameter_list_test

    with Pool(4) as p:
        p.map(compute_shapelet_distances_mp, parameter_list)