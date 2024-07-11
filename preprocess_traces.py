import pickle
from pipelinetools import *
import multiprocessing

train_loc  = '../merged_train'
with open(train_loc, 'rb') as f:
    traces_train = pickle.load(f)

print("Processing Traces (train)")
train_pos = process_traces(traces_train, "p")
with open(train_loc + '_pos', 'wb') as f:
    pickle.dump(train_pos, f)

train_neg = process_traces(traces_train, "n")
with open(train_loc + '_neg', 'wb') as f:
    pickle.dump(train_neg, f)

test_loc  = '../merged_test'
with open(test_loc, 'rb') as f:
    traces_test = pickle.load(f)

print("Processing Traces (test)")
test_pos = process_traces(traces_test, "p") 
with open(test_loc + '_pos', 'wb') as f:
    pickle.dump(test_pos, f)

test_neg = process_traces(traces_test, "n")
with open(test_loc + '_neg', 'wb') as f:
    pickle.dump(test_neg, f)

print(multiprocessing.cpu_count())
make_results_folder()