import pickle
from pipelinetools import *
import multiprocessing

train_loc  = '../merged_train'
with open(train_loc, 'rb') as f:
    traces_train = pickle.load(f)

print("Processing Traces (train)")
train_pos = process_traces(traces_train, "p")
train_neg = process_traces(traces_train, "n")

with open(train_loc + '_pos', 'wb') as f:
    pickle.dump(train_pos)

with open(train_loc + '_neg', 'wb') as f:
    pickle.dump(train_neg)

test_loc  = '../merged_test'
with open(test_loc, 'rb') as f:
    traces_test = pickle.load(f)

print("Processing Traces (test)")
test_pos = process_traces(traces_test, "p")
test_neg = process_traces(traces_test, "n")
    
with open(test_loc + '_pos', 'wb') as f:
    pickle.dump(test_pos)

with open(test_loc + '_neg', 'wb') as f:
    pickle.dump(test_neg)

print(multiprocessing.cpu_count())
make_results_folder()