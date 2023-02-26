import pickle

def unpickle(file):
    with open(file, 'rb') as handle:
        dict = pickle.load(handle, encoding='bytes')
    return dict

def save_pickled(dict, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# This was used to separate the 5 data batches from
# the original cifar-10 into 500 separate batches.
# To achieve this, download the batches, then place them
# in a folder called cifar-10-batches inside utilities.

dict1 = unpickle("./test_batch")

full_data = [*dict1[b'data']]
full_labels = [*dict1[b'labels']]
full_filenames = [*dict1[b'filenames']]

for i in range(5):
    dump_filename = "test_batch_" + str(i+1)
    index_start = i*2000
    index_end = index_start + 2000
    new_dict = {
        'batch_label': "b'test batch " + str(i+1) + " of 500",
        'labels': full_labels[index_start:index_end],
        'data': full_data[index_start:index_end],
        'filenames': full_filenames[index_start:index_end]
    }
    save_pickled(new_dict, dump_filename)