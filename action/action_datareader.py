import numpy as np
from threading import Thread

dataname = '../cross-' + args.split + '/train_ntus.npy'
labelname = '../cross-' + args.split + '/train_label.npy'
lenname = '../cross-' + args.split + '/train_len.npy'
training_data = np.load(dataname)
training_label = np.load(labelname)
training_len = np.load(lenname)
num_all_data = len(training_data)
training_eval_split = 0.95
num_training_data = int(num_all_data * training_eval_split)

all_data_array = np.arange(num_all_data)
np.random.shuffle(all_data_array)

training_data_array = all_data_array[:training_eval_split]
eval_data_array = all_data_array[training_eval_split:]

dataname = '../cross-' + args.split + '/test_ntus.npy'
labelname = '../cross-' + args.split + '/test_label.npy'
lenname = '../cross-' + args.split + '/test_len.npy'
test_data = np.load(dataname)
test_label = np.load(labelname)
test_len = np.load(lenname)

num_test_data = len(test_data)
test_data_array = np.arange(num_test_data)
np.random.shuffle(test_data_array)


class BatchThread(object):
    def __init__(self, result, batch_size, seq_len, training):
        self.result = result
        self.batch_size = batch_size
        self.seq_len = seq_len
        if training == 'train':
            self.data_array = training_data_array
            self.data = training_data
            self.label = training_label
            self.len = training_len
        elif training == 'eval':
            self.data_array = eval_data_array
            self.data = training_data
            self.label = training_label
            self.len = training_len
        elif training == 'test':
            self.data_array = test_data_array
            self.data = test_data
            self.label = test_label
            self.len = test_len

    def __call__(self):
        temp_label = []
        batch_data = []
        temp_index = []
        counter = 0
        for i in range(self.batch_size):
            counter += 1
            if counter == len(self.data_array):
                counter = 0
                np.random.shuffle(self.data_array)
            index = self.data_array[counter]

            label = self.label[index]
            temp_label.append(np.int32(label))
            temp_index.append(np.int32(index))
            dataset = self.data[index]
            data_len = self.len[index]

            sample = np.zeros(tuple((self.seq_len,) + dataset.shape[1:]))
            seg_len = data_len // self.seq_len
            if seg_len == 1 and data_len > self.seq_len:
                start_idx = np.random.randint(data_len - self.seq_len)
                sample = dataset[start_idx:start_idx + self.seq_len]
            elif data_len <= self.seq_len:
                start_idx = np.random.randint(max(self.seq_len - data_len, int(0.25 * self.seq_len)))
                end_idx = min(self.seq_len, start_idx + data_len)
                data_start_idx = 0
                data_end_idx = data_len
                if start_idx + data_len > self.seq_len:
                    data_start_idx = np.random.randint(data_start_idx + data_len - self.seq_len)
                    data_end_idx = data_start_idx + self.seq_len - start_idx
                sample[start_idx:end_idx] = dataset[data_start_idx:data_end_idx]
            else:
                for i in range(self.seq_len):
                    if i == self.seq_len - 1:
                        j = seg_len * i + np.random.randint(data_len - seg_len * (self.seq_len - 1))
                    else:
                        j = seg_len * i + np.random.randint(seg_len)
                    sample[i] = dataset[j]
            batch_data.append(sample)

        self.result['data'] = np.asarray(batch_data, dtype=np.float32)
        self.result['label'] = np.asarray(temp_label, dtype=np.int32)
        self.result['index'] = np.asarray(temp_index, dtype=np.int32)


class DataReader(object):
    def __init__(self, batch_size, seq_len, training):
        self.batch_size = batch_size
        self.thread_result = {}
        self.thread = None
        self.training = training

        self.batch = BatchThread(self.thread_result, self.batch_size, seq_len, training)

        self.dispatch_worker()
        self.join_worker()

    def get_batch(self):
        if self.thread is not None:
            self.join_worker()
        batch_data = self.thread_result['data']
        batch_label = self.thread_result['label']
        batch_index = self.thread_result['index']
        self.dispatch_worker()
        if self.training == 'test':
            return batch_data, batch_label, batch_index
        else:
            return batch_data, batch_label

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def get_data_size(self):
        if self.training == 'train':
            return len(training_data_array)
        elif self.training == 'eval':
            return len(eval_data_array)
        elif self.training == 'test':
            return len(test_data_array)
