# -*- coding: utf-8 -*-
"""Functions and classes that make dataset easy to handle.

Load data using load_data and use DataGenerator class
 for creating data generators for training and validation.

Labels:
    For healthy vs. MDD task:
        Healthy: 0   ---   MDD: 1

    For responder vs. non-responder task:
        Responder: 1   ---   NonResponder: 0
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os
import numpy as np

from tqdm import tqdm


class DataLoader:

    def __init__(self,
                 data_dir,
                 task,
                 data_mode='cross_subject',
                 sampling_rate=256,
                 instance_duration=None,
                 instance_overlap=None):
        assert task in ('rnr', 'hmdd'), "task must be one of {'rnr', 'hmdd'}"
        assert data_mode in ('cross_subject', 'balanced'), "data_mode must be one of {'cross_subject', 'balanced'}"

        self.data_dir = data_dir
        self.task = task
        self.data_mode = data_mode
        self.sampling_rate = sampling_rate
        self.instance_duration = instance_duration
        self.instance_overlap = instance_overlap

    def load_data(self):
        """Loads data according to given data_mode.

        Normalizing the data instances isn't applied on this stage. Make sure to normalize the instances when
         feeding to model, i.e. on data generators.
        """

        if self.data_mode == 'balanced':
            assert isinstance(self.instance_duration, int) and isinstance(self.instance_overlap, int),\
                "make sure to et instance_duration and instance_overlap arguments."
            data_files, raw_labels = self._correct_data()

            data = list()
            labels = list()

            print('\nLoading data ...\n')
            with tqdm(total=len(data_files)) as pbar:
                for label, file_name in zip(raw_labels, data_files):
                    file_path = os.path.join(self.data_dir, file_name)
                    arr = np.load(file_path)
                    instances = self._generate_instances(arr)
                    data.extend(instances)
                    labels.extend([label] * len(instances))
                    pbar.update(1)

            data = np.array(data)
            labels = np.array(labels)

        else:
            data_files, raw_labels = self._correct_data()

            data = list()
            labels = list()

            print('\nLoading data ...\n')
            with tqdm(total=len(data_files)) as pbar:
                for label, file_name in zip(raw_labels, data_files):
                    file_path = os.path.join(self.data_dir, file_name)
                    arr = np.load(file_path)
                    data.append(arr.T)
                    labels.append(label)
                    pbar.update(1)
        return data, labels

    def _correct_data(self):
        data_files = [i for i in os.listdir(self.data_dir) if i.endswith('.npy')]
        labels = [self._label_map(i) for i in data_files]

        if self.task == 'hmdd':
            labels = [0 if label == -1 else 1 for label in labels]
        else:
            data_files = [file for i, file in enumerate(data_files) if labels[i] > -1]
            labels = [l for l in labels if l > -1]

        return data_files, labels

    def _generate_instances(self, arr):
        sample_time_steps = self.instance_duration * self.sampling_rate
        overlap_time_steps = self.instance_overlap * self.sampling_rate
        start_steps = sample_time_steps - overlap_time_steps

        start_indices = np.array([i for i in range(0, arr.shape[1] - sample_time_steps, start_steps)])
        end_indices = start_indices + sample_time_steps
        indices = list(zip(start_indices, end_indices))

        channels = arr.shape[0]
        instances = np.zeros((len(indices), sample_time_steps, channels))
        for ind, (i, j) in enumerate(indices):
            instance = arr[:, i: j]
            # instance = (instance - instance.mean()) / instance.std()
            instances[ind, :, :] = instance.T
        return instances

    @staticmethod
    def _label_map(file_name):
        label = file_name.split('.')[0].split('_')[-1]
        if label == 'h':
            return -1
        elif label == 'r':
            return 1
        elif label == 'nr':
            return 0
        else:
            raise Exception("File label is'nt in (h, r, nr): {}".format(file_name))


class Splitter:

    def __init__(self, test_size):
        self.test_size = test_size

    def balanced_split(self, labels):
        train_c0, val_c0 = self._split_class(labels, 0)
        train_c1, val_c1 = self._split_class(labels, 1)

        train_ind = np.concatenate([train_c0, train_c1])
        val_ind = np.concatenate([val_c0, val_c1])

        np.random.shuffle(train_ind)
        np.random.shuffle(val_ind)
        return train_ind, val_ind

    def cross_subject_split(self, data, labels):
        train_c0, test_c0 = self._split_class(np.array(labels), 0)
        train_c1, test_c1 = self._split_class(np.array(labels), 1)

        train_ind = np.concatenate([train_c0, train_c1])
        test_ind = np.concatenate([test_c0, test_c1])

        np.random.shuffle(train_ind)
        np.random.shuffle(test_ind)

        train_data = [data[i] for i in train_ind]
        train_labels = [labels[i] for i in train_ind]
        test_data = [data[i] for i in test_ind]
        test_labels = [labels[i] for i in test_ind]
        return train_data, train_labels, test_data, test_labels

    def within_subject_split(self, data, labels):
        train_data = list()
        test_data = list()

        print('\nSplitting data into train and test subsets ...\n')
        with tqdm(total=len(data)) as pbar:
            for subject in data:
                sh = subject.shape
                test_start = int(sh[0] * (1 - self.test_size))
                train_data.append(subject[0: test_start, :])
                test_data.append(subject[test_start:, :])
                pbar.update(1)
        return train_data, labels.copy(), test_data, labels.copy()

    def _split_class(self, labels, c):
        indices = np.where(labels == c)[0]
        class_size = len(indices)
        test_len = int(class_size * self.test_size)
        train_len = class_size - test_len
        np.random.shuffle(indices)
        return indices[: train_len], indices[train_len:]


class Generator:

    def __init__(self,
                 is_fixed,
                 is_train):
        assert isinstance(is_fixed, bool), "fixed_len is a boolean variable."
        self.is_fixed = is_fixed
        self.is_train = is_train

    def get_generator(self,
                      data,
                      labels,
                      indxs):
        print('Not implemented.')


class FixedLenGenerator(Generator):

    def __init__(self,
                 batch_size,
                 duration,
                 overlap,
                 sampling_rate,
                 is_train,
                 normalization='channel'):
        super().__init__(True,
                         is_train)
        assert normalization in ('channel', 'instance', 'time'),\
            "passed vale for normalization arg should be one of ('channel', 'instance', 'time')"
        self.batch_size = batch_size
        self.duration = duration
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.belonged_to_subject = list()
        self.normalization = normalization

    def get_generator(self,
                      data,
                      labels,
                      indxs=None):
        if indxs is None:
            x, y = self._generate_data_instances(data, labels)
            if self.is_train:
                print('   training instances: ', len(x))
            else:
                print('   test instances:     ', len(x))
            gen = self._generator(x, y)
            n_iter = len(x) // self.batch_size
        else:
            gen = self._generator(data, labels, indxs)
            n_iter = len(indxs) // self.batch_size
        return gen, n_iter

    def _generate_data_instances(self, data, labels):
        x = list()
        y = list()

        for ind, (d, l) in enumerate(zip(data, labels)):
            instances = self._generate_instances(d)
            n_instances = len(instances)
            x.extend(instances)
            y.extend([l] * n_instances)
            if not self.is_train:
                self.belonged_to_subject.extend([ind] * n_instances)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def _generate_instances(self, arr):
        sample_time_steps = int(self.duration * self.sampling_rate)
        overlap_time_steps = int(self.overlap * self.sampling_rate)
        start_steps = sample_time_steps - overlap_time_steps

        tsteps, channels = arr.shape

        start_indices = np.array([i for i in range(0, tsteps - sample_time_steps, start_steps)])
        end_indices = start_indices + sample_time_steps
        indices = list(zip(start_indices, end_indices))

        # instances = np.zeros((len(indices), sample_time_steps, channels))
        instances = list()
        for ind, (i, j) in enumerate(indices):
            instance = arr[i: j, :]
            if self.normalization == 'instance':
                instance = (instance - instance.mean()) / (instance.std() + 0.0001)
            elif self.normalization == 'channel':
                ch_wise_mean = instance.mean(axis=0, keepdims=True)
                ch_wise_std = instance.std(axis=0, keepdims=True)
                if np.any(ch_wise_std > 60) or np.any(ch_wise_std < 1):
                    continue
                instance = (instance - ch_wise_mean) / (ch_wise_std + 0.0001)
            else:
                time_wise_mean = instance.mean(axis=1, keepdims=True)
                instance = instance - time_wise_mean
            instances.append(instance)
        return instances

    def _generator(self,
                   data,
                   labels,
                   indxs=None):
        """Yields a batch of data and labels in each iteration."""

        if indxs is None:
            indxs = list(range(len(data)))
        n_instances = len(indxs)
        start_indx = list(range(0, n_instances, self.batch_size))
        end_indx = start_indx[1:]
        start_indx = start_indx[: -1]
        start_end = list(zip(start_indx, end_indx))
        while True:
            if self.is_train:
                np.random.shuffle(indxs)
            for s, e in start_end:
                ind = indxs[s: e]
                x_batch = data[ind]
                y_batch = labels[ind]
                yield x_batch, y_batch


class VarLenGenerator(Generator):

    def __init__(self,
                 min_duration,
                 max_duration,
                 iter_per_group,
                 sampling_rate,
                 is_train,
                 normalization='channel'):
        # Note: Don't use this generator as test set generator yet.
        super().__init__(False,
                         is_train)
        assert normalization in ('channel', 'instance', 'time'), \
            "passed vale for normalization arg should be one of " \
            "('channel', 'instance', 'time')"
        self.normalization = normalization
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.iter_per_group = iter_per_group
        self.sampling_rate = sampling_rate
        self.is_train = is_train
        self.belonged_to_subject = list()
        # Todo: Add each data point's parent subject id, just like FixedGenerator

    def get_generator(self, data, labels, indxs=None):
        gen = self._varsize_data_generator(data, labels)
        n_iter = self.iter_per_group * (self.max_duration - self.min_duration + 1)
        return gen, n_iter

    def _get_varsize_data(self, data, labels):
        data_dict = {k: {'data': list(),
                         'labels': list()} for k in range(self.min_duration,
                                                          self.max_duration + 1)}

        for subject, label in zip(data, labels):
            subject_data = self._generate_varsize_data(subject)
            for duration in subject_data.keys():
                duration_data = subject_data[duration]
                data_dict[duration]['data'].extend(duration_data)
                data_dict[duration]['labels'].extend([label] * len(duration_data))
        for duration in range(self.min_duration, self.max_duration + 1):
            ind = np.arange(len(data_dict[duration]['data']))
            np.random.shuffle(ind)
            d_arr = np.array(data_dict[duration]['data'])
            d_label = np.array(data_dict[duration]['labels'])

            data_dict[duration]['data'] = d_arr[ind]
            data_dict[duration]['labels'] = d_label[ind]
        return data_dict

    def _generate_varsize_data(self, subject_data):
        choices = range(self.min_duration, self.max_duration + 1)
        s_dict = {i: list() for i in choices}
        cursor = 0
        tsteps = subject_data.shape[0]
        while cursor < tsteps - (self.min_duration * self.sampling_rate):
            duration = np.random.choice(choices)
            end_ind = cursor + duration * self.sampling_rate
            if end_ind > tsteps:
                break
            sub_array = subject_data[cursor: end_ind, :]

            if self.normalization == 'instance':
                sub_array = (sub_array - sub_array.mean()) / (sub_array.std() + 0.00001)
            elif self.normalization == 'channel':
                ch_wise_mean = sub_array.mean(axis=0, keepdims=True)
                ch_wise_std = sub_array.std(axis=0, keepdims=True)
                if np.any(ch_wise_std > 80) or np.any(ch_wise_std < 1):
                    continue
                sub_array = (sub_array - ch_wise_mean) / (ch_wise_std + 0.00001)
            else:
                time_wise_mean = sub_array.mean(axis=1, keepdims=True)
                sub_array = sub_array - time_wise_mean

            # ch_wise_mean = sub_array.mean(axis=0, keepdims=True)
            # ch_wise_std = sub_array.std(axis=0, keepdims=True)
            # if np.any(ch_wise_std > 80) or np.any(ch_wise_std < 1):
            #     continue
            # sub_array = (sub_array - ch_wise_mean) / (ch_wise_std + 0.00001)
            s_dict[duration].append(sub_array)
            cursor = end_ind
        return s_dict

    def _get_group_indices(self, data_dict):
        indices = dict()
        for duration in data_dict.keys():
            n_samples = len(data_dict[duration]['data'])
            per_iter = n_samples // self.iter_per_group
            start = list(range(0, n_samples - 1, per_iter))
            end = start[1:]
            end.append(n_samples - 1)
            indices[duration] = np.array(list(zip(start, end)))
        return indices

    # def _varsize_test_data_gen(self, data, labels):
    #     data_dict = self._get_varsize_data(data, labels)
    #     group_indices = self._get_group_indices(data_dict)
    #     groups = list(data_dict.keys())
    #     while True:
    #         for i in range(self.iter_per_group):
    #             for j in groups:
    #                 ind = group_indices[j][i]
    #                 x = data_dict[j]['data'][ind[0]: ind[1]]
    #                 batch_mean = x.mean(axis=(1, 2), keepdims=True)
    #                 batch_std = x.std(axis=(1, 2), keepdims=True, ddof=1)
    #                 if not np.all(batch_std):
    #                     batch_std = np.where(batch_std > 0, batch_std, 1)
    #                 x_batch = (x - batch_mean) / batch_std
    #                 y_batch = data_dict[j]['labels'][ind[0]: ind[1]]
    #                 yield x_batch, y_batch

    def _varsize_data_generator(self, data, labels):
        while True:
            data_dict = self._get_varsize_data(data, labels)
            group_indices = self._get_group_indices(data_dict)
            groups = list(data_dict.keys())
            for i in range(self.iter_per_group):
                np.random.shuffle(groups)
                for j in groups:
                    ind = group_indices[j][i]
                    # x = data_dict[j]['data'][ind[0]: ind[1]]
                    # batch_mean = x.mean(axis=(1, 2), keepdims=True)
                    # batch_std = x.std(axis=(1, 2), keepdims=True, ddof=1)
                    # if not np.all(batch_std):
                    #     batch_std = np.where(batch_std > 0, batch_std, 1)
                    # x_batch = (x - batch_mean) / batch_std
                    x_batch = data_dict[j]['data'][ind[0]: ind[1]]
                    y_batch = data_dict[j]['labels'][ind[0]: ind[1]]
                    yield x_batch, y_batch