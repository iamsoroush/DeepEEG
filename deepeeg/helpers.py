# -*- coding: utf-8 -*-
"""Helper functions."""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os
import pickle
from itertools import combinations
from time import sleep

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, mean_squared_error
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

from . import keras

plt.style.use('seaborn-darkgrid')


class CrossValidator:

    def __init__(self,
                 task,
                 data_mode,
                 main_res_dir,
                 model_name,
                 epochs,
                 train_generator,
                 test_generator,
                 t,
                 k,
                 channel_drop=False,
                 np_random_state=71,
                 use_early_stopping_callback=True):
        assert task in ('rnr', 'hmdd'), "task must be one of {'rnr', 'hmdd'}"
        assert data_mode in ('cross_subject', 'balanced'), "data_mode must be one of {'cross_subject', 'balanced'}"

        self.task = task
        self.data_mode = data_mode
        self.main_res_dir = main_res_dir
        self.model_name = model_name
        self.epochs = epochs
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.t = t
        self.k = k
        self.channel_drop = channel_drop
        self.np_random_state = np_random_state
        self.use_early_stopping_callback = use_early_stopping_callback
        self.cv_dir = None
        self.indices_path = None
        self.scores_path = None
        self.rounds_file_names = None
        self.rounds_file_paths = None
        self._initialize()

    def _initialize(self):
        if not self.train_generator.is_fixed:
            train_gen_type = 'var'
            tr_pr_1 = self.train_generator.min_duration
            tr_pr_2 = self.train_generator.max_duration
        else:
            train_gen_type = 'fixed'
            tr_pr_1 = self.train_generator.duration
            tr_pr_2 = self.train_generator.overlap
        if not self.test_generator.is_fixed:
            test_gen_type = 'var'
            te_pr_1 = self.test_generator.min_duration
            te_pr_2 = self.test_generator.max_duration
        else:
            test_gen_type = 'fixed'
            te_pr_1 = self.test_generator.duration
            te_pr_2 = self.test_generator.overlap

        train_data_prefix = train_gen_type + str(tr_pr_1) + str(tr_pr_2)
        test_data_prefix = test_gen_type + str(te_pr_1) + str(te_pr_2)

        self.cv_dir = os.path.join(self.main_res_dir, '{}_{}'.format(self.data_mode, self.task))
        if not os.path.exists(self.cv_dir):
            os.mkdir(self.cv_dir)

        unique_identifier = '{}time-{}fold-{}epochs-tr_{}-te_{}'.format(self.t,
                                                                        self.k,
                                                                        self.epochs,
                                                                        train_data_prefix,
                                                                        test_data_prefix)
        indices_filename = 'train_test_indices-{}.pkl'.format(unique_identifier)
        self.indices_path = os.path.join(self.cv_dir, indices_filename)

        scores_filename = '{}-{}.npy'.format(self.model_name, unique_identifier)
        self.scores_path = os.path.join(self.cv_dir, scores_filename)
        template = '{}-time{}-fold{}-{}epochs-tr_{}-te_{}.npy'
        self.rounds_file_names = [template.format(self.model_name,
                                                  i + 1,
                                                  j + 1,
                                                  self.epochs,
                                                  train_data_prefix,
                                                  test_data_prefix) for i in range(self.t) for j in range(self.k)]
        self.rounds_file_paths = [os.path.join(self.cv_dir, file_name) for file_name in self.rounds_file_names]
        return

    def do_cv(self,
              model_obj,
              data,
              labels):
        if os.path.exists(self.scores_path):
            print('Final scores already exists.')
            final_scores = np.load(self.scores_path, allow_pickle=True)
        else:
            train_indices, test_indices = self._get_train_test_indices(data, labels)
            dir_file_names = os.listdir(self.cv_dir)
            for i in range(self.t):
                print('time {}/{}:'.format(i + 1, self.t))
                for j in range(self.k):
                    print(' step {}/{} ...'.format(j + 1, self.k))
                    ind = int(i * self.k + j)
                    file_name = self.rounds_file_names[ind]
                    file_path = self.rounds_file_paths[ind]
                    if file_name not in dir_file_names:
                        train_ind = train_indices[i][j]
                        test_ind = test_indices[i][j]
                        scores = self._do_train_eval(model_obj,
                                                     data,
                                                     labels,
                                                     train_ind,
                                                     test_ind)
                        np.save(file_path, scores)
            final_scores = self._generate_final_scores()
        self.plot_channel_drop_roc()
        self.plot_scores()
        self.plot_subject_wise_scores()
        return final_scores

    def plot_scores(self, dpi=80):

        def plot_on_ax(scores,  ax, model_name, phase):
            x_coord = 0.8
            y_coord = 0.02
            keys = ['Accuracy', 'F1-Score', 'Sensitivity', 'Specificity']
            for key, values in zip(keys, scores.T):
                line_width = 1
                alpha = 0.6
                if key == 'Accuracy':
                    line_width = 2
                    alpha = 0.8
                    ax.plot(values, linewidth=line_width, marker='o', alpha=alpha)
                elif key == 'Loss':
                    pass
                else:
                    ax.plot(values, linewidth=line_width, alpha=alpha)
                mean = values.mean()
                std = values.std(ddof=1)
                ax.text(x_coord, y_coord, '{}: {:2.3f} +- {:2.3f}'.format(key,
                                                                          mean,
                                                                          std),
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes)
                y_coord += 0.03
                keys.append(key)

            ax.legend(keys, loc='lower left')
            ax.set_title(model_name + ' - ' + phase)
            ax.set_xlabel('# Round')
            ax.set_ylabel('Score')

        if not os.path.exists(self.scores_path):
            print('Final scores does not exist.')
            return

        test_scores = np.array(list(np.load(self.scores_path, allow_pickle=True)[:, 0]))
        train_scores = np.array(list(np.load(self.scores_path, allow_pickle=True)[:, 3]))

        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(20, 16), dpi=dpi)

        plot_on_ax(test_scores, axes[0], self.model_name, 'Test')
        plot_on_ax(train_scores, axes[1], self.model_name, 'Train')

        # x_coord = 0.8
        # y_coord = 0.02
        # for key, values in zip(keys, scores.T):
        #     linewidth = 1
        #     alpha = 0.6
        #     if key == 'Accuracy':
        #         linewidth = 2
        #         alpha = 0.8
        #         ax.plot(values, linewidth=linewidth, marker='o', alpha=alpha)
        #     elif key == 'Loss':
        #         pass
        #     else:
        #         ax.plot(values, linewidth=linewidth, alpha=alpha)
        #     mean = values.mean()
        #     std = values.std(ddof=1)
        #     ax.text(x_coord, y_coord, '{}: {:2.3f} +- {:2.3f}'.format(key,
        #                                                               mean,
        #                                                               std),
        #             verticalalignment='bottom', horizontalalignment='left',
        #             transform=ax.transAxes)
        #     y_coord += 0.03
        #     keys.append(key)
        #
        # ax.legend(keys[1:], loc='lower left')
        # ax.set_title(self.model_name)
        # ax.set_xlabel('# Round')
        # # ax.set_xticks(range(1, t * k + 1),  direction='vertical')
        # ax.set_ylabel('Score')
        # ax.set_ylim(max(0, min_score - 0.2), 1)
        plot_name = '{}.jpg'.format(os.path.basename(self.scores_path).split('.')[0])
        path_to_save = os.path.join(os.path.dirname(self.scores_path), plot_name)
        fig.savefig(path_to_save)

    def plot_subject_wise_scores(self, dpi=80):
        if not os.path.exists(self.scores_path):
            print('Final scores does not exist.')
            return
        scores = np.array(list(np.load(self.scores_path, allow_pickle=True)[:, 1]))
        tns = scores[:, 0]
        fps = scores[:, 1]
        fns = scores[:, 2]
        tps = scores[:, 3]
        acc_vector = (tps + tns) / (tps + fns + fps + tns)
        precision_vector = tps / (tps + fps + 0.001)
        rec_vector = tps / (tps + fns + 0.001)
        spec_vector = tns / (tns + fps + 0.001)
        f_score_vector = 2 * (precision_vector * rec_vector) / (precision_vector + rec_vector + 0.0001)

        keys = ['Accuracy', 'F1-Score', 'Sensitivity', 'Specificity']
        fig, ax = plt.subplots(figsize=(20, 8), dpi=dpi)

        x_coord = 0.8
        y_coord = 0.02
        for key, values in zip(keys, [acc_vector, f_score_vector, rec_vector, spec_vector]):
            line_width = 1
            alpha = 0.6
            if key == 'Accuracy':
                line_width = 2
                alpha = 0.8
                ax.plot(values, linewidth=line_width, marker='o', alpha=alpha)
            else:
                ax.plot(values, linewidth=line_width, alpha=alpha)
            mean = values.mean()
            std = values.std(ddof=1)
            ax.text(x_coord, y_coord, '{}: {:2.3f} +- {:2.3f}'.format(key,
                                                                      mean,
                                                                      std),
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes)
            y_coord += 0.03
            keys.append(key)

        ax.legend(keys, loc='lower left')
        ax.set_title(self.model_name)
        ax.set_xlabel('# Round')
        # ax.set_xticks(range(1, t * k + 1),  direction='vertical')
        ax.set_ylabel('Score')
        # ax.set_ylim(max(0, min_score - 0.2), 1)
        plot_name = '{}_subject-wise-scores.jpg'.format(os.path.basename(self.scores_path).split('.')[0])
        path_to_save = os.path.join(os.path.dirname(self.scores_path), plot_name)
        fig.savefig(path_to_save)

    def plot_channel_drop_roc(self):
        if not os.path.exists(self.scores_path):
            print('Final scores does not exist.')
            return
        scores = np.array(list(np.load(self.scores_path, allow_pickle=True)[:, 2]))
        fprs = scores[:, 0]
        tprs = scores[:, 1]

        self._roc_vs_channel_drop(fprs,
                                  tprs)

    def _roc_vs_channel_drop(self,
                             fprs,
                             tprs):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
        for i, ax in enumerate(axes.flatten()):
            fp = np.array([item[i] for item in fprs])
            tp = np.array([item[i] for item in tprs])
            self._draw_roc_curve(fp, tp, ax)
            ax.set_title('# Channels Dropped: {}'.format(i ** 2))

        fig.suptitle('Model: {}, Task: "{}"'.format(self.model_name, self.task))

        plot_name = '{}_channel-drop.jpg'.format(os.path.basename(self.scores_path).split('.')[0])
        path_to_save = os.path.join(os.path.dirname(self.scores_path), plot_name)
        fig.savefig(path_to_save)

    def _get_train_test_indices(self, data, labels):
        if os.path.exists(self.indices_path):
            with open(self.indices_path, 'rb') as pkl:
                indices = pickle.load(pkl)
            train_indices = indices[0]
            test_indices = indices[1]
            print('Train-test indices already exists.')
        else:
            train_indices = list()
            test_indices = list()
            for i in range(self.t):
                train_indices.append(list())
                test_indices.append(list())
                folds = StratifiedKFold(n_splits=self.k,
                                        shuffle=True,
                                        random_state=(i + 1) * self.np_random_state)
                for train_ind, test_ind in folds.split(data, labels):
                    train_indices[-1].append(train_ind)
                    test_indices[-1].append(test_ind)
            with open(self.indices_path, 'wb') as pkl:
                pickle.dump([train_indices, test_indices], pkl)
            print('Train-test indices generated.')
        return train_indices, test_indices

    def _do_train_eval(self,
                       model_obj,
                       data,
                       labels,
                       train_ind,
                       test_ind):
        """Doing one training-validation step in kfold cross validation.

        At the end, saves a numpy array:
            [[test_loss, test_binary_accuracy, test_f1_score, test_sensitivity, test_specificity],
             [test_subject-wise_TN, test_subject-wise_FP, test_subject-wise_FN, test_subject-wise_TP],
             [ch-drop-fpr, ch-drop-tpr, ch-drop-th, ch-drop-roc-auc],
             [train_loss, train_binary_accuracy, train_f1_score, train_sensitivity, train_specificity]]
        """
        loss = model_obj.loss
        optimizer = model_obj.optimizer

        train_gen, n_iter_train, test_gen, n_iter_test = self._get_data_generators(data,
                                                                                   labels,
                                                                                   train_ind,
                                                                                   test_ind)

        if self.use_early_stopping_callback:
            es_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=3)
            callbacks = [es_callback]
        else:
            callbacks = []
        model = model_obj.create_model()
        model.compile(loss=loss, optimizer=optimizer)

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=n_iter_train,
                            epochs=self.epochs,
                            verbose=False,
                            callbacks=callbacks)

        sleep(1)

        x_test, y_test = self._generate_data_subset(test_gen, n_iter_test, self.test_generator.is_fixed)

        scores = [list() for _ in range(4)]

        # Add test scores
        test_scores = self._calc_scores(model, x_test, y_test)
        print('     test scores: ', test_scores)
        scores[0].extend(test_scores)

        # Add subject-wise test scores
        if self.data_mode == 'cross_subject':
            scores[1].extend(self._calc_subject_wise_scores(model,
                                                            x_test,
                                                            y_test))

        # Add channel-drop scores
        if self.channel_drop:
            scores[2].extend(self._get_channel_drop_scores(x_test,
                                                           y_test,
                                                           model))

        # Add train scores
        x_train, y_train = self._generate_data_subset(train_gen, n_iter_train, self.train_generator.is_fixed)
        train_scores = self._calc_scores(model, x_train, y_train)
        scores[3].extend(train_scores)
        print('     train scores: ', train_scores)
        return np.array(scores)

    def _get_data_generators(self, data, labels, train_ind, test_ind):
        if self.data_mode == 'cross_subject':
            train_data = [data[j] for j in train_ind]
            train_labels = [labels[j] for j in train_ind]
            test_data = [data[j] for j in test_ind]
            test_labels = [labels[j] for j in test_ind]
            train_gen, n_iter_train = self.train_generator.get_generator(data=train_data,
                                                                         labels=train_labels)
            test_gen, n_iter_test = self.test_generator.get_generator(data=test_data,
                                                                      labels=test_labels)
        else:
            train_gen, n_iter_train = self.train_generator.get_generator(data=data,
                                                                         labels=labels,
                                                                         indxs=train_ind)
            test_gen, n_iter_test = self.test_generator.get_generator(data=data,
                                                                      labels=labels,
                                                                      indxs=test_ind)
        return train_gen, n_iter_train, test_gen, n_iter_test

    @staticmethod
    def _generate_data_subset(gen, n_iter, is_fixed):
        x = list()
        y = list()

        if not is_fixed:
            for i in range(n_iter):
                x_batch, y_batch = next(gen)
                x.append(x_batch)
                y.append(y_batch)
        else:
            for i in range(n_iter):
                x_batch, y_batch = next(gen)
                x.extend(x_batch)
                y.extend(y_batch)
            x = np.array(x)
            y = np.array(y)
        return x, y

    @staticmethod
    def _calc_scores(model,
                     x,
                     y):
        """Calculates scores for accuracy, f1-score, sensitivity and specificity.

        x, y are lists of data batches.
        """

        scores = list()
        if not isinstance(y, np.ndarray):  # the generator of data wasn't of fixed type
            y_pred = list()
            y_true = list()
            for x_batch, y_batch in zip(x, y):
                y_pred.extend(model.predict(x_batch).tolist())
                y_true.extend(y_batch)
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
        else:
            y_true = y
            y_pred = model.predict(x)
        scores.append(mean_squared_error(y_true, y_pred))
        y_pred = np.where(y_pred > 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc = (tp + tn) / (tp + tn + fp + fn)
        f_score = f1_score(y_true, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return acc, f_score, sensitivity, specificity

    def _calc_subject_wise_scores(self,
                                  model,
                                  x_test,
                                  y_test):
        subject_ids = np.array(self.test_generator.belonged_to_subject[: len(y_test)])
        y_subjects = list()
        y_preds = list()
        for s_id in np.unique(subject_ids):
            indx = np.where(subject_ids == s_id)[0]
            x_subject = x_test[indx]
            y_subjects.append(int(y_test[indx][0]))
            y_pred_proba = model.predict(x_subject).mean()
            y_preds.append(int(np.where(y_pred_proba > 0.5, 1, 0)))
            # y_pred_proba = model.predict(x_subject)
            # y_pred = np.where(y_pred_proba > 0.5, 1, 0).mean()
            # if y_pred >= 0.5:
            #     y_preds.append(1)
            # else:
            #     y_preds.append(0)
        tn, fp, fn, tp = confusion_matrix(y_subjects, y_preds).ravel()
        return np.array([tn, fp, fn, tp])

    def _generate_final_scores(self):
        final_scores = list()
        for file_path in self.rounds_file_paths:
            final_scores.append(np.load(file_path, allow_pickle=True))
        final_scores = np.array(final_scores)
        np.save(self.scores_path, final_scores)
        for file_path in self.rounds_file_paths:
            os.remove(file_path)
        return final_scores

    def _get_channel_drop_scores(self,
                                 x_test,
                                 y_test,
                                 model):
        """Generates ROC with respect to number of dropped channels for trained model.

        Returns:
            [[fpr_0dropped, fpr_1dropped, fpr_4dropped, fpr_9dropped],
             [tpr_0dropped, tpr_1dropped, tpr_4dropped, fpr_9dropped],
             [thresholds_0dropped, thresholds_1dropped, thresholds_4dropped, thresholds_9dropped],
             [roc_auc_0dropped, roc_auc_1dropped, roc_auc_4dropped, roc_auc_9dropped]]
        """

        fpr = list()
        tpr = list()
        th = list()
        rocauc = list()
        for drop in range(4):
            if drop == 0:
                x_dropped = x_test
            else:
                x_dropped = self.drop_channels(x_test, drop ** 2)
            y_prob = model.predict(x_dropped)[:, 0]
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            fpr.append(false_positive_rate)
            tpr.append(true_positive_rate)
            th.append(thresholds)
            rocauc.append(roc_auc)
        return [fpr, tpr, th, rocauc]

    @staticmethod
    def _draw_roc_curve(fps, tps, ax):
        roc_auc = list()
        for i, j in zip(fps, tps):
            roc_auc.append(auc(i, j))
            linewidth = 1
            alpha = 0.6
            ax.plot(i, j, linewidth=linewidth, alpha=alpha)

        # mean_tpr = np.mean(tps, axis=0)
        # mean_fpr = np.linspace(0, 1, 100)
        # std_tpr = np.std(tps, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.axis('tight')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        # ax.grid(False)

        roc_auc = np.array(roc_auc)
        ax.text(0.8, 0.05,
                'AUC = {:2.2f} +- {:2.2f}'.format(roc_auc.mean(), roc_auc.std()),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes)

    @staticmethod
    def drop_channels(arr, drop=2):
        n_samples, n_times, n_channels = arr.shape
        to_drop = np.random.randint(low=0, high=n_channels, size=(n_samples, drop))
        dropped_x = arr.copy()
        for i, channels in enumerate(to_drop):
            dropped_x[i, :, channels] = 0
        return dropped_x


class StatisticalTester:

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def do_t_test(self, res_dir, reference_file=None):
        assert os.path.exists(res_dir), "specified directory not found."
        scores_paths = [os.path.join(res_dir, p) for p in os.listdir(res_dir) if
                        p.endswith('.npy') and (not p.startswith('train_test_indices'))]
        if not scores_paths:
            print('Can not find any score file.')
            return
        if reference_file is not None:
            l = [i for i in scores_paths if os.path.basename(i) == reference_file]
            if not l:
                print('reference file not found.')
                return
            comb = [(l[0], i) for i in scores_paths if i != l[0]]
            for res1_path, res2_path in comb:
                self._ttest(res2_path, res1_path)
                self._ttest(res1_path, res2_path)
        else:
            comb = combinations(scores_paths, 2)
            for res1_path, res2_path in comb:
                self._ttest(res1_path, res2_path)
                self._ttest(res2_path, res1_path)

    def _ttest(self, res1_path, res2_path):
        """Does a less-than test, i.e. tests the null hypothesis of res1_path's
         measure is equal or less than the res2's, versus the alternative hypothesis
         of "res1_path's measure is higher than res2_path's".
        """
        fn1 = res1_path.split('/')[-1]
        fn2 = res2_path.split('/')[-1]
        print("H0: x({}) <= x({})".format(os.path.basename(fn1), os.path.basename(fn2)))
        acc_diff, f_score_diff = self._get_diffs_mode1(res1_path, res2_path)

        t_stat, p_val = ttest_1samp(acc_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' Accuracies:')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

        t_stat, p_val = ttest_1samp(f_score_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' F1-scores:')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

        acc_diff, f_score_diff = self._get_diffs_mode2(res1_path, res2_path)

        t_stat, p_val = ttest_1samp(acc_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' Accuracies (SW):')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

        t_stat, p_val = ttest_1samp(f_score_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' F1-scores (SW):')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

    @staticmethod
    def _get_diffs_mode1(res1_path, res2_path):
        res1 = np.load(res1_path, allow_pickle=True)[:, 0]
        res2 = np.load(res2_path, allow_pickle=True)[:, 0]

        acc_diff = np.zeros(100)
        f_score_diff = np.zeros(100)
        for i in range(100):
            acc1, fscore_1 = res1[i][:2]
            acc2, fscore_2 = res2[i][:2]
            acc_diff[i] = acc1 - acc2
            f_score_diff[i] = fscore_1 - fscore_2
        return acc_diff, f_score_diff

    def _get_diffs_mode2(self, res1_path, res2_path):
        res1 = np.load(res1_path, allow_pickle=True)[:, 1]
        res2 = np.load(res2_path, allow_pickle=True)[:, 1]

        acc_diff = np.zeros(100)
        f_score_diff = np.zeros(100)
        for i in range(100):
            acc1, f_score1 = self._get_subject_wise_scores(res1[i])
            acc2, f_score2 = self._get_subject_wise_scores(res2[i])

            acc_diff[i] = acc1 - acc2
            f_score_diff[i] = f_score1 - f_score2
        return acc_diff, f_score_diff

    @staticmethod
    def _get_subject_wise_scores(res):
        tns, fps, fns, tps = res
        acc_vector = (tps + tns) / (tps + fns + fps + tns)
        precision_vector = tps / (tps + fps + 0.0001)
        recall_vector = tps / (tps + fns + 0.0001)
        specificity_vector = tns / (tns + fps + 0.0001)
        f_score_vector = 2 * (precision_vector * recall_vector) / (precision_vector + recall_vector + 0.0001)
        return acc_vector, f_score_vector
