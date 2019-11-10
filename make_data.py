# -*- coding: utf-8 -*-
"""Generate numpy data from .edf files."""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from urllib.request import urlretrieve
import zipfile

import mne
import numpy as np
import pandas as pd


def generate_df(src_dir):
    def is_responder(row):
        if row['Healthy']:
            return 'h'
        else:
            if row['Subject Number'] < 17:
                return 'r'
            else:
                return 'nr'

    data_paths = [os.path.abspath(os.path.join(src_dir, data_dir)) for data_dir in os.listdir(src_dir)]

    subject_number = [int(os.path.basename(path).split()[1][1:]) for path in data_paths]
    healthy = [True if 'H ' in path else False for path in data_paths]
    mdd = [True if 'MDD ' in path else False for path in data_paths]
    eo = [True if ' EO' in path else False for path in data_paths]
    ec = [True if ' EC' in path else False for path in data_paths]
    erp = [True if ' TASK' in path else False for path in data_paths]

    mapping_dict = {'Path': data_paths, 'Subject Number': subject_number,
                    'Healthy': healthy, 'MDD': mdd, 'EO': eo, 'EC': ec, 'ERP': erp}

    data = pd.DataFrame(mapping_dict)
    data['Responder'] = data.apply(is_responder, axis=1)

    eyes_closed = data[data['EC'] == True].copy()
    eyes_closed.drop(['EC', 'EO', 'ERP'], axis=1, inplace=True)

    eyes_opened = data[data['EO'] == True].copy()
    eyes_opened.drop(['EC', 'EO', 'ERP'], axis=1, inplace=True)

    df = pd.concat([eyes_opened, eyes_closed])
    return df


def generate_data(src_dir, dst_dir, n_channels=19):
    """Loads each trial, multiplies by 1e6 and saves  the normalized array as numpy array.
    """

    df = generate_df(src_dir)
    s = df.iloc[0]
    raw = mne.io.read_raw_edf(s['Path'], preload=True, verbose=False)
    raw.pick_types(eeg=True)
    channels = raw.ch_names[:n_channels]

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    print('Loading samples ...')
    with tqdm(total=len(df)) as pbar:
        for i, subject in enumerate(df.values):
            path = subject[0]
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.pick_types(eeg=True)
            raw.pick_channels(channels)
            arr = raw.get_data() * 1e6
            label = subject[-1]

            path_to_save = os.path.join(dst_dir, 's{}_{}.npy'.format(i + 1, label))

            np.save(path_to_save, arr)
            pbar.update(1)
    return


def download_data(download_dir):
    zip_file_name = 'eeg_data.zip'
    zip_file_path = os.path.join(download_dir, zip_file_name)
    data_dir = os.path.join(download_dir, 'eeg_data')

    if os.path.exists(data_dir):
        print('Data exists.')
        return
    elif os.path.exists(zip_file_path):
        print('Zip file exists.')
    else:
        url = 'https://ndownloader.figshare.com/articles/4244171/versions/2'
        print('Downloading file {} ...'.format(zip_file_name))
        urlretrieve(url, zip_file_path)
        print('File downloaded to ', zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print('Zip file extracted to ', data_dir)
    os.remove(os.path.join(data_dir, '6921143_H S15 EO.edf'))
    os.rename(os.path.join(data_dir, '6921959_H S15 EO.edf'),
              os.path.join(data_dir, 'H S15 EO.edf'))


if __name__ == '__main__':
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', required=True, type=str, help='Complete path to the folder containing edf data.')
    parser.add_argument('--dst_folder', required=False, type=str, default='data')
    args = parser.parse_args()

    src_dir = args.data_dir
    dst_dir = os.path.abspath(os.path.join(os.path.dirname(src_dir), args.dst_folder))
    generate_data(src_dir=src_dir, dst_dir=dst_dir)
