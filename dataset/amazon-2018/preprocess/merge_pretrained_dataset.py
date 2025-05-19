# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2025/1/16
import csv
import gzip
import json
import os

def get_raw_file(path):
    """
    获取亚马逊子类别数据集的原始文件路径
    :param path: 子类别数据集的根目录
    :return: 原始文件路径（交互csv文件以及商品元数据）
    """
    raw_files = os.listdir(path)
    inter_file = [file for file in raw_files if file.endswith(".csv")][0]
    inter_file = os.path.join(path, inter_file)
    meta_file = [file for file in raw_files if file.startswith("meta") and file.endswith(".json.gz")][0]
    meta_file = os.path.join(path, meta_file)
    return inter_file, meta_file

def merge_csv(pretrained_datasets_name, csv_files, file_output_path, file_output_name):
    all_rows = []
    print('Merge csv...')
    for i, file in enumerate(csv_files):
        with open(file, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if len(row) >= 2:
                    row[0] = pretrained_datasets_name[i] + '-' + row[0]
                    row[1] = pretrained_datasets_name[i] + '-' + row[1]
                all_rows.append(row)
    with open(os.path.join(file_output_path, file_output_name), mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(all_rows)
    print(f'Merge completed, output path is: {os.path.join(file_output_path, file_output_name)}')

def merge_meta(pretrained_datasets_name, meta_files, file_output_path, file_output_name):
    all_meta_rows = []
    print('Merge meta...')
    for i, meta_file in enumerate(meta_files):
        with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                json_data = eval(line)
                json_data['asin'] = pretrained_datasets_name[i] + '-' + str(json_data['asin'])
                all_meta_rows.append(json_data)
    with gzip.open(os.path.join(file_output_path, file_output_name), 'wt', encoding='utf-8') as f:
        for item in all_meta_rows:
            f.write(json.dumps(item) + '\n')
    print(f'Merge completed, output path is: {os.path.join(file_output_path, file_output_name)}')


if __name__ == '__main__':
    raw_path = os.path.join('..', 'raw')
    prefix_path = os.path.join('..', 'raw')
    pretrained_dataset_output_name = 'FHCKM'
    csv_file_output_name = 'FHCKM.csv'
    meta_file_output_name = 'meta_FHCKM.json.gz'
    pretrained_datasets = ['Food', 'CDs', 'Kindle', 'Movies', 'Home']
    inter_files = []
    meta_files = []

    for dataset in pretrained_datasets:
        temp_inter_file, temp_meta_file = get_raw_file(os.path.join(raw_path, dataset))
        inter_files.append(temp_inter_file)
        meta_files.append(temp_meta_file)

    pretrained_output_path = os.path.join(prefix_path, pretrained_dataset_output_name)
    if not os.path.exists(pretrained_output_path):
        os.mkdir(pretrained_output_path)
    merge_csv(pretrained_datasets, inter_files, pretrained_output_path, csv_file_output_name)
    merge_meta(pretrained_datasets, meta_files, pretrained_output_path, meta_file_output_name)


