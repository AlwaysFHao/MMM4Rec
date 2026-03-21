import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import jsonlines
import requests
from tqdm import tqdm

from utils import get_sub_paths, load_inter_file, load_meta_file, filter_inters_by_metas, filter_metas_by_inters, \
    filter_k_core_inters, group_inters_by_user, filter_metas_without_modality

import pickle
import torch
import numpy as np

""" 2018亚马逊数据集处理，参考自 https://github.com/kz-song/MMSRec """


def parse_args():
    """
    定义基础参数
    :return: 基础参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='../raw', type=str, help='raw data path')
    parser.add_argument('--processed_path', default='../processed', type=str, help='processed data path')
    # 最后保存的item信息中，text以及vision文件的前缀路径
    parser.add_argument('--prefix_path', default='./dataset/amazon-2018/preprocess', type=str, help='prefix path')

    parser.add_argument('--k_core', default=5, type=int, help='filter inters by k core')
    parser.add_argument('--vision_filter', default=False, type=bool, help='Throw items without vision. Default False')
    parser.add_argument('--text_filter', default=False, type=bool, help='Throw items without text. Default False')

    parser.add_argument('--item_outfile', default='item.jsonl', type=str, help='processed items meta file')
    parser.add_argument('--train_seq_outfile', default='train_seq.jsonl', type=str, help='processed train seq file')
    parser.add_argument('--eval_seq_outfile', default='eval_seq.jsonl', type=str, help='processed eval seq file')
    parser.add_argument('--test_seq_outfile', default='test_seq.jsonl', type=str, help='processed test seq file')
    parser.add_argument('--item2id_outfile', default='item2id.jsonl', type=str, help='processed item2id file')
    parser.add_argument('--user2id_outfile', default='user2id.jsonl', type=str, help='processed user2id file')
    parser.add_argument('--text_out_path', default='texts', type=str, help='text raw files')
    parser.add_argument('--vision_out_path', default='visions', type=str, help='vision raw files')
    parser.add_argument('--merge_train_eval_test', default=False, type=str, help='merge train eval test file')
    parser.add_argument('--sub_dataset', default=[], nargs='*', help='datasets to be processed')
    parser.add_argument('--is_hm4sr', default=False, nargs='*', help='preprocess for hm4sr')

    args = parser.parse_args()
    return args


class AmazonProcessor(object):
    def __init__(self, args):
        """
        2014亚马逊数据集处理类
        :param args: 配置参数
        """
        self.args = args
        self.prefix_path = args.prefix_path

        self.raw_path = args.raw_path
        self.processed_path = args.processed_path

        self.target_item_file = args.item_outfile
        self.target_train_seq_file = args.train_seq_outfile
        self.target_eval_seq_file = args.eval_seq_outfile
        self.target_test_seq_file = args.test_seq_outfile

        self.text_out_path = args.text_out_path
        self.vision_out_path = args.vision_out_path

        # 获取亚马逊数据集中的子类别文件夹
        self.sub_paths = get_sub_paths(self.raw_path)
        self.item2id = {}
        self.user2id = {}
        self.item2id_file = args.item2id_outfile
        self.user2id_file = args.user2id_outfile

    # @staticmethod
    # def request_picture(image_url, save_image_path):
    #     """
    #     根据url下载图片
    #     :param image_url: 图片url
    #     :param save_image_path: 保存图片路径
    #     :return: 是否成功
    #     """
    #     headers = {'Connection': 'close'}
    #     try:
    #         with requests.get(url=image_url, headers=headers) as request_result:
    #             if request_result.status_code == 200:
    #                 with open(save_image_path, 'wb') as fileObj:
    #                     fileObj.write(request_result.content)
    #                 return True
    #     except Exception:
    #         return False

    @staticmethod
    def request_picture(image_url, save_image_path):
        headers = {'Connection': 'close'}
        max_retries = 3
        timeout = 15  # 设置超时为15秒
        for attempt in range(max_retries):
            try:
                # 发起请求，并设置超时
                with requests.get(url=image_url, headers=headers, timeout=timeout) as request_result:
                    if request_result.status_code == 200:
                        with open(save_image_path, 'wb') as fileObj:
                            fileObj.write(request_result.content)
                        return True
            except requests.exceptions.Timeout:
                # print(f"Attempt {attempt + 1} timed out. Retrying...")
                pass
            except Exception as e:
                # print(f"Attempt {attempt + 1} failed with exception: {e}. Retrying...")
                pass
            # 等待一段时间再重试
            sleep(1)  # 可以根据需要调整重试等待时间
        return False  # 三次重试后返回 False

    def multiple_process_item(self, id, meta, path):
        """
        多线程处理item数据的执行函数
        :param id: 商品id
        :param meta: 对应商品元数据
        :param path: 子类别数据集根目录
        :return: 商品id、图片保存路径 和 文本文件保存路径
        """
        # 取出商品元数据中的文本和图片url
        text = meta["text"]
        image = meta["vision"]
        # 生成文本txt文件名，文件名为商品id
        text_file = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.text_out_path,
                                 f"{str(id)}.txt")
        # 判断是否存在文本文件，不存在则写入（确保可断点执行以及线程安全）
        if not os.path.exists(text_file):
            with open(text_file, "w", encoding="utf-8") as fobj:
                fobj.write(text)
        # 生成文本txt文件的路径
        text_path = os.path.join(self.prefix_path, text_file)

        # 图片可能存在缺失，先初始化为None
        image_path = None
        if image is not None:
            # 生成图片文件的文件名，文件名为商品id
            image_file = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)),
                                      self.vision_out_path, f"{str(id)}{os.path.splitext(image)[-1]}")
            # 先判断是否已经存在对应图片，如果不存在再尝试下载
            if os.path.exists(image_file) or self.request_picture(image, image_file):
                # 已经存在或者下载成功就直接生成图片文件路径
                image_path = os.path.join(self.prefix_path, image_file)
        return id, image_path, text_path

    def process_item_data(self, metas, path):
        """
        下载并保存处理商品模态信息
        :param metas: 商品元数据
        :param path: 子类别数据集根目录
        :return:
        """
        # 创建保存商品文本的文件夹
        text_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.text_out_path)
        os.makedirs(text_path, exist_ok=True)
        # 创建保存商品图片的文件夹
        vision_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.vision_out_path)
        os.makedirs(vision_path, exist_ok=True)

        print(f"Process Item Data: {len(metas)}")
        # 新的商品元数据字典
        new_metas = {}
        # 开启最多为128线程的线程池
        with ThreadPoolExecutor(max_workers=512) as executor:
            # 处理线程列表
            process_list = []
            # 遍历商品元数据
            for id, meta in metas.items():
                # 提交处理线程
                process = executor.submit(self.multiple_process_item, id, meta, path)
                process_list.append(process)

            for process in tqdm(as_completed(process_list), total=len(process_list)):
                id, image_path, text_path = process.result()
                new_metas[id] = {"vision": image_path, "audio": None, "text": text_path}

        return new_metas

    def write_item_file(self, metas, path):
        """
        生成商品元数据文件
        :param metas: 商品元数据
        :param path: 子类别数据集根目录
        :return:
        """
        item_data = {}
        for id, meta in tqdm(metas.items(), desc="write item file"):
            item_data[self.item2id[id]] = {"vision": meta["vision"], "text": meta["text"]}

        item_file = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)), self.target_item_file)
        with open(item_file, mode='w') as f:
            json.dump(item_data, f, indent=4, sort_keys=True, ensure_ascii=False)

    def write_seq_file(self, users, path):
        """
        生成并写入交互序列文件
        :param users: users的交互序列
        :param path: 子类别根目录
        :return:
        """
        print(f"Process Seq Data: {len(users)}")
        if self.args.merge_train_eval_test:
            seq_data = []
            # 遍历users交互序列字典
            for id, interacts in tqdm(users.items()):
                uid = self.user2id[id]
                # 交互序列根据时间进行排序
                interacts = sorted(interacts, key=lambda item: item["time"])
                interacts = [(self.item2id[item["item"]], item["time"]) for item in interacts]

                # 生成当前用户的子序列作为训练集，最后两位空出
                for index in range(2, len(interacts) + 1):
                    seq_data.append((uid, interacts[:index]))
            # 生成子类别数据集路径
            target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
            os.makedirs(target_path, exist_ok=True)

            # 保存训练集
            train_file = os.path.join(target_path, self.target_train_seq_file)
            with jsonlines.open(train_file, mode='w') as wfile:
                for line in seq_data:
                    wfile.write(line)
        else:
            # 训练、验证和测试集
            train_seq_data = []
            eval_seq_data = []
            test_seq_data = []
            # 遍历users交互序列字典
            for id, interacts in tqdm(users.items()):
                uid = self.user2id[id]
                # 交互序列根据时间进行排序
                interacts = sorted(interacts, key=lambda item: item["time"])
                interacts = [(self.item2id[item["item"]], item["time"]) for item in interacts]

                # 生成当前用户的子序列作为训练集，最后两位空出
                for index in range(2, len(interacts) - 1):
                    train_seq_data.append((uid, interacts[:index]))

                # 子序列顺序逆置
                # for index in range(len(interacts) - 2, 1, -1):
                #     train_seq_data.append((uid, interacts[:index]))

                # 截取至倒数第二位作为验证集
                eval_seq_data.append((uid, interacts[:-1]))
                # 截取至最后一位作为测试集
                test_seq_data.append((uid, interacts[:]))
            # 生成子类别数据集路径
            target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
            os.makedirs(target_path, exist_ok=True)

            # 保存训练集
            train_file = os.path.join(target_path, self.target_train_seq_file)
            with jsonlines.open(train_file, mode='w') as wfile:
                for line in train_seq_data:
                    wfile.write(line)

            # 保存验证集
            eval_file = os.path.join(target_path, self.target_eval_seq_file)
            with jsonlines.open(eval_file, mode='w') as wfile:
                for line in eval_seq_data:
                    wfile.write(line)

            # 保存测试集
            test_file = os.path.join(target_path, self.target_test_seq_file)
            with jsonlines.open(test_file, mode='w') as wfile:
                for line in test_seq_data:
                    wfile.write(line)

    def _get_raw_file(self, path):
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

    def _generate_item2id(self, meta, path):
        # 交互数据中存在的商品id
        items = set()
        # 统计所有的商品id
        for item_id in tqdm(meta.keys(), desc="generate item2id file"):
            items.add(item_id)
        # set转list按照字符串进行排序
        items = sorted(items)
        for index, item in enumerate(items):
            self.item2id[item] = index + 1
        # 生成子类别数据集路径
        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        os.makedirs(target_path, exist_ok=True)
        # 保存id映射表
        item2id_file = os.path.join(target_path, self.item2id_file)
        with jsonlines.open(item2id_file, mode='w') as wfile:
            for key in self.item2id:
                wfile.write((key, self.item2id[key]))

    def _generate_user2id(self, users_inter, path):
        # 交互数据中存在的用户id
        users = []
        # 统计所有的商品id
        for user_id in tqdm(users_inter.keys(), desc="generate user2id file"):
            users.append(user_id)
        for index, user in enumerate(users):
            self.user2id[user] = index
        # 生成子类别数据集路径
        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        os.makedirs(target_path, exist_ok=True)
        # 保存id映射表
        user2id_file = os.path.join(target_path, self.user2id_file)
        with jsonlines.open(user2id_file, mode='w') as wfile:
            for key in self.user2id:
                wfile.write((key, self.user2id[key]))

    def _prepare_category(self, metas, path, padding_idx=0):
        """
        从商品元数据中提取类别信息，生成每个item的类别one-hot向量。
        对应原始代码中的 prepare_category 函数。
        :param metas: 过滤后的商品元数据字典 {item_id: meta_dict}
        :param path: 子类别数据集根目录
        :param padding_idx: 填充索引，默认为0
        """
        # 确保item2id已经生成
        if not self.item2id:
            raise ValueError("item2id mapping must be generated before preparing categories.")

        # 1. 加载商品元数据文件以获取原始类别信息
        # 注意：您当前的metas是处理后的，只包含vision和text路径。
        # 原始的类别信息在第一次调用load_meta_file得到的metas中。
        # 为了获取类别，我们需要重新加载原始meta文件，或修改流程传递原始元信息。
        # 这里我们假设有一个方法能获取到包含原始类别信息的meta字典 (raw_metas)。
        # 由于在process方法中，原始的meta字典被过滤和覆盖了，我们需要调整流程。
        # 建议：在process方法中，在过滤k-core之后，保存一份包含原始信息的metas_copy用于此处。
        # 以下代码逻辑假设我们接收到的 `metas` 参数是包含原始类别信息的字典。
        # 在后续的process方法修改中，我们会传入正确的数据。

        print(f"Preparing category features for {len(metas)} items.")
        cat_dict = {}  # 映射: item_id (新ID) -> 类别ID列表
        cat_type_dict = {}  # 映射: 原始类别字符串 -> 全局类别ID

        for raw_item_id, meta in metas.items():
            # 获取该商品在过滤、重映射后的新ID
            if raw_item_id not in self.item2id:
                continue
            new_item_id = self.item2id[raw_item_id]

            # 从meta中提取类别。假设类别信息在 meta['categories'] 中，是一个列表的列表。
            # 例如： [["Electronics", "Computers & Accessories", "Touchscreen Tablet Accessories"]]
            # 我们取最内层的列表，或者根据您的数据结构调整。
            categories = meta.get("categories")
            if categories and len(categories) > 0:
                # 通常 categories 是一个list of list，我们取第一个子列表
                category_list = categories[0] if isinstance(categories[0], list) else categories
                category_id_list = []
                for cat in category_list:
                    if cat not in cat_type_dict:
                        cat_type_dict[cat] = len(cat_type_dict)  # 从0开始分配ID
                    category_id_list.append(cat_type_dict[cat])
                cat_dict[new_item_id] = torch.tensor(category_id_list, dtype=torch.long)
            else:
                # 没有类别的商品，赋予一个空张量
                cat_dict[new_item_id] = torch.tensor([], dtype=torch.long)

        # 2. 构建类别特征矩阵
        num_items = len(self.item2id) + 1  # +1 是为了包含padding索引0
        num_cat_types = len(cat_type_dict)
        result_list = [None] * num_items

        # 首先，为padding索引（0）放置一个全零向量
        result_list[padding_idx] = torch.zeros((1, num_cat_types), dtype=torch.long)

        # 为每个有效的item生成one-hot向量
        for new_item_id in range(1, num_items):  # item_id 从1开始
            if new_item_id in cat_dict:
                cat_ids = cat_dict[new_item_id]
                if len(cat_ids) > 0:
                    # 对多个类别ID进行one-hot编码并求和
                    one_hot = torch.nn.functional.one_hot(cat_ids, num_classes=num_cat_types).sum(dim=0)
                else:
                    one_hot = torch.zeros((num_cat_types,), dtype=torch.long)
            else:
                # 理论上不会进入这里，因为cat_dict应包含所有item
                one_hot = torch.zeros((num_cat_types,), dtype=torch.long)
            result_list[new_item_id] = one_hot.view(1, -1)  # 保持为2D: (1, num_cat_types)

        # 拼接所有向量
        category_tensor = torch.cat(result_list, dim=0)  # 形状: (num_items, num_cat_types)

        # 3. 保存到文件
        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        output_file = os.path.join(target_path, "cat.pt")
        torch.save(category_tensor, output_file)
        print(f"Category tensor saved to {output_file}, shape: {category_tensor.shape}")

        # 可选：保存类别类型映射，便于调试
        cat_type_map_file = os.path.join(target_path, "cat_type_dict.pkl")
        with open(cat_type_map_file, 'wb') as f:
            pickle.dump(cat_type_dict, f)

    def _calculate_time_stats(self, users, path):
        """
        计算并保存时间相关的统计信息：
        1. 用户连续交互之间的最大时间间隔（取log2）。
        2. 整个数据集中最早和最晚的日期（按天计算）。
        对应原始代码中的 local_timestamp 和 local_minmax_day 函数。
        :param users: 用户交互序列字典 {user_id: [interacts]}
        :param path: 子类别数据集根目录
        """
        print("Calculating time statistics...")
        max_interval = 0
        min_date = float('inf')
        max_date = 0

        for uid, interacts in users.items():
            # 确保交互按时间戳排序
            sorted_interacts = sorted(interacts, key=lambda x: x["time"])
            last_time = None
            for interact in sorted_interacts:
                timestamp = interact["time"]
                # 计算最小/最大日期（天）
                day = timestamp // 86400  # 转换为天数
                if day < min_date:
                    min_date = day
                if day > max_date:
                    max_date = day

                # 计算连续交互的最大间隔
                if last_time is not None:
                    interval = timestamp - last_time
                    if interval > max_interval:
                        max_interval = interval
                last_time = timestamp

        # 对最大间隔取 log2(interval + 1)，避免对0取log
        if max_interval > 0:
            max_interval_log = torch.log2(torch.tensor(max_interval, dtype=torch.float) + 1.0).item()
        else:
            max_interval_log = 0.0

        # 保存结果
        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        os.makedirs(target_path, exist_ok=True)

        # 保存最大间隔
        interval_file = os.path.join(target_path, "max_interval.bin")
        with open(interval_file, 'wb') as f:
            pickle.dump(max_interval_log, f)
        print(f"Max log2(interval+1) saved to {interval_file}: {max_interval_log}")

        # 保存日期范围
        date_range_file = os.path.join(target_path, "date_range.bin")
        with open(date_range_file, 'wb') as f:
            pickle.dump((int(min_date), int(max_date)), f)  # 保存为整数
        print(f"Date range saved to {date_range_file}: ({int(min_date)}, {int(max_date)})")

    def process(self, preprocess_dataset=None, is_hm4sr: bool = False):
        # 遍历所有已经存在的子类别目录
        for path in self.sub_paths:
            if path.split('/')[-1] in preprocess_dataset if preprocess_dataset is not None else True:
                print(f"\n-----Processing data {path}")
                # 获取原始文件路径
                inter_file, meta_file = self._get_raw_file(path)
                # 根据交互csv文件得到交互数据集合 set(set(user, item, rate, time))
                inters = load_inter_file(inter_file)
                # 加载meta文件得到商品元数据字典 (包含原始类别信息)
                raw_metas = load_meta_file(meta_file)
                # 根据商品元数据筛选交互数据
                inters = filter_inters_by_metas(inters, raw_metas)
                # k-core过滤
                inters = filter_k_core_inters(inters, self.args.k_core, self.args.k_core)
                # 根据现有的交互数据过滤商品元数据
                filtered_metas = filter_metas_by_inters(raw_metas, inters)
                # 下载并保存处理商品模态信息
                processed_metas = self.process_item_data(filtered_metas, path)
                # 处理缺失模态的商品
                processed_metas = filter_metas_without_modality(processed_metas, self.args.vision_filter,
                                                                self.args.text_filter)

                # 重新根据商品元数据筛选交互数据
                inters = filter_inters_by_metas(inters, processed_metas)
                # 重新筛选k-core
                inters = filter_k_core_inters(inters, self.args.k_core, self.args.k_core)
                # 根据交互序列过滤商品元信息 dict{id: {info}}
                final_metas = filter_metas_by_inters(processed_metas, inters)

                # 建立 item2id 映射
                self._generate_item2id(final_metas, path)

                if is_hm4sr:
                    # --- hm4sr独有步骤1: 准备类别特征 ---
                    metas_for_category = {item_id: raw_metas[item_id] for item_id in final_metas.keys() if
                                          item_id in raw_metas}
                    # 调用类别准备函数
                    self._prepare_category(metas_for_category, path, padding_idx=0)

                # 生成users交互序列
                users = group_inters_by_user(inters)

                if is_hm4sr:
                    # --- hm4sr独有步骤2: 计算时间统计 ---
                    self._calculate_time_stats(users, path)

                # 生成user2id的映射
                self._generate_user2id(users, path)
                # 切分生成交互序列数据集
                self.write_seq_file(users, path)
                # 写入商品元信息 (处理后的，只有路径)
                self.write_item_file(final_metas, path)

if __name__ == '__main__':
    # sub_dataset = ['Scientific']
    # 初始化配置参数
    args = parse_args()
    # 实例化亚马逊处理对象
    api = AmazonProcessor(args)
    # 要处理的数据集
    sub_dataset = args.sub_dataset
    if len(sub_dataset) == 0:
        sub_dataset = None
    # 调用处理方法
    api.process(sub_dataset, is_hm4sr=args.is_hm4sr)
