# DATASET

*You can refer to this documentation to understand our dataset processing pipeline.*

Due to the complexity and time-consuming nature of dataset processing, we strongly recommend downloading our preprocessed datasets directly via the anonymous link [https://figshare.com/s/f7603ea556c23c2aef88](https://figshare.com/s/f7603ea556c23c2aef88) for quick setup.

Below we describe how to use our provided dataset preprocessing pipeline:

## 1. Raw Data Acquisition
Our work focuses on the [`🛒 Amazon Review 2018`](https://nijianmo.github.io/amazon/index.html) dataset. To obtain the original dataset files corresponding to the domains in our paper:

1. Visit the official website: [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html) or [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
2. For the Scientific domain (as an example):
   - Locate the `Industrial_and_Scientific` category. 
   - Select both [`🧷 metadata`](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Industrial_and_Scientific.json.gz) and [`🧷 ratings_only`](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Industrial_and_Scientific.csv) options for download. 
3. You will receive two files:
   - 📖 `meta_Industrial_and_Scientific.json.gz`
   - 📖 `Industrial_and_Scientific.csv`
4. Place these files in: [`📁 amazon-2018/📁 raw/📁 Scientific/`](amazon-2018/raw/Scientific)

## 2. Folder Structure

You can click on the directory below to expand and view the folder structure: 

<details><summary>📁 dataset</summary>
   <ul>
      <li>📁 amazon-2018</li>
         <ul>
            <li>📁 preprocess</li>
               <ul>
                  <li>📁 extract_utils</li>
                     <ul>
                        <li>🐍 siglip.py</li>
                        <li>🐍 ...</li>
                     </ul>
                  <li>🚅 run.sh</li>
                  <li>🚅 pretrain_dataset.sh</li>
                  <li>🐍 process_item.py</li>
                  <li>🐍 extract_feature.py</li>
                  <li>🐍 merge_pretrained_dataset.py</li>
                  <li>🐍 utils.py</li>
               </ul>
            <li>📁 processed</li>
               <ul>
                  <li>📁 Scientific</li>
                     <ul>
                        <li>📁 text</li>
                           <ul>
                              <li>📃 {item}.txt</li>
                              <li>📃 ...</li>
                           </ul>
                        <li>📁 vision</li>
                           <ul>
                              <li>📷 {item}.jpg</li>
                              <li>📷 ...</li>
                           </ul>
                        <li>📊 train_seq.jsonl</li>
                        <li>📊 eval_seq.jsonl</li>
                        <li>📊 test_seq.jsonl</li>
                        <li>💿 image_feat.pth</li>
                        <li>💿 text_feat.pth</li>
                        <li>📊 item.jsonl</li>
                        <li>📊 item2id.jsonl</li>
                        <li>📊 user2id.jsonl</li>
                     </ul>
                  <li>📁 ...</li>
               </ul>
            <li>📁 raw</li>
               <ul>
                  <li>📁 Scientific</li>
                     <ul>
                        <li>📖 Industrial_and_Scientific.csv</li>
                        <li>📖 meta_Industrial_and_Scientific.json.gz</li>
                     </ul>
                  <li>📁 ... </li>
               </ul>
         </ul>
      <li>Ⓜ️ READEM.md</li>
    </ul>
</details>

## 3. Pretraining Dataset Processing
Before running the experiments, you need to prepare the raw data for the following five domains:
- `Grocery and Gourmet Food`
- `Home and Kitchen` 
- `CDs and Vinyl`
- `Kindle Store`
- `Movies and TV`

After obtaining all required datasets, simply execute our pre-configured shell script:

```shell
cd ./amazon-2018/preprocess
/bin/bash/ pretrain_dataset.sh
cd ../../
```

The workflow consists of three main steps:
1. **Dataset Merging**: Combine datasets from different domains
2. **Dataset Serialization**: 
   - Serialize the processed data
   - Download raw multimodal information for items
3. **Feature Extraction**:
   - Extract multimodal features using pretrained models (e.g. SigLIP)

## 4. Downstream Dataset Processing
After preparing the downstream dataset for your target domain, simply execute our provided shell script:

```shell
cd ./amazon-2018/preprocess
/bin/bash/ run.sh
cd ../../
```

Note: The script currently uses the **Scientific** domain as default. For other domains, modify the corresponding parameters `sub_dataset` in the shell script [`run.sh`](./amazon-2018/preprocess/run.sh).

Update: We have now added data preprocessing support for **HM4SR**. Please modify the `is_cm4sr` parameter in the shell script to "True". 

## 5.Important Notes
### 5.3 Processing Variability
The dataset processing uses set operations for deduplication, which may result in varying output ordering across different runs. However, This does not affect the dataset's equivalence. Experimental results may show minor fluctuations. 

### 5.4 Image Downloading
When downloading raw product images:

- Network congestion may occur

- We've implemented multi-threading optimization

- Consider retrying if downloads fail

## 6. Acknowledgement
For dataset processing, we referenced approaches from [MMSRec](https://github.com/kz-song/MMSRec), [UniSRec](https://github.com/RUCAIBox/UniSRec), and [MISSRec](https://github.com/gimpong/MM23-MISSRec). 

Thank you for reading this far! We hope you find this resource valuable for your research.