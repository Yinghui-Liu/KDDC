# KDDC

This is the official implementation of our paper titled **"KDDC: Knowledge-Driven Disentangled Causal Metric Learning for Pre-Travel Out-of-Town Recommendation"**, which has been accepted by the IJCAI 2024 conference.

Pytorch versions are provided.

> Pytorch: https://pytorch.org

## Data

We have released the travel behavior dataset Foursquare and Yelp which are generated based on the [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquaredataset) and [Yelp](https://www.yelp.com.tw/dataset) dataset. You can run the model with these out-of-town data provided in the respective folder.

## Requirements

- python 3.x
- paddle 2.x / torch >= 1.7
- pgl / dgl>=0.6

## Run Our Model

Simply run the following command to train and evaluate:
```cmd
cd ./code
python main.py --ori_data {...} --dst_data {...} --trans_data {...} --pp_graph_path {...} ---save_path {...} --mode train --kg --train_trans --dc --infer --dm
```
