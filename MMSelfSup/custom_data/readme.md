# 自定义数据集

## 1. 数据集格式

自监督无需标签，直接

``` 
MMSelfSup/data/custom_data/
├── meta
│   ├── train.txt
└── train
    ├── 001.png
    ├── 002.png
    └── ...
```

train.txt内容如下   无标签图像，仅需图像路径即可。

```
001.png
002.png
...
```

## 2. 启用

```python
train_dataloader = dict(
    dataset=dict(
      	type="ImageList" # MMSelfSup内置数据集解析
        data_root="data/custom_data/"# 数据集根路径 
        ann_file="meta/train.txt", 	# 训练集的图像列表
        data_prefix=dict(img_path="train/"),# 图像路径的前缀
        ...
    ),
)
```



### 参考

- [添加数据集-官方文档](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/advanced_guides/add_datasets.html)

- [ImageList-API](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/api.html#)