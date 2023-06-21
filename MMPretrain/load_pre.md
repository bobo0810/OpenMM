# 训练时加载 预训练权重

```bash
bash tools/dist_train.sh configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e.py  2 --cfg-options model.pretrained=/xxx/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth
```

- 不加载，loss≈0.97

- 加载权重，loss≈0.4



### 参考

- [使用自定义数据集进行预训练](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/4_pretrain_custom_dataset.html#mmselfsup)

