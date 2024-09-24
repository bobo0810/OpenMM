# OpenMM最佳食用手册 && 碎碎念

# 一 碎碎念

## Transformers用法

- [Siglip导出ONNX](Other/siglip2onnx.py)

# 二 OpenMM用法

- 该仓库收录于[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)
- [荣誉证书 ](荣誉证书)


## MMEngine
- Hook用法 [教程1](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html#id2)  [教程2](https://mmengine.readthedocs.io/zh_CN/latest/design/hook.html)
- [为模型不同模块分别指定学习率](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md#%E4%B8%BA%E6%A8%A1%E5%9E%8B%E4%B8%8D%E5%90%8C%E9%83%A8%E5%88%86%E7%9A%84%E5%8F%82%E6%95%B0%E8%AE%BE%E7%BD%AE%E4%B8%8D%E5%90%8C%E7%9A%84%E8%B6%85%E5%8F%82%E7%B3%BB%E6%95%B0)



## MMCV

- 验证图像完整性  [教程](MMCV/img_full.md) 

## MMPretrain

- `下游识别模型`加载`自监督预训练权重`   [教程](MMPretrain/load_weight.md) 
- 训练时加载`预训练权重`   [教程](MMPretrain/load_pre.md) 
- 自监督模型&识别模型 如何导出ONNX
  ```python
  from mmpretrain import get_model

  model_name="mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k"
  pretrained="./mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220826-25213343.pth"

  model = get_model(model_name, pretrained=pretrained)
  
  torch_model=model.backbone  # 自监督模型，与model.extract_feat等价,仅包含backbone
  torch_model=model  # 识别模型，包含backbone、neck、head
  
  torch_model.eval()
  onnx_path="./model.onnx" # 文件夹

  imgs = torch.ones(tuple([1,3,224,224]))
  with torch.no_grad():
      torch.onnx.export(
              torch_model,
              imgs,
              onnx_path,
              verbose=False,
              opset_version=17,
              input_names=["input"],
              output_names=["output"],
              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, # Batch维度动态
          )
  ```

![MMPretrain-导出](assets/MMPretrain-导出.jpg)

## MMYOLO目标检测

![MMYOLO定制](assets/MMYOLO定制.svg)

------


<details>
<summary>以下内容已过时</summary>
## MMClassification图像识别

![配置文件](assets/配置文件.svg)

<center>整体框架图</center>

[配置文件-官方教程](https://mmclassification.readthedocs.io/zh_CN/dev-1.x/user_guides/config.html)

优势

- 模型库：支持内置库、timm、huggingface
- 任务：支持单任务、多任务、TTA测试等

用法

- 启用timm模型库

  ```python
  model = dict(
      _delete_=True,
      type="TimmClassifier",
      model_name="swinv2_base_window16_256",
      pretrained=True, # timm接口参数
      loss=xxx,
      train_cfg=xxx,
  )
  ```

- 命令行改参数

  ```bash
  bash ./tools/dist_train.sh  xx.py  --amp  --cfg-options train_dataloader.batch_size=12
  ```



## MMSelfSup自监督

- 比较两个图像的相似度   [代码](MMSelfSup/cos/cosine.py)
- 自定义数据集训练  [代码](MMSelfSup/custom_data/readme.md)
</details>









