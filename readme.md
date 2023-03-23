# OpenMM最佳食用手册



## MMClassification

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





## MMYOLO

- 定制部分
  - datasets
    - transforms 包含各种数据增强变换。
  - models
    - detectors 定义所有检测模型类。
    - data_preprocessors 用于预处理模型的输入数据。
    - backbones 包含各种骨干网络
    - necks 包含各种模型颈部组件
    - dense_heads 包含执行密集预测的各种检测头。
    - losses 包含各种损失函数
    - task_modules 为检测任务提供模块。例如 assigners、samplers、box coders 和 prior generators。
    - layers 提供了一些基本的神经网络层
  - engine
    - optimizers 提供优化器和优化器封装。
    - hooks 提供 runner 的各种钩子。