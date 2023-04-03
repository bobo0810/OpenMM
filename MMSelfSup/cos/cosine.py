import argparse
import copy
import os
import os.path as osp
import time
import torch.nn.functional as F
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmselfsup.evaluation.functional import knn_eval
from mmselfsup.models.utils import Extractor
from mmselfsup.registry import MODELS
from mmselfsup.utils import register_all_modules
import cv2
from PIL import Image
from timm.data.transforms_factory import create_transform as timm_transform
from sklearn.metrics.pairwise import cosine_similarity



def create_model(config,checkpoint):
    """初始化模型"""
    # 构建模型
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu") # 加载权重
    return model


def process(img_path):
    """图像预处理"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.fromarray(img)
    img_trans = timm_transform([224, 224]) # timm库的默认预处理
    img = img_trans(img).unsqueeze(0)  # [1,C,H,W]
    return img


@torch.no_grad()
def get_feature(input, model):
    """
    获取特征
    """
    input = input.cuda()

    model.eval()
    model.cuda()

    # 主干网络
    features = model.backbone(input)  # [1,2048,7,7]

    # 颈网络
    features = model.neck([features[-1]])
    # flat features 拉直特征
    flat_features = [feat.view(feat.size(0), -1) for feat in features]

    # 正则化
    l2_feature = F.normalize(flat_features[0], p=2, dim=1)
    return l2_feature


# 初始化模型
config = "configs/selfsup/_base_/models/mocov3_resnet50.py" # 模型
checkpoint = "mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth" # 权重
model=create_model(config,checkpoint)
# 图像预处理
img_path="xxx.jpg"
input=process(img_path)
# 获取特征
l2_feature = get_feature(input, model) #[1,256]


# 计算两个图像的余弦相似度
# cosine = cosine_similarity([feature_1], [feature_2]) # l2_feature[0].detach().cpu().numpy()


