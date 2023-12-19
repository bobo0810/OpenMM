# 验证图像完整性  



```python
from PIL import Image
import os
from tqdm import tqdm
import mmcv
import mmengine.fileio as fileio
def is_image_corrupted(image_path):
    try:
      	# PIL验证
        Image.open(image_path).verify()

        # mmcv验证
        img_bytes = fileio.get(image_path)
        img = mmcv.imfrombytes(img_bytes)
        if img is None:
            return True # 图像损坏
        return False  # 图像未损坏
    except (IOError, SyntaxError) as e:
        return True  # 图像损坏
```

