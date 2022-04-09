# One Stage Detector (SAPD w/ Dynamic Weighted Loss Mask + MaxPoolFilter)

---

_* 透過動態損失權重遮罩，可在相同精度下得到最佳的訓練時間成本_

_* 歡迎指點另一貢獻 [PCB Defect Detection Based on SAPD with Mix Subnetwork](https://github.com/gogo12235LYH/keras-pcb-sapd-mix)_

* Tensorflow 2.6.0
* Keras
* 3DMF, [End-to-End Object Detection with Fully Convolutional Network](https://arxiv.org/abs/2012.03544)

---

## Updates

1. 2020-04-06 -> First Commit.
2. 2020-04-09 -> DWLM w/ MF on 100 eps.

## ToDo

1. Keras Sequence generator
2. README.md 補完
---

## 目錄

1. 安裝
2. 訓練
3. 評估
4. 參考

---

## 1. 安裝

### 請確保自己的環境是否已安裝下列包及對應版本 :

* tensorflow 2.3.0 絕對不行 !

```
Cython
keras==2.6.0
opencv-contrib-python
Pillow
progressbar2
tensorflow==2.6.0
tensorflow-addons==0.14.0
```

### setup.py (compute_overlap.c and compute_overlap.pyx)

```commandline
python setup.py build_ext --inplace
```

## 2. 訓練

### config.py and generator/pipline.py

以下可以更改超參數，週期數、每週期疊代次數、總預測類別、批次量、訓練解析度、多卡訓練及混合經度訓練。

```python
""" Hyper-parameter setting """
DB_MODE = 'tf'  # 'tf' or 'keras', it means that using tf.data or keras.util.sequence.
EPOCHs = 100
STEPs_PER_EPOCH = 500  # steps in one epoch
BATCH_SIZE = 2  # Global Batch size
NUM_CLS = 6
PHI = 1  # B0:(512, 512), B1:(640, 640), B2:(768, 768), B3:(896, 896), B4:(1024, 1024) ~ B7(1048, 1048)
MULTI_GPU = 0
MIXED_PRECISION = 1
```

_待..._

---

## 3. 評估

* DWLM: Dynamic Weighted Loss Mask
* MF: MaxFilter ( Single Level Pooling )
* 3DMF: 3DMaxFilter ( Three Level Pooling ) 參考 5.1

### 3.1 Deep PCB:
* 訓練及評估影像大小: 640 * 640 ( PHI=1 )，就那個黑白的 PCB 瑕疵資料集
* 在 50 epochs 下，原始 SAPD 訓練時間約為 1.9 小時; DWLM 約為 1.1小時，供參考(RTX 3060 6g)

| subnetworks  | backbone | setting    | mAP    | AP.5   | AP.75  | AP.9   |
|--------------|----------|------------|--------|--------|--------|--------|
| SAPD - org   | R50      | 50 epochs  | 0.7530 | 0.9751 | 0.8881 | 0.3518 |
| DWLM         | R50      | 50 epochs  | 0.7643 | 0.9866 | 0.9122 | 0.3881 |
| DWLM w/ MF   | R50      | 50 epochs  | 0.7684 | 0.9886 | 0.9150 | 0.3803 |
| DWLM w/ MF   | R50      | 100 epochs | 0.7853 | 0.9878 | 0.9440 | 0.4261 |
| DWLM w/ 3DMF | R50      | 50 epochs  | -      | -      | -      | -      |
| DWLM w/ 3DMF | R50      | 100 epochs | 0.7828 | 0.9853 | 0.9373 | 0.4115 |

* PCB-Defects 資料集，本貢獻不使用，此資料集建立在人工瑕疵，模型訓練到後期會記住人工修改的特徵。

### 3.2 VOC 2007 + 2012


_待..._

---

## 5. 參考

1. [https://github.com/Megvii-BaseDetection/DeFCN](https://github.com/Megvii-BaseDetection/DeFCN)
2. [https://keras.io/examples/vision/retinanet/](https://keras.io/examples/vision/retinanet/)
3. [https://github.com/xuannianz/SAPD](https://github.com/xuannianz/SAPD)