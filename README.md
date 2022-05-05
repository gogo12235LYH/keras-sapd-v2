# One Stage Detector (SAPD w/ Dynamic Weighted Loss Mask, Lite Version of SAPD)

---

* 透過動態損失權重遮罩，可在相同精度下得到最佳的訓練時間成本
* [PCB Defect Detection Based on SAPD with Mix Subnetwork](https://github.com/gogo12235LYH/keras-pcb-sapd-mix)
* Tensorflow 2.6.0
* 3DMF, [End-to-End Object Detection with Fully Convolutional Network](https://arxiv.org/abs/2012.03544)

---

![image](https://github.com/gogo12235LYH/keras-sapd-v2/blob/main/images/v2.png)

## Updates

1. 2022-04-06 -> First Commit.
2. 2022-04-09 -> DWLM w/ MF on 100 eps.
3. 2022-04-11 -> Fixed NaN ( tf.reduced_sum 問題 )
4. 2022-04-28 -> 修正 keras sequence generator (現在可使用 tf.data 或 keras.sequence 囉)

## ToDo

1. ~~Keras Sequence generator~~
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

---

## 3. 評估

* DWLM: Dynamic Weighted Loss Mask
* MF: MaxFilter ( Single Level Pooling )
* 3DMF: 3DMaxFilter ( Three Level Pooling ) 參考 5.1

更改 evaluation.py 中切入點中的 model_weight_path 權重路徑即可。
```python
if __name__ == '__main__':
    init_()
    config.EVALUATION = 1
    main(
        model_weight_path='20220427-DPCB-HStd-SGDWE100BS2B1R50D4.h5'
    )
```

### 3.1 Deep PCB:
* 訓練及評估影像大小: 640 * 640 ( PHI=1 )，就那個黑白的 PCB 瑕疵資料集
* 在 50 epochs 下，原始 SAPD 訓練時間約為 1.9 小時; DWLM 約為 1.1小時，供參考(RTX 3060 6g)
* 下表展示訓練結果，將原本需訓練的 FSN 優化為 DWLM 提升效果非常穩健。並搭配 MF 及 3DMF 效果也相當出色

| subnetworks  | backbone | setting    | mAP    | AP.5   | AP.75  | AP.9   |
|--------------|----------|------------|--------|--------|--------|--------|
| SAPD - org   | R50      | 50 epoch   | 0.7530 | 0.9751 | 0.8881 | 0.3518 |
| DWLM         | R50      | 50 epoch   | 0.7643 | 0.9866 | 0.9122 | 0.3881 |
| DWLM w/ MF   | R50      | 50 epoch   | 0.7684 | 0.9886 | 0.9150 | 0.3803 |
| DWLM w/ MF   | R50      | 100 epoch  | 0.7853 | 0.9878 | 0.9440 | 0.4261 |
| DWLM w/ 3DMF | R50      | 100 epoch  | 0.7828 | 0.9853 | 0.9373 | 0.4115 |


* 我將想法退回到 FCOS 的 centerness branch 上，搭配與分類子網路相乘的 3DMF 優化方法。
* DWLM* 為取前三高權重作為soft-anchor，加上 centerness branch 與分類子網路兩者輸出sigmoid 再相乘之優化子網路。

| subnetworks | backbone | setting    | mAP        | AP.5       | AP.75      | AP.9       |
|-------------|----------|------------|------------|------------|------------|------------|
| DWLM        | R50      | 50 epoch   | 0.7643     | **0.9866** | 0.9122     | 0.3881     |
| DWLM*       | R50      | 50 epoch   | **0.8094** | 0.9838     | **0.9422** | **0.4710** |
| DWLM*       | R50      | 100 epoch  | **0.8194** | 0.9849     | **0.9449** | **0.4981** |

---

## 5. 參考

1. [https://github.com/Megvii-BaseDetection/DeFCN](https://github.com/Megvii-BaseDetection/DeFCN)
2. [https://keras.io/examples/vision/retinanet/](https://keras.io/examples/vision/retinanet/)
3. [https://github.com/xuannianz/SAPD](https://github.com/xuannianz/SAPD)