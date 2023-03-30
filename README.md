# README

## SNNで構成された畳み込みオートエンコーダ

畳み込みオートエンコーダをスパイキングニューラルネットワーク(SNN)で表現

SNNによるオートエンコーダが見つからなかったため自作

## Requirements

- torch==1.11.0
- omegaconf==2.1.2
- hydra-core==1.0.3
- numpy
- pillow
- matplotlib

## Sample

#### オートエンコーダの学習

~~~ 
python CSNN_AE/sample/train_auto_encoder.py
~~~


#### 学習結果の確認

~~~
python CSNN_AE/sample/test_auto_encoder.py
~~~

大きなガウシアンノイズがかかっても、元の画像が再構成される

![alt](/CSNN_AE/sample/reconst_imgs/reconst_img_epoch5000.png)
