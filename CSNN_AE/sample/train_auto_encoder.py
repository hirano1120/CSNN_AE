"""
畳み込みスパイキングニューラルネットワークで構成されたオートエンコーダの学習
"""

import os
import sys
from pathlib import Path
ROOT=Path(__file__).parent.parent
PARENT=Path(__file__).parent
sys.path.append(str(ROOT))
sys.path.append(str(PARENT))

import torch
from torch import Tensor
import numpy as np
from omegaconf import DictConfig
import hydra
from PIL import Image
import matplotlib.pyplot as plt 

from src import ImgCSNNAEv3


COLOR="\033[35m"
RESET="\033[0m"


@hydra.main(config_path="conf",config_name="main")
def main(cfg:DictConfig):

    train_img_path=os.listdir(f"{str(PARENT)}/train_imgs")
    train_data=[np.array(Image.open(f"{str(PARENT)}/train_imgs/{img_path}").convert("L").resize((128,128))) for img_path in train_img_path] #学習データ
    train_data=Tensor(np.array(train_data)).unsqueeze(dim=1)/255.0 #学習データをテンソル化＆正規化
    

    auto_encoder=ImgCSNNAEv3(cfg=cfg.auto_encoder) #オートエンコーダのインスタンス化
    optimizer=torch.optim.Adam(params=auto_encoder.parameters(),lr=0.003) #ネットワークの最適化アルゴリズム


    ###学習開始###
    epoches=5000
    save_interval=1000
    auto_encoder.train()
    for epoch in range(epoches):

        img_reconst,img_org,_=auto_encoder.forward(img=train_data) #画像の再構成
        loss_reconst=auto_encoder.loss_func(target_spike=img_org,predict_spike=img_reconst) #もとの画像と再構成画像の誤差
        loss_reconst.backward() #誤差逆伝播
        optimizer.step() #オートエンコーダの重み更新
        optimizer.zero_grad() #誤差の重み微分をリセット

        print(f"{COLOR}[{epoch+1}/{epoches}] EPOCH LOSS:{RESET}{loss_reconst.item()}") #lossの表示
        if (epoch+1)%save_interval==0:
            torch.save(auto_encoder.state_dict(),f=f"{PARENT}/params/param_epoch{epoch+1}.pth") #パラメータの保存

    ###学習終了###


if __name__=="__main__":
    main()