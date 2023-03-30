"""
学習したオートエンコーダのテスト
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
from matplotlib.colors import Normalize

from src import ImgCSNNAEv3


COLOR="\033[35m"
RESET="\033[0m"


def noise_filter(img:np.ndarray):
    """
    img:
        ０～１に正規化しておく
    """
    noise=np.random.normal(loc=0,scale=0.9,size=img.shape)
    img_noised=img+noise
    img_noised[img_noised>1]=1.0
    img_noised[img_noised<0]=0.0

    return img_noised


@hydra.main(config_path="conf",config_name="main")
def main(cfg:DictConfig):

    train_img_path=os.listdir(f"{str(PARENT)}/train_imgs")
    train_data=np.array([np.array(Image.open(f"{str(PARENT)}/train_imgs/{img_path}").convert("L").resize((128,128))) for img_path in train_img_path]) #学習データ
    

    auto_encoder=ImgCSNNAEv3(cfg=cfg.auto_encoder) #オートエンコーダのインスタンス化
    auto_encoder.load_state_dict(torch.load(f"{str(PARENT)}/params/param_epoch5000.pth")) #学習済みを読み込む

    
    test_idx=np.random.randint(low=0,high=train_data.shape[0])
    test_img=train_data[test_idx]/255.0 #テスト用のデータを1つ選ぶ&正規化

    test_img=noise_filter(img=test_img) #ガウシアンノイズをかける
    test_img=Tensor(test_img).unsqueeze(0).unsqueeze(0) #テンソル化＆チャンネル次元とバッチ次元追加


    img_reconst,img_org,_=auto_encoder.forward(img=test_img) #画像の再構成


    fig,axs=plt.subplots(nrows=1,ncols=3)
    xlabels=["original image","input image","reconstruct image"]
    for i,ax in enumerate(axs):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabels[i])
    axs[0].imshow(train_data[test_idx]/255.0,cmap="gray",norm=Normalize(vmin=0,vmax=1))
    axs[1].imshow(test_img.detach().numpy()[0,0],cmap="gray",norm=Normalize(vmin=0,vmax=1))
    axs[2].imshow(1.0-np.mean(img_reconst.detach().numpy()[:,0,0],axis=0),cmap="gray",norm=Normalize(vmin=0,vmax=1))
    plt.show()


if __name__=="__main__":
    main()