"""
発火率デコーダー
参考論文 : https://arxiv.org/pdf/2010.09635.pdf
Tang, Guangzhi and Kumar, Neelesh and Yoo, Raymond and Michmizos, Konstantinos P.,
Deep Reinforcement Learning with Population-Coded Spiking Neural Network for Continuous Control,2020
"""

import torch
from torch import Tensor,nn
from omegaconf import DictConfig
from math import sqrt

class Decoder(nn.Module):

    def __init__(self,cfg:DictConfig):
        super(Decoder,self).__init__()
        """
        ★必ずCSNNをインスタンス化してからDecoderをインスタンス化する.

          そうしないと,Decoderのinput_sizeが分からない
        cfg:
            decoder:
                in_size: None(config書くときはNone. CSNNのインスタンス作成時に書き込まれる)

                in_channel:
        """
        n=cfg.decoder.in_size
        self.in_channel=cfg.decoder.in_channel
        self.output_size=n*n*self.in_channel


        mean=torch.zeros(size=(n,n))
        std=sqrt(2/(n+1e-16))*torch.ones(size=(n,n))

        self.w=nn.Parameter(torch.normal(mean=mean,std=std)) #Heの初期値で初期化
        self.b=nn.Parameter(torch.zeros_like(self.w))
        

    def forward(self,input_spike:torch.Tensor)->Tensor:
        """
        -input-
            input_spike:[nt x batch x channel x H x W]

        -return-
            out:[batch x channel x H x W]
        """
        nt=input_spike.shape[0]
        firing_rate=torch.sum(input_spike,dim=0)/nt #発火率計算
        out=firing_rate*self.w + self.b #要素積&bais加算

        return out


        