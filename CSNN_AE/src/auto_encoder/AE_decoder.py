import torch
from torch import Tensor,nn
from omegaconf import DictConfig
import numpy as np

from ..csnn_transpose import CSNNTranpose
from ..pop_encoder import PopEncoder

"""
潜在変数からもとの画像を復元するデコーダクラス

population codingで発火率から時間軸を復元する
"""


class ImgCSNNDecoderv3(nn.Module):

    def __init__(self,cfg:DictConfig):
        """
        cfg:
            img_decoder:
                type: CSNN

                nt:

                csnn_transpose:
                    layer_num:レイヤー数(1以上)

                    latent_channel: 潜在画像のchannel数

                    latent_size: 潜在画像のサイズ

                    trans: 全て1つ以上の要素を持つ
                        hidden_channel:[4,5,1](list)
                            hidden_channelの最後は元の画像のchannel数

                        kernel: 基本的にkernel=strideにする.
                            そうすると,Hout=Hin x stride で計算が簡単になる

                        stride:
        """
        super(ImgCSNNDecoderv3,self).__init__()

        in_size=cfg.csnn_transpose.latent_channel * (cfg.csnn_transpose.latent_size)**2
        self.nt=cfg.nt
        self.pop_encoder=PopEncoder(
            pop_dim=1,nt=self.nt,device=torch.Tensor([0,0]).device,
            mu=np.zeros(in_size),sigma=2.0*np.ones(in_size),
        )
        self.img_decoder=CSNNTranpose(cfg=cfg)
        self.latent_size=cfg.csnn_transpose.latent_size
        self.latent_channel=cfg.csnn_transpose.latent_channel
        self.flatten=torch.nn.Flatten()

    def forward(self,z:torch.Tensor):
        """
        -input-
            z:[batch x channel x H x W]
                発火率に変換し,時間方向の情報を失った潜在
        """
        
        batch,channel,H,W=z.shape
        z=self.flatten(z) #pop_encoderに入力するために一旦平たくする
        pop_spike=self.pop_encoder.forward(x=z) #pop_encoderでspikeを復元
        pop_spike=pop_spike.view(self.nt,batch,channel,H,W) #画像次元に直す
        spike_reconst=self.img_decoder.forward(pop_spike)

        return spike_reconst