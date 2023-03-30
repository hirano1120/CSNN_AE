import torch
from torch import Tensor,nn
from omegaconf import DictConfig

from ..csnn import CSNN
from ..encoder import PoissonSpikeEncoder
from ..decoder import Decoder


"""
入力画像を潜在変数に変換するエンコーダクラス
"""

class ImgCSNNEncoder(nn.Module):

    def __init__(self,cfg:DictConfig):
        """
        cfg:
            img_encoder:
                encoder:
                    dt: 1
                    T: 30

                csnn:
                    layer_num: 3
                    img_size: 64
                    in_channel: 1
                    conv:
                        hidden_channel: [2,8,16]
                        padding: [1,1,1]
                        stride: [1,1,1]
                        kernel: [3,3,3]
                    pool:
                        kernel: [2,2,2]

                decoder:
                    in_size: None
                    in_channel: ${csnn.conv.hidden_channel[-1]}
                    hidden_channel: [128,128] とか

        """
        super(ImgCSNNEncoder,self).__init__()

        self.poisson_encoder=PoissonSpikeEncoder(cfg)
        self.csnn=CSNN(cfg)
        # self.spike_mean=Decoder(cfg)
        # self.spike_logvar=Decoder(cfg=cfg)
        self.out=Decoder(cfg=cfg)

    #まずはDeterministicでやる
    def forward(self,img:Tensor):
        """
        -input-
            img:[batch x channel x H x W]
                255で割って正規化しとく
        -return-
            spike_img:[nt x batch x channel x H x W]

            latent
        
            z:[batch x channel x latent_size x latent_size]
        """
        spike_img=self.poisson_encoder(img=img) #画像をspike状に変換
        latent_spike:Tensor=self.csnn(spike_img) #潜在spikeの抽出 これを使って元のspike_imgにもどす
        
        #latent_spike_=copy.deepcopy(latent_spike.detach()) #dynamicsから見る潜在スパイクは定数
        #追記1 計算グラフを繋げないと,latent_spikeは潜在だけど, 発火率に直したものが潜在じゃなくなってしまう
        z=self.out.forward(latent_spike)

        return spike_img,latent_spike,(z)


    # def forward(self,img:Tensor):
    #     """
    #     -input-
    #         img:[batch x channel x H x W]
    #             255で割って正規化しとく
    #     -return-
    #         spike_img:[nt x batch x channel x H x W]

    #         latent_spike:[nt x batch x channel x latent_size x latent_size]
        
    #         z:[batch x (channel x latent_size x latent_size)]

    #         z_mean:

    #         z_logvar:
    #     """
    #     spike_img=self.poisson_encoder(img=img) #画像をspike状に変換
    #     latent_spike:Tensor=self.csnn(spike_img) #潜在spikeの抽出 これを使って元のspike_imgにもどす
    #     z_mean=self.spike_mean(latent_spike)
    #     z_logvar=self.spike_logvar(latent_spike)

    #     eps=torch.randn_like(z_logvar)
    #     z:Tensor=z_mean + (torch.exp(0.5*z_logvar))*eps

    #     z_=copy.deepcopy(z.detach()) #ここで計算グラフを切る. zはdynamicsに入力するだけ

    #     return spike_img,latent_spike,self.flatten(z_),self.flatten(z_mean),self.flatten(z_logvar)
