import torch
from torch import Tensor,nn
from omegaconf import DictConfig

from .AE_encoder import ImgCSNNEncoder
from .AE_decoder import ImgCSNNDecoderv3


class ImgCSNNAEv3(nn.Module):
    """
    【v3】
    時間軸の復元方法をpopulation codingにしてみる
    """

    def __init__(self,cfg:DictConfig):
        """
        cfg:
            type: CSNN

            img_decoder:

            img_encoder:
        """
        super(ImgCSNNAEv3,self).__init__()

        #print(cfg.img_encoder)

        self.img_encoder=ImgCSNNEncoder(cfg=cfg.img_encoder)
        self.img_decoder=ImgCSNNDecoderv3(cfg=cfg.img_decoder)
        self.flatten=nn.Flatten()


    def forward(self,img:torch.Tensor):
        """
        -input-
            img:[batch x channel x H x W]
        -reurn-
            predict_spike:[nt x batch x channel x H x W]
                decoderで戻したスパイクimage
            spike_img:[nt x batch x channel x H x W]
                poisson_encoderで画像をspike化したもの.これがdecoderの教師spikeになる
            z:[batch x (channel x latent_size x latent_size)]
                latent_spikeを発火率に変換し,flattenかけたもの
        """

        #spike_img,latent_spike,z,z_mean,z_logvar=self.img_encoder.forward(img)
        spike_img,latent,z=self.img_encoder.forward(img)

        predict_spike=self.img_decoder(z)

        return predict_spike,spike_img,self.flatten(z)

    def loss_func(self,target_spike:Tensor,predict_spike:Tensor):
        """
        -return-
            spike_loss:スカラ
        """

        target_r=double_exp_filter(spike=target_spike)
        predict_r=double_exp_filter(spike=predict_spike)
        loss_reconst=(target_r-predict_r)**2
        loss_reconst=torch.sum(loss_reconst,dim=0) #時間方向には足す
        loss_reconst=torch.sum(loss_reconst,dim=1) #channel方向には足す

        loss=torch.mean(loss_reconst) 
        return loss

    def encode(self,img:Tensor):
        """
        -input-
            img:[batch x channel x H x W]
        -return-
            z:[batch x channel x latent_size x latent_size]
                latent_spikeを発火率に変換したもの
        """

        _,_,z=self.img_encoder.forward(img)

        return self.flatten(z)


def double_exp_filter(spike:Tensor,dt=0.05,td=20,tr=2):
    """
    スパイクのlossを計算するときに使う

    spikeを2重指数フィルターで畳み込んでから2乗誤差をとる

    これをvan Rossum距離という

    -input-
        spike:[nt x batch x channel x H x W]

    -return-
        r:[nt x batch x channel x H x W]
            畳み込み後の電流
    """

    nt,batch,channel,size,_=spike.shape
    r=torch.zeros(size=(nt,batch,channel,size,size))
    hr=torch.zeros_like(r)

    for t in range(nt):
        if t==0:
            r[t]=r[t]*(1-dt/tr) + hr[t]*dt
            hr[t]=hr[t]*(1-dt/td) + spike[t]/(tr*td)
        elif t>0:
            r[t]=r[t-1]*(1-dt/tr) + hr[t-1]*dt
            hr[t]=hr[t-1]*(1-dt/td) + spike[t]/(tr*td)

    return r

