import torch
from torch import Tensor
from omegaconf import DictConfig

class PoissonSpikeEncoder():
    """
    定常ポアソンスパイクエンコーダー

    画像は0~255で範囲が決まってるし,わざわざpopSpikeでスパイクの仕方を学習する必要はない気がする
    """
    def __init__(self,cfg:DictConfig):
        """
        cfg:
            encoder:
                T:スパイク全時間(だいたい30とか)

                dt: 微小時間(基本的に1)
        """
        self.dt=cfg.encoder.dt
        self.nt=round(cfg.encoder.T/cfg.encoder.dt)

    @torch.no_grad()
    def __call__(self,img:torch.Tensor):
        """
        -input-
            img:[batch x channel x H x W]

        -return-
            poisson_spike:[nt x batch x channel x H x W]
        """
        batch,channel,img_size,_=img.shape
        poisson_spike=torch.where(
            torch.rand(size=(self.nt,batch,channel,img_size,img_size))>img*self.dt,1.0,0.0
            )

        return poisson_spike