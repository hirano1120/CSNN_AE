"""
畳み込みスパイキングニューラルネットワーク
"""

import torch
from torch import Tensor,nn
from math import sqrt
from omegaconf import DictConfig

import time

CYAN="\033[36m"
RESETCOL='\033[0m'

FIRING_THRESHOLD_VOLTAGE=1 #発火閾電圧 [1.5, 2.0, 0.2]のどれか
SYNAPS_DECAY=0.5 #シナプス電流の時定数
NEURON_DECAY=3/4 #ニューロンの時定数


def get_conv_out_size(img:Tensor,conv:nn.Conv2d):
    """
    conv2d&pooling2dの出力サイズを調べる関数

    -reutrn-
        conv(img),conv(img).size
    """
    img=conv(img)

    return img, img.shape[2]


class Firing_func(torch.autograd.Function):
    """
    発火用関数.
    これだけbackwardをカスタムすれば,
    あとはtorchが勝手に自動微分してくれる
    """

    def forward(ctx,neuron_voltage):
        ctx.save_for_backward(neuron_voltage)
        return torch.where(neuron_voltage>FIRING_THRESHOLD_VOLTAGE,1.0,0.0)

    
    def backward(ctx,grad_out:Tensor):
        """
        論文の式(26)で近似してみる
        """
        neuron_voltage,=ctx.saved_tensors
        grad_input=grad_out.clone()
        a4=1 #係数
        spike_grad=(1/sqrt(2*torch.pi*a4)) * torch.exp( -(neuron_voltage-FIRING_THRESHOLD_VOLTAGE)**2 / (2*a4) ) #論文の式(26)
        #spike_grad=(torch.abs(neuron_voltage-FIRING_THRESHOLD_VOLTAGE)<0.5).float()
        return  grad_input*spike_grad


class CSNN(nn.Module):

    def __init__(self,cfg:DictConfig):
        """
        cfg:
            csnn:
                layer_num:レイヤー数

                img_size:画像サイズ(高さor幅)

                in_channel: 画像のchannel数(RGBなら3,グレースケールなら1)

                conv:
                    hidden_channel:[4,5,1]みたいな感じ(list)

                    padding:

                    kernel:

                    stride:

                pool:
                    kernel:
        """
        super(CSNN,self).__init__()
        
        self.layer_num=cfg.csnn.layer_num
        self.channel=cfg.csnn.conv.hidden_channel

        #--conv synapsレイヤーの作成
        #--変数名がconfigによって変わるからexec使ってる
        self.synaps_layers=[]
        for i in range(self.layer_num):

            if i==0:
                exec(f"""self.conv{i}=nn.Conv2d(
                    in_channels=cfg.csnn.in_channel,
                    out_channels=cfg.csnn.conv.hidden_channel[i],
                    kernel_size=cfg.csnn.conv.kernel[i],
                    padding=cfg.csnn.conv.padding[i],
                    stride=cfg.csnn.conv.stride[i])"""
                    )
            else:
                exec(f"""self.conv{i}=nn.Conv2d(
                    in_channels=cfg.csnn.conv.hidden_channel[i-1],
                    out_channels=cfg.csnn.conv.hidden_channel[i],
                    kernel_size=cfg.csnn.conv.kernel[i],
                    padding=cfg.csnn.conv.padding[i],
                    stride=cfg.csnn.conv.stride[i])"""
                    )
            if not len(cfg.csnn.pool.kernel)==0:
                exec(f"self.pool{i}=nn.AvgPool2d(kernel_size=cfg.csnn.pool.kernel[i])")
                exec(f"self.synaps{i}=torch.nn.Sequential(self.conv{i},self.pool{i})")
            else:
                exec(f"self.synaps{i}=torch.nn.Sequential(self.conv{i})")

            exec(f"self.synaps_layers.append(self.synaps{i})")
        #--

        #--conv2dの出力サイズを調べて保存(ニューロン電圧とかのtensorつくる時に使う)
        img=torch.empty(size=(1,cfg.csnn.in_channel,cfg.csnn.img_size,cfg.csnn.img_size))
        self.conv_out_size=[]
        for i in range(self.layer_num):
            img,out_size=get_conv_out_size(img=img,conv=self.synaps_layers[i])
            self.conv_out_size.append(out_size)
        cfg.decoder.in_size=self.conv_out_size[-1] #decoderのインプットサイズの書き込み
        cfg.decoder.in_channel=self.channel[-1]
        #--
        
        self.firing_func=Firing_func.apply


        #--階層構造を表示
        print("\n","="*5,f"{CYAN}CSNN STRUCT{RESETCOL}","="*30)
        i=0
        for syn,out_size in (zip(self.synaps_layers,self.conv_out_size)):
            print(f"{CYAN}csnn{i+1}:{RESETCOL}",syn,f"{CYAN}out channel:{RESETCOL}{self.channel[i]}",f"{CYAN}out size:{RESETCOL}{out_size} x {out_size}")
            if not i==self.layer_num-1:
                print("---")
            i+=1
        print("="*50,"\n")
        #time.sleep(3)
        #--

    def neuron_model(self,synaps_func,pre_layer_spike,pre_time_current,pre_time_volt,pre_time_spike):
        """
        シナプス+ニューロン項

        -input-
            synaps_func:
                入力spikeに重みとバイアスを加える関数.

                nn.Linearをそのままぶち込めばいい
            pre_layer_spike:
                前のlayerの現timestepのspike
            pre_time_current:
                このlayerの前timestepのシナプス電流
            pre_time_volt:
                このlayerの前timestepのニューロン電圧
            pre_time_spike:
                このlayerの前timestepのspike

        -return-
            current,volt,spike

            [batch x dim]
        """
        current=synaps_func(pre_layer_spike) #軸索(&シナプス電流計算)
        current=pre_time_current*SYNAPS_DECAY + current #シナプス電流の畳み込み. ただ,この論文ではやってない. (super spikeとかだとやるからやったほうがいいとは思う)
        volt=pre_time_volt*NEURON_DECAY*(1-pre_time_spike) + current #ニューロン電圧の計算
        spike=self.firing_func(volt) #発火

        return current,volt,spike #次のtimestepのために'シナプス電流,ニューロン電圧,spike'を出力する
        

    def forward(self,input_spike:torch.Tensor):
        """
        -input-
            input_spike:[nt x batch x channel x H x W ]
        -return-
            out_spike:[nt x batch x channel x H x W]
        """

        nt,batch,_,_,_=input_spike.shape

        #--シナプス電流,ニューロン電圧,出力スパイクの設定
        #--全部sizeは[layer_num x [batch x channel x img_size x img_size]]
        current,voltage,spike=[],[],[]
        for i in range(self.layer_num):
            current.append(torch.zeros(size=(batch,self.channel[i],self.conv_out_size[i],self.conv_out_size[i])))
            voltage.append(torch.zeros(size=(batch,self.channel[i],self.conv_out_size[i],self.conv_out_size[i])))
            spike.append(torch.zeros(size=(batch,self.channel[i],self.conv_out_size[i],self.conv_out_size[i])))
        #--

        #--最終層のspikeだけ時系列保持
        out_spike=torch.zeros(
            size=(nt,batch,self.channel[-1],self.conv_out_size[-1],self.conv_out_size[-1])
            )
        #--

        for t in range(nt):

            spike_t=input_spike[t,:,:,:,:]

            for layer_i in range(self.layer_num):

                if layer_i==0:
                    current[0],voltage[0],spike[0]=self.neuron_model(
                        synaps_func=self.synaps_layers[layer_i],pre_layer_spike=spike_t,
                        pre_time_current=current[layer_i],pre_time_volt=voltage[layer_i],pre_time_spike=spike[layer_i]
                    )
                elif layer_i>0:
                    current[layer_i],voltage[layer_i],spike[layer_i]=self.neuron_model(
                        synaps_func=self.synaps_layers[layer_i],pre_layer_spike=spike[layer_i-1],
                        pre_time_current=current[layer_i],pre_time_volt=voltage[layer_i],pre_time_spike=spike[layer_i]
                    )
            out_spike[t]=out_spike[t]+spike[-1]

        return out_spike #最終層のスパイク列を出力

