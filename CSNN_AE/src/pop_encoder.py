"""
連続値を0か1のスパイク列に変化するクラス
population_encoding
"""

import torch
from torch import nn,Tensor
import numpy as np

EPSILON=0.01 #発火閾値

class pop_firing(torch.autograd.Function):
    """
    pop_encoder用の発火関数

    流れてきたlossをそのまま後に流す(0とか無限が掛からないようにする)
    """
    @staticmethod
    def forward(ctx,input):
        return input.gt(1-EPSILON).to(torch.float)

    @staticmethod
    def backward(ctx,out):
        grad_in=out.clone()
        return grad_in

class PopEncoder(nn.Module):
    def __init__(self,pop_dim,nt,mu,sigma,device="cpu") -> None:
        super(PopEncoder,self).__init__()

        self.nt=nt
        self.pop_dim=pop_dim
        self.input_dim=len(mu)
        self.device=device
        self.pop_firing=pop_firing.apply

        if pop_dim==1:
            self.mu=nn.Parameter(Tensor(mu,device=device))
            self.sigma=nn.Parameter(Tensor(sigma,device=device))
        elif pop_dim>1:
            alpha=3 #99.73%帯域
            mu_max=mu+sigma*alpha/2
            mu_min=mu-sigma*alpha/2
            mu_delta=(mu_max-mu_min)/(self.pop_dim-1)

            mu_=(np.array([mu_min+mu_delta*i for i in range(self.pop_dim)]).T).flatten()
            #print(f"mu_min:{mu_min}")
            #print(f"mu:{mu_}")
            sigma_=(np.array([sigma for _ in range(self.pop_dim)]).T).flatten()

            #print(mu_)
            self.mu=nn.Parameter(Tensor(np.array(mu_,dtype=np.float32),device=device)) #[(input_dim x pop_dim)]
            self.sigma=nn.Parameter(Tensor(np.array(sigma_,dtype=np.float32),device=device)) #[(input_dim x pop_dim)]

        
    def forward(self,x:Tensor):
        """
        -input-
            x:
                入力 [batch x input_dim]

        -output-
            xs:
                [nt x batch x (input_dim x pop_dim)]
        """

        batch_size,input_dim=x.shape

        #print(s)
        #--次元を増やすとき
        if self.pop_dim>1:
            x_=x.to("cpu").detach().numpy()
            x_=np.array([[x_[:,i] for _ in range(self.pop_dim)] for i in range(input_dim)])
            x_=x_.transpose(0,2,1)
            x_tmp=x_[0]
            for i in range(1,x_.shape[0]):
                x_tmp=np.concatenate([x_tmp,x_[i]],axis=1)
            x=x_tmp.copy()     
        #--

        x=Tensor(x)
        Ae=torch.exp(-0.5 * torch.pow((x-self.mu)/self.sigma,2)) #疑似シナプス項
        voltage=torch.zeros(size=(batch_size,input_dim*self.pop_dim),device=self.device)
        xs=torch.empty(size=(self.nt,batch_size,input_dim*self.pop_dim),device=self.device)

        for t in range(self.nt):
            voltage=voltage+Ae
            xs[t,:,:]=self.pop_firing(voltage)
            voltage=voltage - xs[t,:,:]*(1-EPSILON) #発火したら電圧をリセット 

        return xs