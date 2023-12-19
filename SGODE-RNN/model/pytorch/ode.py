import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode
class ODEFunc_SGODE(nn.Module):  
    def __init__(self, hidden_size, dropout, num_nodes,embed_dim,Atype):
        super(ODEFunc_SGODE, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.x0 = torch.zeros(64, num_nodes, hidden_size)
          
        softmax = nn.Softmax(dim=0)
        self.node_embeddings1 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
        initial_type=Atype
        if initial_type==1:
            self.node_embeddings3 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
            self.node_embeddings4 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
        elif initial_type==2:
            self.node_embeddings3 = nn.Parameter(1e-6*(torch.ones(num_nodes, embed_dim)), requires_grad=True)
            self.node_embeddings4 = nn.Parameter(1e-6*(torch.ones(num_nodes, embed_dim)), requires_grad=True)
        self.C = nn.Parameter(softmax(torch.rand(num_nodes)), requires_grad=True)                  
        self.relu = nn.ReLU()
        self.wt = nn.Linear(hidden_size, hidden_size)

    def forward(self, t, x):  
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        pos = torch.mm(self.node_embeddings1, self.node_embeddings2.transpose(0, 1))
        pos = self.relu(pos)
        
        neg = torch.mm(self.node_embeddings3, self.node_embeddings4.transpose(0, 1))    
        neg = self.relu(neg)               
        K_weight = pos - neg
        self_x = self.C.reshape(1,-1,1) * x
        x = torch.einsum("nm,bmc->bnc", K_weight, x) 
        x = x + self_x 
        x = self.wt(x)
        x = x + self.x0     
        x = self.dropout_layer(x)
        x = F.relu(x)
        return x

class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-2, atol=1e-3, method='euler', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10