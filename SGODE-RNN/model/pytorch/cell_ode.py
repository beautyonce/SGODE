import numpy as np
import torch
from lib import utils
from model.pytorch.ode import ODEBlock,ODEFunc_SGODE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class ODE_DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, input_dim, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True,embed_dim=10,Atype=2):
        """
        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_ode_for_gru: whether to use ode Graph convolution to calculate the reset and update gates.
        """
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        if input_dim:
            self._gode = ODEBlock(ODEFunc_SGODE(num_units+input_dim, 
                                            dropout=0.0, num_nodes=num_nodes, 
                                            embed_dim=embed_dim,Atype=Atype
                                            ),adjoint=False)  
        else:
            self._gode = ODEBlock(ODEFunc_SGODE(num_units, 
                                            dropout=0.0, num_nodes=num_nodes, 
                                            embed_dim=embed_dim,Atype=Atype
                                            ),adjoint=False)  
        self._use_ode_for_gru = True
 
        self.map = torch.nn.Sequential(
            torch.nn.Linear(num_units+input_dim,num_units+input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(num_units+input_dim,num_units+input_dim)
        )

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        
        fn = self._gconv
        
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size)) #[batchsize,207,128]
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units)) 
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs,  r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs,  state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1)) #[64,207,2]
        state = torch.reshape(state, (batch_size, self._num_nodes, -1)) #[64,207,64]
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) # [1,207,64*66]

        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        if self._use_ode_for_gru:
            if self._max_diffusion_step == 0:
                pass
            else:
                vtime = torch.linspace(start=0, end=1,steps=self._max_diffusion_step + 1).float()
                vtime = vtime.type_as(x)
                self._gode.odefunc.x0 = self.map(inputs_and_state)
                x = self._gode(vtime , inputs_and_state) #[num_matrices,batch_size, num_nodes, input_size]                
                x = x.permute(1,2,0,3)
                weights = self._gconv_params.get_weights((num_matrices,input_size, output_size))
                
                x = torch.einsum("bnkm,kmc->bnc", x, weights)
            
        biases = self._gconv_params.get_biases(output_size, bias_start) # 128
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
    
    

