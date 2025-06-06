import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, ChebConv, TransformerConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros

from model.graph_functions import image_to_graph, flatten, Graph, unflatten
from model.utils import add_positional_encoding

import gc 
import random
import numpy as np
import psutil
import os

from model.model import GConvLSTM, GConvLSTM_Simple, GConvGRU, SplitGConvLSTM, DummyLSTM, CONVOLUTION_KWARGS, CONVOLUTIONS

class NoGConvLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(NoGConvLSTM, self).__init__(*args, **kwargs)

    def forward(self, input, edge_index=None, edge_weight=None, H=None, C=None):
        # Simply bypass a and b and call the parent class's forward method
        o, (h, c) = super(NoGConvLSTM, self).forward(input.unsqueeze(0), (H.unsqueeze(0), C.unsqueeze(0)))
        return o.squeeze(0), h.squeeze(0), c.squeeze(0)


class Encoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1, conv_type='GCNConv', rnn_type='LSTM', n_conv=3, dummy=False):
        super().__init__()

        assert rnn_type in ['GRU', 'LSTM', 'SplitLSTM', 'NoConvLSTM']

        if rnn_type == 'LSTM' or rnn_type == 'NoConvLSTM':
            rnn = GConvLSTM
        elif rnn_type == 'GRU':
            rnn = GConvGRU
        elif rnn_type == 'SplitLSTM':
            rnn = SplitGConvLSTM
        else:
            raise ValueError
        
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dummy = dummy

        if not self.dummy:
            self.rnns = nn.ModuleList(
                [rnn(input_features, hidden_size, conv_type=conv_type, n_conv=n_conv, name='encoder')] + \
                [rnn(hidden_size, hidden_size, conv_type=conv_type, n_conv=n_conv, name='encoder') for _ in range(n_layers-1)]
            )
        
        self.dropout = nn.Dropout(dropout)

        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        
    def forward(self, X, edge_index, edge_weight, H=None, C=None):

        if self.dummy:
            return H, C

        X = X.squeeze(0)

        _, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H, C=C)
        hidden_layer = self.dropout(hidden_layer)

        hidden_layer, cell_layer = hidden_layer.squeeze(0), cell_layer.squeeze(0)

        # First layer
        hidden_layer = self.norm_h(hidden_layer)
        if self.rnn_type != 'GRU':
            cell_layer = self.norm_c(cell_layer)  # GRU does not have a cell state

        # Subsequent layers
        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            _, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=None, C=None)
            hidden_layer = self.dropout(hidden_layer)

            hidden_layer = self.norm_h(hidden_layer)
            if self.rnn_type != 'GRU':
                cell_layer = self.norm_c(cell_layer)  # GRU does not have a cell state

            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell) if self.rnn_type != 'GRU' else [None]*len(hidden)  # GRU does not have a cell state
        return hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1, concat_layers_dim=3, conv_type='GCNConv', rnn_type='LSTM', n_conv=3, binary=False, dummy=False, multitask=True):
        super().__init__()

        assert rnn_type in ['GRU', 'LSTM', 'SplitLSTM', 'NoConvLSTM']

        if rnn_type == 'LSTM':
            rnn = GConvLSTM
        elif rnn_type == 'GRU':
            rnn = GConvGRU
        elif rnn_type == 'SplitLSTM':
            rnn = SplitGConvLSTM
        elif rnn_type == 'NoConvLSTM':
            rnn = NoGConvLSTM
        else:
            raise ValueError
        
        self.rnn_type = rnn_type
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.binary = binary
        self.dummy = dummy
        self.multitask = multitask

        n_conv = 1  # Hard-coded single convolutional layer in the decoder (TODO: make this not hard coded.)

        if not self.dummy:
            if rnn_type == 'NoConvLSTM':
                self.rnns = nn.ModuleList([NoGConvLSTM(input_features, hidden_size, 1)] +  [NoGConvLSTM(hidden_size, hidden_size, 1) for _ in range(n_layers-1)]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            else:
                self.rnns = nn.ModuleList(
                    [rnn(input_features, hidden_size, conv_type=conv_type, n_conv=n_conv, name='decoder')] + \
                    [rnn(hidden_size, hidden_size, conv_type=conv_type, n_conv=n_conv, name='decoder') for _ in range(n_layers-1)]
                    )

        # Dummy convolutions don't project the data into a new dimensionality
        # in_channels = hidden_size + 0 + concat_layers_dim if not (dummy or conv_type=='Dummy') else 3 + 0 + concat_layers_dim
        in_channels = hidden_size + 0 + 0 if not (dummy or conv_type=='Dummy') else 3 + 0 + 0

        conv_func = CONVOLUTIONS[conv_type]
        conv_func_kwargs = CONVOLUTION_KWARGS[conv_type]
        
        outdim = 2 if self.multitask else 1
        self.gnn_out1 = conv_func(in_channels=in_channels, out_channels=hidden_size, **conv_func_kwargs)
        self.gnn_out2 = conv_func(in_channels=hidden_size, out_channels=outdim, **conv_func_kwargs)  # MULTITASK
        
        # Uncomment to initialize weights to 0
        # torch.nn.init.zeros_(self.fc_out1.weight)
        # torch.nn.init.zeros_(self.fc_out1.bias)
        # torch.nn.init.zeros_(self.fc_out2.weight)
        # torch.nn.init.zeros_(self.fc_out2.bias)
        # torch.nn.init.zeros_(self.fc_out3.weight)
        # torch.nn.init.zeros_(self.fc_out3.bias)
        
        self.gnn1 = conv_func(in_channels=concat_layers_dim, out_channels=hidden_size, **conv_func_kwargs)

        self.norm_o = nn.LayerNorm(hidden_size)
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, edge_index, edge_weight, concat_layers, y_initial, H, C):

        if self.dummy:
            if concat_layers is not None:
                X = torch.cat([X, concat_layers], dim=-1)
            X = self.gnn_out(X, edge_index, edge_weight)
            return X, H, C
        
        # First layer
        output, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H[0], C=C[0])
        hidden_layer = self.dropout(hidden_layer)
        output = self.dropout(output)

        hidden_layer = self.norm_h(hidden_layer)
        if self.rnn_type != 'GRU':
            cell_layer = self.norm_c(cell_layer)  # GRU does not have a cell state

        # Subsequent layers
        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            output, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=H[i], C=C[i])

            hidden_layer = self.norm_h(hidden_layer)
            
            if self.rnn_type != 'GRU':
                cell_layer = self.norm_c(cell_layer)  # GRU does not have a cell state

            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell) if self.rnn_type != 'GRU' else [None]*len(hidden)  # GRU does not have a cell state

        # Use top layer's output
        output = F.leaky_relu(output)

        # Concatenate with the concat layers
        if concat_layers is not None:
            gnn_output = self.gnn(concat_layers, edge_index, edge_weight)
            output = output + gnn_output

        # Pass output through the final GNN to reduce to desired dimensionality
        output = self.mlp_out(output, edge_index, edge_weight)

        # Squeeze everything to (-1, 1) 
        if not self.multitask:
            output = torch.tanh(output)
        
        # Add to previous step's SIC map, OR to the launch date's SIC map (not used)
        output = output + y_initial
        
        # MULTITASK
        if self.multitask:
            sic = torch.sigmoid(output[..., [0]])
            sip = output[..., [1]]
            output = torch.cat((sic, sip), dim=-1)

        if self.binary:
            output = torch.sigmoid(output)

        return output, hidden, cell

    def mlp_out(self, x, edge_index, edge_weight):
        x = self.gnn_out1(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.gnn_out2(x, edge_index, edge_weight)
        x = self.dropout(x)
        return x
    
    def gnn(self, x, edge_index, edge_weight):
        x = self.gnn1(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        return x        

class Seq2Seq(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 thresh,
                 input_timesteps=3,
                 input_features=4, #3 node_size
                 output_timesteps=5,
                 n_layers=4,
                 n_conv=2,
                 transform_func=None,
                 condition='max_larger_than',
                 remesh_input=False,
                 conv_type='ChebConv',
                 rnn_type='LSTM',
                 binary=False,
                 image_shape=None,
                 dummy=False,
                 device=None,
                 debug=False,
                 multitask=True):
        super().__init__()
        
        self.encoder = Encoder(
            input_features,
            hidden_size,
            dropout,
            n_layers=n_layers,
            conv_type=conv_type,
            rnn_type=rnn_type,
            n_conv=n_conv,
            dummy=dummy,
            )
        self.decoder_1 = Decoder(
            1+3 if not multitask else 2+3,  # 1 output variable + 3 (positional encoding and node_size)  # MULTITASK
            hidden_size,
            dropout,
            n_layers=n_layers,
            concat_layers_dim=2,
            conv_type=conv_type,
            rnn_type=rnn_type,
            n_conv=n_conv,
            binary=binary,
            dummy=dummy,
            multitask=multitask
            )
        
        # this was previously used to have multiple decoders for different forecast steps
        # self.decoder_2 = Decoder(
        #     1+3,  # 1 output variable + 3 (positional encoding and node_size)
        #     hidden_size,
        #     dropout,
        #     n_layers=n_layers,
        #     concat_layers_dim=2,
        #     conv_type=conv_type,
        #     rnn_type=rnn_type,
        #     n_conv=n_conv,
        #     binary=binary,
        #     dummy=dummy,
        #     )
        
        # self.decoder_3 = Decoder(
        #     1+3,  # 1 output variable + 3 (positional encoding and node_size)
        #     hidden_size,
        #     dropout,
        #     n_layers=n_layers,
        #     concat_layers_dim=2,
        #     conv_type=conv_type,
        #     rnn_type=rnn_type,
        #     n_conv=n_conv,
        #     binary=binary,
        #     dummy=dummy,
        #     )


        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers
        self.condition = condition
        self.remesh_input = remesh_input
        self.debug = debug
        self.image_shape = image_shape

        self.conv_type = conv_type
        self.multitask = multitask

        # These convolutions can accept edge attributes, the others cannot.
        if conv_type in ['MHTransformerConv', 'TransformerConv', 'GATConv']:
            self.use_edge_attrs = True
        else:
            self.use_edge_attrs = False

        self.thresh = thresh
        self.transform_func = transform_func
        self.dropout = nn.Dropout(dropout)
        self.graph = None
        self.device = device

    def process_inputs(self, x, mask=None, high_interest_region=None, graph_structure=None):

        self.mask = mask
        
        # Create the graph structure if one is not already pre-defined, otherwise use the pre-defined structure
        # to transform the input images to their graph representations (and add node sizes as features)
        if graph_structure is None:

            # Use only the first input if we want to re-mesh the input, otherwise create the graph by super-imposing all inputs
            if self.remesh_input:
                x0 = add_positional_encoding(x[[0]])
                graph_structure = image_to_graph(
                    x0,
                    thresh=self.thresh,
                    mask=mask,
                    high_interest_region=high_interest_region,
                    transform_func=self.transform_func,
                    condition=self.condition,
                    use_edge_attrs=self.use_edge_attrs
                    )
            else:
                x = add_positional_encoding(x)
                graph_structure = image_to_graph(
                    x,
                    thresh=self.thresh,
                    mask=mask,
                    high_interest_region=high_interest_region,
                    transform_func=self.transform_func,
                    condition=self.condition,
                    use_edge_attrs=self.use_edge_attrs
                    )
        else:
            graph_structure['data'] = x

        # Create Graph object, add data to pyg Data object
        self.graph = Graph(graph_structure['edge_index'], graph_structure['edge_attrs'])
        self.graph.pyg.x = graph_structure['data']

        # Store mapping, n_pixels_per_node, image_shape in the Graph object
        self.graph.mapping = graph_structure['mapping']
        self.graph.n_pixels_per_node = graph_structure['n_pixels_per_node']
        self.graph.image_shape = self.image_shape
        self.graph.pyg.to(self.device)
        
        # Encoder ------------------------------------------------------------------------------------------------------------
        self.graph.hidden, self.graph.cell = None, None
        for t in range(self.input_timesteps):

            # Perform encoding step
            hidden, cell = self.encoder(
                X=self.graph.pyg.x if self.remesh_input else self.graph.pyg.x[[t]], 
                edge_index=self.graph.pyg.edge_index, 
                edge_weight=self.graph.pyg.edge_attr, 
                H=self.graph.hidden[-1] if self.graph.hidden is not None else None, 
                C=self.graph.cell[-1] if self.graph.cell is not None else None
                )
            
            # If the threshold is -np.inf (no quadtree decomposition), re-meshing is not relevant
            # since we use the same mesh every step.
            if t < self.input_timesteps:
                if self.thresh != -np.inf:
                    if self.remesh_input:
                        self.do_remesh_input(x[[t+1]], hidden, cell, mask, high_interest_region)
                    else:
                        self.graph.hidden = hidden
                        self.graph.cell = cell
                else:
                    self.graph.hidden = hidden
                    self.graph.cell = cell

        # Persistance (removed for now)
        self.graph.persistence = x[-1, :, [0]]
        
        # First input to the decoder is the last input to the encoder 
        self.graph.pyg.x = self.graph.pyg.x[-1, :, [0, -3, -2, -1]]  # Node size
        
        # Add SIP and reorder as [SIC, SIP, x, y, node_size]  # MULTITASK
        if self.multitask:
            self.graph.pyg.x = torch.cat([(self.graph.pyg.x[..., [0]] > 0.15).float(), self.graph.pyg.x], -1)
            self.graph.pyg.x = self.graph.pyg.x[:, [1, 0, 2, 3, 4]]


    def unroll_output(self, unroll_steps, y, concat_layers=None, teacher_forcing_ratio=0.5, mask=None, high_interest_region=None, remesh_every=1):

        # Lists to store decoder outputs
        outputs = []
        output_mappings = []

        for t in unroll_steps:

            if self.debug:
                if self.device.type == 'cuda':
                    print(
                        f'Decoder step {t} \n' + \
                        f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB\n" + \
                        f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB\n" + \
                        f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB",
                        # end='\033[A\033[A\033[A'
                    )
                else:
                    pid = os.getpid()
                    python_process = psutil.Process(pid)
                    memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB
                    print('CPU memory usage:', memoryUse, 'GB', end='\r')
            
            if concat_layers is not None:
                concat_layers_t = torch.cat([concat_layers[t], (torch.ones_like(concat_layers[t]) * t)/self.output_timesteps], dim=-1)
                self.graph.concat_layers = concat_layers_t
            else:
                concat_layers_t = None
                
            self.graph.pyg.to(self.device)

            # Perform decoding step -- uncomment to use multiple decoders for different forecast steps
            # if t < 5:
            #     decoder = self.decoder_1
            # elif t < 15:
            #     decoder = self.decoder_2
            # else:
            #     decoder = self.decoder_3
            decoder = self.decoder_1
                
            output, hidden, cell = decoder(
                X=self.graph.pyg.x,
                edge_index=self.graph.pyg.edge_index,
                edge_weight=self.graph.pyg.edge_attr, 
                concat_layers=self.graph.concat_layers if hasattr(self.graph, 'concat_layers') else None,
                y_initial=self.graph.persistence if not self.multitask else self.graph.pyg.x[:, :2],
                H=self.graph.hidden, 
                C=self.graph.cell
                )

            # This is the prediction we are outputting. 
            outputs.append(output)
            output_mappings.append(self.graph.mapping)

            # Decide whether to use the prediction or ground truth for the input to the next rollout step
            teacher_force = random.random() < teacher_forcing_ratio
            teacher_input = y[[t]] if teacher_force else None

            # Re-mesh only if we're using the quadtree decomposition, and only every remesh_every step.
            if (self.thresh != -np.inf) and ((t+1) % remesh_every == 0):
                self.do_remesh(output, hidden, cell, mask, high_interest_region, teacher_force=teacher_force, teacher_input=teacher_input)
            else:
                self.update_without_remesh(output, hidden, cell, teacher_force=teacher_force, teacher_input=teacher_input)

        return outputs, output_mappings
        

        
    def forward(self, x, y=None, concat_layers=None, teacher_forcing_ratio=0.5, mask=None, high_interest_region=None, graph_structure=None, remesh_every=1):
        # Encoder
        self.process_inputs(x, mask=mask, high_interest_region=high_interest_region, graph_structure=graph_structure)
        
        # Uncomment to use different unroll steps for different epochs
        # if self.training and self.epoch < 2:
        #     unroll_steps = range(5)
        # elif self.training and self.epoch < 4:
        #     unroll_steps = range(15)
        # elif self.training and self.epoch < 6:
        #     unroll_steps = range(45)
        # else:
        #     unroll_steps = range(self.output_timesteps)

        # Decoder
        outputs, output_mappings = self.unroll_output(
            range(self.output_timesteps),
            y,
            concat_layers=concat_layers,
            teacher_forcing_ratio=teacher_forcing_ratio,
            mask=mask,
            high_interest_region=high_interest_region,
            remesh_every=remesh_every
            )

        return outputs, output_mappings

    def update_without_remesh(self, data, hidden, cell, teacher_force=False, teacher_input=None):
        if teacher_force:
            pos_encoding = self.graph.pyg.x[..., 1:]
            self.graph.pyg.x = torch.cat([teacher_input.squeeze(0), pos_encoding], dim=-1)
        else:
            # Add positional encoding
            pos_encoding = self.graph.pyg.x[..., 1:]  if not self.multitask else self.graph.pyg.x[..., 2:] # MULTITASK
            self.graph.pyg.x = torch.cat([data, pos_encoding], dim=-1)

        self.graph.hidden = hidden
        self.graph.cell = cell


    def do_remesh(self, data, hidden, cell, mask=None, high_interest_region=None, teacher_force=False, teacher_input=None):
        image_shape = self.graph.image_shape

        # Output is a prediction on the original graph structure
        # First convert it back to its grid representation
        # We also convert the hidden and cell state in the same way 
        data_img = unflatten(data, self.graph.mapping, image_shape)
        hidden_img = unflatten(hidden, self.graph.mapping, image_shape)
        cell_img = unflatten(cell, self.graph.mapping, image_shape)

        del self.graph.mapping

        # Then we convert it back to a graph representation where the graph is determined by
        # its own data (rather than the one created by the input images / previous step)
        if teacher_force:
            teacher_input = add_positional_encoding(teacher_input)
            graph_structure = image_to_graph(
                teacher_input, 
                thresh=self.thresh, 
                mask=mask, 
                high_interest_region=high_interest_region, 
                transform_func=self.transform_func, 
                condition=self.condition, 
                use_edge_attrs=self.use_edge_attrs
                )
        else:
            data_img = add_positional_encoding(data_img.unsqueeze(0))  # Add pos. embedding
            graph_structure = image_to_graph(
                data_img,
                thresh=self.thresh, 
                mask=mask, 
                high_interest_region=high_interest_region, 
                transform_func=self.transform_func, 
                condition=self.condition, 
                use_edge_attrs=self.use_edge_attrs
                )

        concat_layers = graph_structure['data'][:, :, [0]]  # Removed for now

        # Use the graph structure to convert the hidden and cell states to their graph representations
        hidden_img, cell_img = torch.swapaxes(hidden_img, 0, -1), torch.swapaxes(cell_img, 0, -1)
        hidden = flatten(hidden_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        cell = flatten(cell_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        hidden, cell = torch.swapaxes(hidden, 0, -1), torch.swapaxes(cell, 0, -1)

        # Update the graph for the next rollout 
        self.graph.pyg.edge_index = graph_structure['edge_index']
        self.graph.pyg.edge_attr = graph_structure['edge_attrs']
        self.graph.pyg.x = graph_structure['data'].squeeze(0)
        self.graph.concat_layers = concat_layers  # Removed for now
        self.graph.mapping = graph_structure['mapping']
        self.graph.n_pixels_per_node = graph_structure['n_pixels_per_node']

        self.graph.hidden = hidden
        self.graph.cell = cell

        self.graph.image_shape = image_shape

    def do_remesh_input(self, data_img, hidden, cell, mask=None, high_interest_region=None):
        image_shape = self.graph.image_shape

        # Convert H and C to their image represenation using the old graph
        hidden_img = unflatten(hidden, self.graph.mapping, image_shape)
        cell_img = unflatten(cell, self.graph.mapping, image_shape)

        # Create graph using the current input
        data_img = add_positional_encoding(data_img)  # Add pos. embedding
        graph_structure = image_to_graph(
            data_img, 
            thresh=self.thresh, 
            mask=mask, 
            high_interest_region=high_interest_region,
            transform_func=self.transform_func, 
            condition=self.condition, 
            use_edge_attrs=self.use_edge_attrs
            )

        # Use the graph structure to convert the hidden and cell states to their graph representations
        hidden_img, cell_img = torch.swapaxes(hidden_img, 0, -1), torch.swapaxes(cell_img, 0, -1)
        hidden = flatten(hidden_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        cell = flatten(cell_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        hidden, cell = torch.swapaxes(hidden, 0, -1), torch.swapaxes(cell, 0, -1)

        # Create a graph object for input into next rollout
        self.graph = Graph(graph_structure['edge_index'], graph_structure['edge_attrs'])
        self.graph.pyg.x = graph_structure['data']
        self.graph.mapping = graph_structure['mapping']
        self.graph.n_pixels_per_node = graph_structure['n_pixels_per_node']

        self.graph.hidden = hidden
        self.graph.cell = cell

        self.graph.image_shape = image_shape
    
