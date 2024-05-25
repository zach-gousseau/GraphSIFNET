import numpy as np
import torch
import random
import time
import glob
import xarray as xr

import argparse

from model.utils import int_to_datetime

from model.mpnnlstm import NextFramePredictorS2S

from torch.utils.data import DataLoader

from ice_dataset import IceDataset

from model.graph_functions import (
    create_static_heterogeneous_graph, 
    create_static_homogeneous_graph,
    flatten,
    unflatten)


if __name__ == '__main__':

    np.random.seed(41)
    random.seed(41)
    torch.manual_seed(41)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', nargs='?', default=5, type=int)
    parser.add_argument('--n_epochs', nargs='?', default=30, type=int)
    parser.add_argument('--hidden_size', nargs='?', default=32, type=int)
    parser.add_argument('--n_conv', nargs='?', default=3, type=int)
    parser.add_argument('--input_timesteps', nargs='?', default=10, type=int)
    parser.add_argument('--output_timesteps', nargs='?', default=90, type=int)
    parser.add_argument('--mesh_size', nargs='?', default=4, type=int)
    parser.add_argument('--mesh_type', nargs='?', default='heterogeneous', type=str)
    parser.add_argument('--conv_type', nargs='?', default='TransformerConv', type=str)  # GCNConv, GINE, TransformerConv
    parser.add_argument('--directory', nargs='?', default='test', type=str)
    

    args = vars(parser.parse_args())
    month = int(args['month'])
    n_epochs = int(args['n_epochs'])
    hidden_size = int(args['hidden_size'])
    n_conv = int(args['n_conv'])
    input_timesteps = int(args['input_timesteps'])
    output_timesteps = int(args['output_timesteps'])
    mesh_size = int(args['mesh_size'])
    mesh_type = args['mesh_type']
    conv_type = args['conv_type']
    directory = args['directory']

    # Defaults
    lr = 0.001
    multires_training = False
    truncated_backprop = 0
    
    training_years = range(1993, 2014)
    
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf', 'usi', 'vsi', 'sithick', 'thetao', 'so']
    y_vars = ['siconc']
    input_features = len(x_vars)
    rnn_type = 'NoConvLSTM'  # LSTM, GRU, 
    
    cache_dir = '/home/zgoussea/scratch/data_cache/'

    binary=False
        
    use_edge_attrs = False if conv_type == 'GCNConv' else True
        
    # -------------------------------------------

    # Full resolution dataset
    ds = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS
    mask = np.isnan(ds.siconc.isel(time=0)).values

    image_shape = mask.shape

    if mesh_type == 'heterogeneous':
        graph_structure = create_static_heterogeneous_graph(image_shape, mesh_size, mask, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
    elif mesh_type == 'homogeneous':
        graph_structure = create_static_homogeneous_graph(image_shape, mesh_size, mask, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)

    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    data_test = IceDataset(ds, range(training_years[-1]+1, training_years[-1]+3), month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    data_val = IceDataset(ds, range(training_years[-1]+1+3, training_years[-1]+1+3+3), month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=None, flatten_y=False)

    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)
    climatology = torch.moveaxis(climatology, 0, -1)
    climatology = flatten(climatology, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
    climatology = torch.moveaxis(climatology, -1, 0)

    # Arguments passed to Seq2Seq constructor
    model_kwargs = dict(
        hidden_size=hidden_size,
        dropout=0.1,
        n_layers=1,
        dummy=False,
        n_conv=n_conv,
        rnn_type=rnn_type,
        conv_type=conv_type,
    )

    experiment_name = f'M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'

    model = NextFramePredictorS2S(
        experiment_name=experiment_name,
        directory=directory,
        input_features=input_features,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        device=device,
        binary=binary,
        debug=False, 
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())

    model.model.train()

    # Train with full resolution. Use high interest region.
    model.train(
        loader_train,
        loader_test,
        climatology,
        lr=lr,
        n_epochs=n_epochs,
        mask=mask,
        truncated_backprop=truncated_backprop,
        graph_structure=graph_structure,
        )

    model.loss.to_csv(f'{directory}/loss_{experiment_name}.csv')
    model.load(directory)
    
    # Generate predictions
    model.model.eval()
    val_preds = model.predict(
        loader_val,
        climatology,
        mask=mask,
        graph_structure=graph_structure
        )
    
    # Save results
    launch_dates = [int_to_datetime(t) for t in loader_val.dataset.launch_dates]
    
    y_true = loader_val.dataset.y

    ds = xr.Dataset(
        data_vars=dict(
            y_hat_sic=(["launch_date", "timestep", "latitude", "longitude"], val_preds[..., 0].astype('float')),
            y_hat_sip=(["launch_date", "timestep", "latitude", "longitude"], val_preds[..., 1].astype('float')),
            y_true=(["launch_date", "timestep", "latitude", "longitude"], y_true.squeeze(-1).astype('float')),
        ),
        coords=dict(
            longitude=ds.longitude,
            latitude=ds.latitude,
            launch_date=launch_dates,
            timestep=np.arange(1, output_timesteps+1),
        ),
    )
    ds.to_netcdf(f'{directory}/valpredictions_{experiment_name}.nc')
    print(f'Finished model {month} in {((time.time() - start) / 60)} minutes')
