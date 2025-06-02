import netCDF4
import numpy as np
import torch
import random
import time
import glob
import xarray as xr
import argparse
import logging

from model.utils import int_to_datetime
from model.mpnnlstm import NextFramePredictorS2S
from torch.utils.data import DataLoader
from ice_dataset import IceDataset
from model.graph_functions import (
    create_static_heterogeneous_graph, 
    create_static_homogeneous_graph,
    flatten,
    unflatten
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def create_datasets(ds, train_years, month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure, mask, cache_dir):
    val_years = [train_years[-1] + 1]
    test_years = [train_years[-1] + 2]
    
    data_train = IceDataset(ds, train_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    data_val = IceDataset(ds, val_years, month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    data_test = IceDataset(ds, test_years, month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir, flatten_y=False)

    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=False)

    return loader_train, loader_val, loader_test

if __name__ == '__main__':

    logger.info("Starting script")

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', nargs='?', default=5, type=int)
    parser.add_argument('--n_epochs_init', nargs='?', default=30, type=int)
    parser.add_argument('--n_epochs_retrain', nargs='?', default=10, type=int)
    parser.add_argument('--hidden_size', nargs='?', default=32, type=int)
    parser.add_argument('--n_conv', nargs='?', default=3, type=int)
    parser.add_argument('--input_timesteps', nargs='?', default=10, type=int)
    parser.add_argument('--output_timesteps', nargs='?', default=90, type=int)
    parser.add_argument('--mesh_size', nargs='?', default=4, type=int)
    parser.add_argument('--mesh_type', nargs='?', default='heterogeneous', type=str)
    parser.add_argument('--conv_type', nargs='?', default='TransformerConv', type=str)
    parser.add_argument('--directory', nargs='?', default='test', type=str)
    
    args = vars(parser.parse_args())
    logger.info(f'Arguments: {args}')
    
    month = int(args['month'])
    n_epochs_init = int(args['n_epochs_init'])
    n_epochs_retrain = int(args['n_epochs_retrain'])
    hidden_size = int(args['hidden_size'])
    n_conv = int(args['n_conv'])
    input_timesteps = int(args['input_timesteps'])
    output_timesteps = int(args['output_timesteps'])
    mesh_size = int(args['mesh_size'])
    mesh_type = args['mesh_type']
    conv_type = args['conv_type']
    directory = args['directory']

    # Defaults
    lr = 0.0001
    multires_training = False
    truncated_backprop = 0
    
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf', 'usi', 'vsi', 'sithick', 'thetao', 'so']
    y_vars = ['siconc']
    input_features = len(x_vars)
    rnn_type = 'NoConvLSTM'  # LSTM, GRU, 
    
    cache_dir = '/path/to/data_cache/'
    
    test = False
    
    if test:
        cache_dir = None

    binary=False
        
    use_edge_attrs = False if conv_type == 'GCNConv' else True
        
    logger.info("Loading dataset")
    ds = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS

    if test:
        ds = ds.isel(latitude=slice(100, 125), longitude=slice(200, 225))
    
    mask = np.isnan(ds.siconc.isel(time=0)).values

    image_shape = mask.shape

    logger.info("Creating graph structure")
    if mesh_type == 'heterogeneous':
        graph_structure = create_static_heterogeneous_graph(image_shape, mesh_size, mask, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
    elif mesh_type == 'homogeneous':
        graph_structure = create_static_homogeneous_graph(image_shape, mesh_size, mask, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)

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
    
    experiment_name = 'experiment'
    logger.info(f'Experiment name: {experiment_name}')
    
    logger.info("Initializing model")
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

    logger.info(f'Num. parameters: {model.get_n_params()}')

    train_years = range(1993, 2003)
    test_year = train_years[-1] + 2
    n_epochs = n_epochs_init
    while test_year < 2021:
        
        logger.info(f'Creating datasets for {train_years}')
        loader_train, loader_val, loader_test = create_datasets(ds, train_years, month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure, mask, cache_dir)

        model.model.train()

        logger.info("Starting training")
        model.train(
            loader_train,
            loader_val,
            climatology,
            lr=lr,
            n_epochs=n_epochs,
            mask=mask,
            truncated_backprop=truncated_backprop,
            graph_structure=graph_structure,
        )

        logger.info("Saving training loss")
        model.loss.to_csv(f'{directory}/loss_{experiment_name}_{test_year}.csv')
        model.load(directory)
        
        # Generate predictions
        logger.info("Generating predictions")
        model.model.eval()
        test_preds = model.predict(
            loader_test,
            climatology,
            mask=mask,
            graph_structure=graph_structure
        )
        
        # Save results
        logger.info("Saving results")
        launch_dates = [int_to_datetime(t) for t in loader_test.dataset.launch_dates]
        
        y_true = loader_test.dataset.y

        ds_result = xr.Dataset(
            data_vars=dict(
                y_hat_sic=(["launch_date", "timestep", "latitude", "longitude"], test_preds[..., 0].astype('float')),
                y_hat_sip=(["launch_date", "timestep", "latitude", "longitude"], test_preds[..., 1].astype('float')),
                y_true=(["launch_date", "timestep", "latitude", "longitude"], y_true.squeeze(-1).astype('float')),
            ),
            coords=dict(
                longitude=ds.longitude,
                latitude=ds.latitude,
                launch_date=launch_dates,
                timestep=np.arange(1, output_timesteps+1),
            ),
        )
        result_path = f'{directory}/testpredictions_{experiment_name}_{test_year}.nc'
        ds_result.to_netcdf(result_path)
        
        train_years = [train_years[-1] + 1]
        test_year = train_years[-1] + 2
        n_epochs = n_epochs_retrain

    logger.info(f'Finished all experiments in {((time.time() - start) / 60)} minutes')

"""
module load StdEnv/2023
module load gcc/12.3
module load eccodes/2.31.0
module load openmpi/4.1.5
module load hdf5/1.14.2
module load netcdf/4.9.2
mpirun -np 1 python experiment.py --month 5 --n_epochs 5 --hidden_size 8 --n_conv 1 --input_timesteps 5 \
--output_timesteps 5 --mesh_size 1 --mesh_type heterogeneous --conv_type GCNConv --directory scrap
"""
