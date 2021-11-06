import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import torch
import h5py

import math
import argparse
import h5py
import numpy as np
from scipy import interpolate

from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
import os
from argparse import Namespace

from probabilistic_forecast.nn_mc import NN_MC
from probabilistic_forecast.ensemble import Deep_Ensemble
from probabilistic_forecast.swag import SWAG
from probabilistic_forecast.lstm_mc import LSTM_MC
from probabilistic_forecast.bnn import BNN
from probabilistic_forecast.gnn_mc import GNN_MC
from probabilistic_forecast.nn_standard import NN_Standard

from probabilistic_forecast.utils.data_utils import data_loader
from probabilistic_forecast.utils.torch_utils import torch_loader


def evaluate_single_model(args):


    models_types= {'NN_MC': {'model_class':NN_MC, 'sequential': False }, 
            'Deep_Ensemble':{'model_class':Deep_Ensemble, 'sequential': False }, 
            'SWAG':{'model_class':SWAG, 'sequential': False }, 
            'LSTM_MC': {'model_class':LSTM_MC, 'sequential': True }, 
            'BNN': {'model_class':BNN, 'sequential': False}, 
            'NN_Standard': {'model_class':NN_Standard, 'sequential': False }, 
            'GNN_MC':{'model_class':GNN_MC, 'sequential': True }
        }
    model_class = models_types[args.model]['model_class']
    sequential = models_types[args.model]['sequential']
    X_train, y_train, X_test, y_test, stats =data_loader(args.data_dir,
                                                    task=args.task, 
                                                    historical_sequence_length=args.historical_sequence_length, 
                                                    forecast_horizon=args.forecast_horizon, 
                                                    start_train=args.start_train,
                                                    end_train=args.end_train,
                                                    start_test=args.start_test,
                                                    end_test=args.end_test)
    train_loader, test_loader = torch_loader(X_train, y_train, X_test, y_test, args.historical_sequence_length, args.batch_size, sequential)
    input_dim = X_train.shape[-1] if sequential else X_train.shape[-1] * args.historical_sequence_length
    output_dim = y_train.shape[-1]

    model = model_class(input_dim, output_dim, args)

    if args.model == 'SWAG':
        # BatchNorm buffers update using train dataset.
        results = model.evaluate(test_loader, args.n_samples,  args.pretrained_dir,  train_loader,  args.adversarial_training)
    else:
        results = model.evaluate(test_loader, args.n_samples, args.pretrained_dir, args.adversarial_training)

    return results, stats


def evaluate_all_models(task):
    args_dict = {'adversarial_training': False, 'batch_size':128, 'data_dir':'../dataset', 'end_test':'2020-02-01',
            'end_train':'2019-12-31', 'forecast_horizon':24, 'historical_sequence_length':24, 'mode':'evaluate',
            'n_epochs':2000, 'n_samples':1000, 'plots_dir':'../plots', 'pretrained_dir':'../pretrained', 
            'start_test':'2020-01-01', 'start_train':'2019-01-01', 'task':task}

    resutls_all_models = {}
    # models = ['BNN', 'NN_MC', 'Deep_Ensemble', 'LSTM_MC', 'GNN_MC', 'SWAG' ]
#     models = ['NN_Standard', 'Deep_Ensemble' ,'SWAG', 'NN_MC', 'BNN']
    models = ['NN_Standard', 'BNN']

    for model in models:
        args_dict['model'] = model
        args = Namespace(**args_dict)
        results, stats = evaluate_single_model(args)
        results_model={}
        if task == 'regression':
            if model == 'NN_Standard':
                results_model['target_test'], results_model['mean'], results_model['stats'] = results[0], results[1], stats
            else:
                results_model['target_test'], results_model['samples_mean'], results_model['samples_var'], results_model['stats'] = results[0], results[1], results[2], stats
        elif task == 'classification':
            if model == 'NN_Standard':
                results_model['target_test'], results_model['prob'], results_model['stats'] = results[0], results[1], stats
            else:
                results_model['target_test'], results_model['samples'], results_model['stats'] = results[0], results[1], stats
        resutls_all_models[model] = results_model
    return resutls_all_models


def plot_epistemic_vs_aleatoric():

    results_class= evaluate_all_models(task='classification')
    value = results_class['BNN']
    target_test, samples, stats = value['target_test'], value['samples'], value['stats']

    value_standard = results_class['NN_Standard']
    prob_standard = value_standard['prob']

    y_true = target_test.astype(int)

    samples_size= samples.shape[0]
    rows = len(stats['thre_cols'])
    fig, axs = plt.subplots(1, 1, figsize=(8.5, 6))
    fig2, axs2 = plt.subplots(figsize=(12, 10),subplot_kw={"projection": "3d"})

    for i in range(0, rows):
        if '_pm25' in stats['thre_cols'][i]:
            particulate = "PM2.5"
        else:
            particulate = "PM10"
                  
        if (stats['thre_cols'][i].split('_')[0] == 'Elgeseter') and (particulate == "PM2.5"):

            tau1_list = np.arange(0, 0.8, 0.05)
            tau2_list = np.arange(0, 0.8, 0.05)

            f1_2d = 10*np.ones(shape=(len(tau1_list),len(tau2_list)))
           
            f1_standard_list, precision_standard_list, recall_standard_list = [], [], []
            
            for idx1, tau1 in enumerate(tau1_list):

                y_pred_standard = np.greater(prob_standard, tau1).astype(int)
                f1_standard_list.append(f1_score(y_true=y_true[:, i], y_pred=y_pred_standard[:, i]))
                precision_standard_list.append(precision_score(y_true=y_true[:, i], y_pred=y_pred_standard[:, i]))
                recall_standard_list.append(recall_score(y_true=y_true[:, i], y_pred=y_pred_standard[:, i]))
                
                samples_pred = np.greater(samples, tau1).astype(int).sum(axis=0) / samples_size

                samples_pred_i = samples_pred[:, i]
                for idx2, tau2 in enumerate(tau2_list):
                    y_pred_with_confidence = np.zeros(shape=samples_pred_i.shape)
                    y_pred_with_confidence[np.argwhere(samples_pred_i > tau2)]=1
                    
                    f1_2d[idx1, idx2]=f1_score(y_true=y_true[:, i], y_pred=y_pred_with_confidence)
                    
            tau2_list = (tau2_list - tau2_list.min())/(tau2_list.max() - tau2_list.min())

            tau1_list = (tau1_list - tau1_list.min())/(tau1_list.max() - tau1_list.min())

            axs.plot(tau1_list , f1_standard_list, color='b', label='F1 score',)
            axs.plot(tau1_list, precision_standard_list, color='r', label='Precision', linestyle='--', linewidth=1.0)
            axs.plot(tau1_list, recall_standard_list, color='g', label='Recall', linestyle='--', linewidth=1.0)
            axs.set_ylabel('Decision Score')
            axs.set_xlabel(r'Class probability threshold $\tau_1$'+'\n (Normalized Aleatoric Confidence)' )
          
            axs.text(tau1_list[np.argmax(f1_standard_list)], 0.1,
                        '$\longleftarrow$', ha="right", va="bottom", fontsize=20, color='r')
            axs.text(tau1_list[np.argmax(f1_standard_list)]-0.07, 0.1,
                        'More False Postive: \n Higher costs of \n unnecessary measures',
                        ha="right", va="bottom", fontsize=14, color='k')
            
            axs.text(tau1_list[np.argmax(f1_standard_list)]+0.13, 0.1,
                        '$\longrightarrow$', ha="right", va="bottom", fontsize=20, color='g')
            axs.text(tau1_list[np.argmax(f1_standard_list)]+0.15, 0.1,
                        'More False Negative: \n Higher risk of violating\n pollution regulations',
                        ha="left", va="bottom", fontsize=14, color='k')
            axs.legend(loc='center left', bbox_to_anchor=(0.75, 0.7))
            
            X, Y = np.meshgrid(tau1_list, tau2_list)
            Z = f1_2d.reshape(X.shape)
            
            surf = axs2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

            fig2.colorbar(surf, shrink=0.5, aspect=15)
            
            axs2.set_xlabel(r'Class probability threshold $\tau_1$' + '\n (Normalized Aleatoric confidence)', labelpad=10)
            axs2.set_ylabel(r'Confidence threshold $\tau2$'+ '\n (Normalized Epistemic confidence)',labelpad=10)
            axs2.set_zlabel('Decision Score (F1)')

        
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig('./plots/decision_making_standard.pdf', bbox_inches='tight')
    fig2.savefig('./plots/decision_making_probabilistic.pdf', bbox_inches='tight')

    #save to .hf, which can be used for ParaView rendering
    f = h5py.File('.tests/data.h5', 'w' )
    f['xcoordinates'] = tau1_list
    f['ycoordinates'] = tau2_list
    f['f1'] = f1_2d
    f.close()

    #Convert the surface .h5 file to a .vtp file.
    h5_to_vtp(surf_file='./tests/data.hf', surf_name='f1', zmax=1)

# This function adopted from:https://github.com/tomgoldstein/loss-landscape

def h5_to_vtp(surf_file, surf_name='f1', log=False, zmax=-1, interp=-1):
    #set this to True to generate points
    show_points = False
    #set this to True to generate polygons
    show_polys = True

    f = h5py.File(surf_file,'r')

    [xcoordinates, ycoordinates] = np.meshgrid(f['xcoordinates'][:], f['ycoordinates'][:][:])
    vals = f[surf_name]

    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals[:].ravel()

    # Interpolate the resolution up to the desired amount
    if interp > 0:
        m = interpolate.interp2d(xcoordinates[0,:], ycoordinates[:,0], vals, kind='cubic')
        x_array = np.linspace(min(x_array), max(x_array), interp)
        y_array = np.linspace(min(y_array), max(y_array), interp)
        z_array = m(x_array, y_array).ravel()

        x_array, y_array = np.meshgrid(x_array, y_array)
        x_array = x_array.ravel()
        y_array = y_array.ravel()

    vtp_file = surf_file + "_" + surf_name
    if zmax > 0:
        z_array[z_array > zmax] = zmax
        vtp_file +=  "_zmax=" + str(zmax)

    if log:
        z_array = np.log(z_array + 0.1)
        vtp_file +=  "_log"
    vtp_file +=  ".vtp"
    print("Here's your output file:{}".format(vtp_file))

    number_points = len(z_array)
    print("number_points = {} points".format(number_points))

    matrix_size = int(math.sqrt(number_points))
    print("matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    print("poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    print("number_polys = {}".format(number_polys))

    min_value_array = [min(x_array), min(y_array), min(z_array)]
    max_value_array = [max(x_array), max(y_array), max(z_array)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    poly_count = 0
    for column_count in range(poly_size):
        stride_value = column_count * matrix_size
        for row_count in range(poly_size):
            temp_index = stride_value + row_count
            averaged_z_value = (z_array[temp_index] + z_array[temp_index + 1] +
                                z_array[temp_index + matrix_size]  +
                                z_array[temp_index + matrix_size + 1]) / 4.0
            averaged_z_value_array.append(averaged_z_value)
            poly_count += 1

    avg_min_value = min(averaged_z_value_array)
    avg_max_value = max(averaged_z_value_array)

    output_file = open(vtp_file, 'w')
    output_file.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    output_file.write('  <PolyData>\n')

    if (show_points and show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_points, number_polys))
    elif (show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_polys))
    else:
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(number_points, number_points))

    # <PointData>
    output_file.write('      <PointData>\n')
    output_file.write('        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value_array[2], max_value_array[2]))
    for vertexcount in range(number_points):
        if (vertexcount % 6) is 0:
            output_file.write('          ')
        output_file.write('{}'.format(z_array[vertexcount]))
        if (vertexcount % 6) is 5:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 6) is not 5:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </PointData>\n')

    # <CellData>
    output_file.write('      <CellData>\n')
    if (show_polys and not show_points):
        output_file.write('        <DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(avg_min_value, avg_max_value))
        for vertexcount in range(number_polys):
            if (vertexcount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) is not 5:
            output_file.write('\n')
        output_file.write('        </DataArray>\n')
    output_file.write('      </CellData>\n')

    # <Points>
    output_file.write('      <Points>\n')
    output_file.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value, max_value))
    for vertexcount in range(number_points):
        if (vertexcount % 2) is 0:
            output_file.write('          ')
        output_file.write('{} {} {}'.format(x_array[vertexcount], y_array[vertexcount], z_array[vertexcount]))
        if (vertexcount % 2) is 1:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 2) is not 1:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Points>\n')

    # <Verts>
    output_file.write('      <Verts>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_points - 1))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount))
            if (vertexcount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) is not 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_points))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount + 1))
            if (vertexcount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) is not 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Verts>\n')

    # <Lines>
    output_file.write('      <Lines>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Lines>\n')

    # <Strips>
    output_file.write('      <Strips>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Strips>\n')

    # <Polys>
    output_file.write('      <Polys>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    if (show_polys):
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) is 0:
                    output_file.write('          ')
                output_file.write('{} {} {} {}'.format(temp_index, (temp_index + 1), (temp_index + matrix_size + 1), (temp_index + matrix_size)))
                if (polycount % 2) is 1:
                    output_file.write('\n')
                else:
                    output_file.write(' ')
                polycount += 1
        if (polycount % 2) is 1:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    if (show_polys):
        for polycount in range(number_polys):
            if (polycount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format((polycount + 1) * 4))
            if (polycount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (polycount % 6) is not 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Polys>\n')

    output_file.write('    </Piece>\n')
    output_file.write('  </PolyData>\n')
    output_file.write('</VTKFile>\n')
    output_file.write('')
    output_file.close()

    print("Done with file:{}".format(vtp_file))


if __name__ == "__main__":
    plot_epistemic_vs_aleatoric()