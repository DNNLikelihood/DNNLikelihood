import codecs
import json
import math
import multiprocessing
import os
import pickle
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from timeit import default_timer as timer

import ipywidgets as widgets
import joblib
import keras
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import tensorflow as tf
from corner import corner, quantile
from IPython.display import Javascript, display
from jupyterthemes import jtplot
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import (EarlyStopping, History, ReduceLROnPlateau,
                             TerminateOnNaN, ModelCheckpoint, LambdaCallback)
from keras.layers import (AlphaDropout, BatchNormalization, Dense, Dropout,
                          Input)
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.utils import plot_model, multi_gpu_model
import keras2onnx
import onnx
from livelossplot import PlotLossesKeras
from pandas import DataFrame
from scipy import stats
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from sklearn import datasets
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import the toy_likelihood module is needed by the tmu and related functions used here
import toy_likelihood
from toy_likelihood import *

tf.random.set_random_seed(1)
#np.random.seed(1)
pd.set_option('max_colwidth', 150)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
th_props = [
  ('font-size', '10px'),
  ('font-family', 'times')
  ]

# Set CSS properties for td elements in dataframe
td_props = [
  ('font-size', '10px'),
  ('font-family', 'times')
  ]
# Set table styles
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]
# initialize seed
seed = 111511
K.set_floatx('float64')
#K.set_epsilon(1e-20)
#print('K.float = ',K.floatx())
#print('K.epsilon = ',K.epsilon())

availableCPUCoresNumber = multiprocessing.cpu_count()
print(str(availableCPUCoresNumber)+" CPU cores available")

kubehelix = sns.color_palette("cubehelix", 30)
reds = sns.color_palette("Reds", 30)
greens = sns.color_palette("Greens", 30)
blues = sns.color_palette("Blues", 30)

class BlockPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def flatten_list(l):
    l = [item for sublist in l for item in sublist]
    return l

def closest_power2(x):
    op = math.floor if bin(int(x))[3] != "1" else math.ceil
    return 2**(op(math.log(x,2)))

def generate_gaussian_data(Ndim,Nevt,test_size,weighted = False,clip=1e-2):
    CovM = datasets.make_spd_matrix(n_dim=Ndim, random_state = seed)
    Mu = np.zeros(Ndim)
    Ymax = multivariate_normal.pdf(Mu, mean=Mu, cov=CovM)
    X = np.random.multivariate_normal(Mu, CovM, size=Nevt)
    Y = multivariate_normal.pdf(X, mean=Mu, cov=CovM)
    Y = Y/Ymax
    W = np.ones(Nevt)
    if weighted == True:
        xbin = np.sort(np.array([1/3**(i) for i in range(60)]))
        histo, edges = np.histogram(Y[:], bins=xbin, range=(0.,1.), density=True)
        inds = np.digitize(Y,edges[:len(xbin)-2])
        W = np.ones(len(Y))
        for i in range(len(Y)):
            W[i] = 1/np.clip(np.power(histo[inds[i]],1/2),clip,None)
    X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, test_size=test_size, random_state=seed)
    return [X_train, X_test, Y_train, Y_test, W_train, W_test]

def import_X_train(file,n):
    pickle_in = open(file,'rb')
    if type(n) == str:
        if n == 'all':
            allsamples_train = pickle.load(pickle_in)
            print('Imported', str(n), 'X_train samples')
        else:
            print('Invalid string')
            return []
    else:
        allsamples_train = pickle.load(pickle_in)[n]
        print('Imported', str(len(n)), 'X_train samples')
    pickle_in.close()
    return allsamples_train

def import_X_test(file,n):
    pickle_in = open(file,'rb')
    for i in range(4):
        _ = pickle.load(pickle_in)
    if type(n) == str:
        if n == 'all':
            allsamples_test = pickle.load(pickle_in)
            print('Imported', str(n), 'X_test samples')
        else:
            print('Invalid string')
            return []
    else:
        allsamples_test = pickle.load(pickle_in)[n]
        print('Imported', str(len(n)), 'X_test samples')
    pickle_in.close()
    return allsamples_test

def import_Y_train(file,n):
    pickle_in = open(file,'rb')
    _ = pickle.load(pickle_in)
    if type(n) == str:
        if n == 'all':
            logprob_values_train = pickle.load(pickle_in)
            print('Imported', str(n), 'Y_train samples')
        else:
            print('Invalid string')
            return []
    else:
        logprob_values_train = pickle.load(pickle_in)[n]
        print('Imported', str(len(n)), 'Y_train samples')
    pickle_in.close()
    return logprob_values_train

def import_Y_test(file,n):
    pickle_in = open(file,'rb')
    for i in range(5):
        _ = pickle.load(pickle_in)
    if type(n) == str:
        if n == 'all':
            logprob_values_test = pickle.load(pickle_in)
            print('Imported', str(n), 'Y_test samples')
        else:
            print('Invalid string')
            return []
    else:
        logprob_values_test = pickle.load(pickle_in)[n]
        print('Imported', str(len(n)), 'Y_test samples')
    pickle_in.close()
    return logprob_values_test

def import_XY_train(file,n):
    pickle_in = open(file,'rb')
    if type(n) == str:
        if n == 'all':
            allsamples_train = pickle.load(pickle_in)
            logprob_values_train = pickle.load(pickle_in)        
            print('Imported', str(n), '(X_train, Y_train) samples')
        else:
            print('Invalid string')
            return []
    else:
        allsamples_train = pickle.load(pickle_in)[n]
        logprob_values_train = pickle.load(pickle_in)[n]
        print('Imported', str(len(n)), '(X_train, Y_train) samples')
    pickle_in.close()
    return [allsamples_train, logprob_values_train]

def import_XY_test(file,n):
    pickle_in = open(file,'rb')
    for i in range(4):
        _ = pickle.load(pickle_in)
    if type(n) == str:
        if n == 'all':
            allsamples_test = pickle.load(pickle_in)
            logprob_values_test = pickle.load(pickle_in)
            print('Imported', str(n), '(X_test, Y_test) samples')
        else:
            print('Invalid string')
            return []  
    else:
        allsamples_test = pickle.load(pickle_in)[n]
        logprob_values_test = pickle.load(pickle_in)[n]
        print('Imported', str(len(n)), '(X_test, Y_test) samples')
    pickle_in.close()
    return [allsamples_test, logprob_values_test]

def compute_sample_weights(sample, bins, power=1):
    hist, edges = np.histogram(sample, bins=bins)
    hist = np.where(hist < 5, 5, hist)
    tmp = np.digitize(sample, edges, right=True)
    W = 1/np.power(hist[np.where(tmp == bins, bins-1, tmp)],power)
    W = W/np.sum(W)*len(sample)
    return W

def R2_metric(y_true, y_pred):
    from keras import backend as K
    MSE_model =  K.sum(K.square( y_true-y_pred )) 
    MSE_baseline = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - MSE_model/(MSE_baseline + K.epsilon()))

def Rt_metric(y_true, y_pred):
    from keras import backend as K
    MAPE_model =  K.sum(K.abs( 1-y_pred/(y_true + K.epsilon()))) 
    MAPE_baseline = K.sum(K.abs( 1-K.mean(y_true)/(y_true+ K.epsilon()) ) ) 
    return ( 1 - MAPE_model/(MAPE_baseline + K.epsilon()))
        
def model_define_old(Ndim,hid_layers,dropout_rate,act_func_out_layer,batch_norm,verbose=False):
    inputLayer = Input(shape=(Ndim,))
    x = Dense(hid_layers[0][0], activation=hid_layers[0][1])(inputLayer)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    if len(hid_layers)>1:
        for i in hid_layers[1:]:
            x = Dense(i[0], activation=i[1])(x)
            if batch_norm:
                x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
    outputLayer = Dense(1, activation=act_func_out_layer)(x)
    model = Model(inputs=inputLayer, outputs=outputLayer)
    if verbose:
        print(model.summary())
    return model

def model_define(Ndim,hid_layers,dropout_rate,act_func_out_layer,batch_norm,verbose=False):
    inputLayer = Input(shape=(Ndim,))
    if batch_norm:
        x = BatchNormalization()(inputLayer)
    if hid_layers[0][1] == 'selu':
        x = Dense(hid_layers[0][0], activation=hid_layers[0][1], kernel_initializer='lecun_normal')(inputLayer)
    else:
        x = Dense(hid_layers[0][0], activation=hid_layers[0][1], kernel_initializer='glorot_uniform')(inputLayer)
    if batch_norm:
        x = BatchNormalization()(x)
    if dropout_rate != 0:
        if hid_layers[0][1] == 'selu':
            x = AlphaDropout(dropout_rate)(x)
        else:
            x = Dropout(dropout_rate)(x)
    if len(hid_layers)>1:
        for i in hid_layers[1:]:
            if i[1] == 'selu':
                x = Dense(i[0], activation=i[1], kernel_initializer='lecun_normal')(x)
            else:
                x = Dense(i[0], activation=i[1], kernel_initializer='glorot_uniform')(x)
            if batch_norm:
                x = BatchNormalization()(x)
            if dropout_rate != 0:
                if i[1] == 'selu':
                    x = AlphaDropout(dropout_rate)(x)
                else:
                    x = Dropout(dropout_rate)(x)
    outputLayer = Dense(1, activation=act_func_out_layer)(x)
    model = Model(inputs=inputLayer, outputs=outputLayer)
    if verbose:
        print(model.summary())
    return model

def model_define_stacked(members):
	for i in range(len(members)):
		model = members[i][0]
		for layer in model.layers:
			layer.trainable = False
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	ensemble_visible = [model[0].input for model in members]
	ensemble_outputs = [model[0].output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(8, activation='selu')(merge)
	output = Dense(1, activation='linear')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	return model

def model_params(model):
    return int(model.count_params())

def model_trainable_params(model):
    return int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

def model_non_trainable_params(model):
    return int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

def model_compile(model,loss,optimizer,metrics,multi_gpu,verbose=False):
    availableGPUs = K.tensorflow_backend._get_available_gpus()
    availableCPUCoresNumber = multiprocessing.cpu_count()
    if len(availableGPUs)>1:
        if verbose:
            print(str(len(availableGPUs))+" GPUs available")
        if multi_gpu:
            if verbose:
                print("Compiling model on available GPUs")
            # Replicates `model` on 2 GPUs.
            # This assumes that your machine has 8 available GPUs.
            parallel_model = multi_gpu_model(model, gpus=len(availableGPUs))
            parallel_model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
            return parallel_model
        else:
            if verbose:
                print("MULTI_GPU flag False: Compiling model on single GPU")
            model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
            return model
    elif len(availableGPUs)==1:
        if verbose:
            print("1 GPU available")
            print("Compiling model on single GPU")
        model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
        return model
    else:
        if verbose:
            print("no GPU available, proceeding with CPUs")
            print("Compiling model for CPU")
            print(str(availableCPUCoresNumber)+" CPU cores available")
        model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
        return model
    
def model_train(model,X_train,Y_train,X_val,Y_val,scalerX,scalerY,epochs,batch_size,sample_weights=None,folder=None,title=None,monitored_metric='loss',plotlosses=False, model_checkpoint=False,early_stopping=False,restore_best_weights=False,reduceLR=True,reduce_LR_patience=5,min_delta=1e-4,verbose=False):
    jtplot.reset()
    try:
        plt.style.use('matplotlib.mplstyle')
    except:
        plt.style.use(r"%s" % ('/'.join(folder.rstrip('/').split('/')
                                    [:-1])+'/matplotlib.mplstyle'))
    folder = folder.rstrip('/')
    modname = title.replace(": ", "_")
    checkpoint_filename = r"%s" % (folder + "/" + modname + "_best_model.{epoch:02d}-{"+monitored_metric+":.2f}.hdf5")
    figname = modname + "_figure_training_livelossplot.pdf"
    if monitored_metric == 'loss':
        monitored_metric = 'val_loss'
    else:
        monitored_metric = "val_"+metric_name_unabbreviate(monitored_metric)
    X_train = scalerX.transform(X_train)
    X_val = scalerX.transform(X_val)
    Y_train = scalerY.transform(Y_train.reshape(-1, 1)).reshape(len(Y_train))
    Y_val = scalerY.transform(Y_val.reshape(-1, 1)).reshape(len(Y_val))
    if type(model.input_shape) == list:
        X_train = [X_train for _ in range(len(model.input_shape))]
        X_val = [X_val for _ in range(len(model.input_shape))]
    start = timer()
    callbacks = []
    if plotlosses:
        callbacks.append(PlotLossesKeras(fig_path=r"%s" %(folder + "/" + figname)))
    if early_stopping:
        callbacks.append(EarlyStopping(monitor=monitored_metric, mode="min", patience=1.2*reduce_LR_patience, min_delta=min_delta, restore_best_weights=restore_best_weights, verbose=verbose))
    if reduceLR:
        callbacks.append(ReduceLROnPlateau(monitor=monitored_metric, mode="min", factor=0.2, min_lr=0.00008,patience=reduce_LR_patience, min_delta=min_delta, verbose=verbose))
    if model_checkpoint:
        callbacks.append(ModelCheckpoint(checkpoint_filename,monitor=monitored_metric, mode="min", save_best_only=True, period=1))
    callbacks.append(TerminateOnNaN())
    if type(sample_weights) == np.ndarray:
        history = model.fit(X_train, Y_train, sample_weight=sample_weights, epochs=epochs, batch_size=batch_size, verbose = verbose,
                validation_data=(X_val, Y_val),callbacks = callbacks)
    else:
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose = verbose,
                validation_data=(X_val, Y_val),callbacks = callbacks)
    end = timer()
    training_time = end - start
    return [history, training_time]
    
def model_save_fig(folder,history,title,summary_text,metrics=['loss'], yscale='log',verbose=True):
    folder = folder.rstrip('/')
    modname = title.replace(": ", "_")
    metrics = np.unique(metrics)
    for metric in metrics:
        metric = metric_name_unabbreviate(metric)
        val_metric = 'val_'+ metric
        figname = modname + "_figure_training_"+ metric+".pdf"
        jtplot.reset()
        try:
            plt.style.use('matplotlib.mplstyle')
        except:
            plt.style.use(r"%s" % ('/'.join(folder.split('/')
                                        [:-1])+'/matplotlib.mplstyle'))
        if type(history) is dict:
            plt.plot(history[metric])
            plt.plot(history[val_metric])
        else:
            plt.plot(history.history[metric])
            plt.plot(history.history[val_metric])
        plt.yscale(yscale)
        plt.grid(linestyle="--", dashes=(5, 5))
        plt.title(r"%s" % title, fontsize=10)
        plt.xlabel(r"epoch")
        ylable = (metric.replace("_", "-"))
        plt.ylabel(r"%s" % ylable)
        plt.legend([r"training", r"validation"])
        plt.tight_layout()
        ax = plt.axes()
        x1, x2, y1, y2 = plt.axis()
        plt.text(0.965, 0.06, r"%s" % summary_text, fontsize=7, bbox=dict(facecolor="green", alpha=0.15,
                                                                          edgecolor='black', boxstyle='round,pad=0.5'), ha='right', ma='left', transform=ax.transAxes)
        plt.savefig(r"%s" % (folder + "/" + figname))
        if verbose:
            print(r"%s" % (folder + "/" + figname +
                           " created and saved."))
        plt.close()
    
def save_data_indices(file,idx_train,idx_val,idx_test):
    pickle_out = open(file, 'wb')
    pickle.dump(idx_train, pickle_out, protocol=4)
    pickle.dump(idx_val, pickle_out, protocol=4)
    pickle.dump(idx_test, pickle_out, protocol=4)
    pickle_out.close()

def load_data_indices(file):
    file = file.replace('model.h5', 'samples_indices.pickle')
    pickle_in = open(file, 'rb')
    idx_train = pickle.load(pickle_in)
    idx_val = pickle.load(pickle_in)
    idx_test = pickle.load(pickle_in)
    pickle_in.close()
    return [idx_train,idx_val,idx_test]
    
def saveHistory(path,history):
    new_hist = {}
    for key in list(history.keys()):
        if type(history[key]) == np.ndarray:
            new_hist[key] == history[key].tolist()
        elif type(history[key]) == list:
            if  type(history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history[key]))
            elif  type(history[key][0]) == np.float32:
                new_hist[key] = list(map(float, history[key]))
            else:
                new_hist[key] = history[key]
        else:
            new_hist[key] = history[key]
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4) 

def model_store(folder, idx_train, idx_val, idx_test, model, scalerX, scalerY, history, title, summary_log, verbose=True):
    folder = folder.rstrip('/')
    modname = title.replace(": ", "_") 
    model_json = model.to_json()
    #Save samples indices
    save_data_indices(r"%s"%(folder + "/" + modname + "_samples_indices.pickle"),idx_train,idx_val,idx_test)
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_samples_indices.pickle" +
              " created and saved."))
    #Save model as JSON
    with open(r"%s"%(folder + "/" + modname + "_model.json"), "w") as json_file:
        json_file.write(model_json)
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_model.json" +
              " created and saved."))
    #Save Keras model
    model.save(r"%s"%(folder + "/" + modname + "_model.h5"))
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_model.h5" +
              " created and saved."))
    #Save Onnx model
    onnx_model = keras2onnx.convert_keras(model, modname)
    onnx.save_model(onnx_model, r"%s"%(folder + "/" + modname + "_model.onnx"))
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_model.onnx" +
              " created and saved."))
    #Save history as json
    if type(history) is dict:
        saveHistory(r"%s"%(folder + "/" + modname + "_history.json"),
                    {**summary_log, **history})
    else:
        saveHistory(r"%s"%(folder + "/" + modname + "_history.json"),
                    {**summary_log, **history.history})
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_history.json" +
              " created and saved."))
    #Save scalers
    joblib.dump(scalerX, r"%s"%(folder + "/" + modname + "_scalerX.jlib"))
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_scalerX.jlib" +
              " created and saved."))
    joblib.dump(scalerY, r"%s"%(folder + "/" + modname + "_scalerY.jlib"))
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_scalerY.jlib" +
              " created and saved."))
    #Save model plot
    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file= r"%s"%(folder + "/" + modname + "_model_graph.pdf"))
    if verbose:
        print(r"%s"%(folder + "/" + modname + "_model_graph.pdf" +
              " created and saved."))

def model_predict(model,scalerX,scalerY,X,batch_size=1,steps=None,verbose=0):
    start = timer()
    X = scalerX.transform(X)
    X_len = len(X)
    if type(model.input_shape) == list:
        X = [X for _ in range(len(model.input_shape))]
    pred = scalerY.inverse_transform(model.predict(X, batch_size=batch_size, steps=steps, verbose = verbose)).reshape(X_len)
    end = timer()
    prediction_time = end - start
    return [pred, prediction_time]

def model_evaluate(model,scalerX,scalerY,X,Y,batch_size=1,steps=None,verbose=0):
    start = timer()
    X = scalerX.transform(X)
    Y = scalerY.transform(Y.reshape(-1, 1)).reshape(len(Y))
    if type(model.input_shape) == list:
        X = [X for _ in range(len(model.input_shape))]
    pred =  model.evaluate(X, Y, batch_size=batch_size, verbose=verbose)
    end = timer()
    prediction_time = end - start
    return pred

def highlight_cols(s, coldict):
    if s.name in coldict.keys():
        return ['background-color: {}'.format(coldict[s.name])] * len(s)
    return [''] * len(s)

def sortby(df, column_min=None, column_max=None, color_min='Green', color_max='Red', 
           highlights_min=['loss_best', 'mse_best', 'mae_best', 'mape_best', 'me_best', 'mpe_best',
                           'val_loss_best', 'val_mse_best', 'val_mae_best', 'val_mape_best', 'val_me_best', 'val_mpe_best',
                           'test_loss_best', 'test_mse_best', 'test_mae_best', 'test_mape_best', 'test_me_best', 'test_mpe_best'],
           highlights_max=['KS test-pred_train median','KS test-pred_val median', 'KS val-pred_test median', '"KS train-test median']):
    df = df.reset_index(drop=True)
    if column_min != None and column_max != None:
        print("Cannot sort by max and min on two different columns simultaneously. Please specify either column_min or column_max")
        return None
    if column_min != None:
        df = df.sort_values(by=[column_min])
        df = df.style.apply(highlight_cols, coldict={column_min: color_min})
    if column_max != None:
        df = df.sort_values(by=[column_max],ascending=False)
        df = df.style.apply(highlight_cols, coldict={column_max: color_max})
    df_styled = df.set_table_styles(
        styles).highlight_min(subset=highlights_min, color=color_min).highlight_max(subset=highlights_max, color=color_max)
    return df_styled

def show_figures(df,n,figures=['figure']):
    for fig in figures:
        try:
            file = df.loc[n,'Name'].replace('_model.json','_'+fig+'.pdf').replace('/','\\')
            os.startfile(r'%s'%file)
            print('File', file, 'opened.')
        except:
            print('File',file,'not found.')
    
def load_model_scaler(model_file_name,verbose=False):
    fileX = model_file_name.replace('model.h5','scalerX.jlib')
    scalerX = joblib.load(fileX)
    if verbose:
        print('Scaler',fileX,'loaded.')
    fileY = model_file_name.replace('model.h5','scalerY.jlib')
    scalerY = joblib.load(fileY)
    if verbose:
        print('Scaler', fileY, 'loaded.')
    return [scalerX,scalerY]

def import_results(folders):
    mylist = []
    i=0
    for thisdir in folders:
        for r, _, f in os.walk(thisdir):
            for file in f:
                if "history.json" in file:
                    current_file = os.path.join(r, file)
                    with open(current_file) as json_file: 
                        data = json.load(json_file)
                    mylist.append({**{'Scan': thisdir}, **{'Number': i}, **{'Name': current_file},**data})
                    i = i + 1
    for dic in mylist:
        if 'loss' in dic.keys():
            dic[dic['Loss']]=dic['loss']
        if 'val_loss' in dic.keys():
            dic['val_'+dic['Loss']]=dic['val_loss']
        if 'mean_absolute_percentage_error' in dic.keys():
            dic['mape']=dic.pop('mean_absolute_percentage_error')
        if 'val_mean_absolute_percentage_error' in dic.keys():
            dic['val_mape']=dic.pop('val_mean_absolute_percentage_error')
        if 'mean_squared_error' in dic.keys():
            dic['mse']=dic.pop('mean_squared_error')
        if 'val_mean_squared_error' in dic.keys():
            dic['val_mse']=dic.pop('val_mean_squared_error')
        if 'mean_squared_logarithmic_error' in dic.keys():
            dic['msle']=dic.pop('mean_squared_logarithmic_error')
        if 'val_mean_squared_logarithmic_error' in dic.keys():
            dic['val_msle']=dic.pop('val_mean_squared_logarithmic_error')
        if 'mean_absolute_error' in dic.keys():
            dic['mae']=dic.pop('mean_absolute_error')
        if 'val_mean_absolute_error' in dic.keys():
            dic['val_mae']=dic.pop('val_mean_absolute_error')
        if 'mean_error' in dic.keys():
            dic['me'] = dic.pop('mean_error')
        if 'val_mean_error' in dic.keys():
            dic['val_me'] = dic.pop('val_mean_error')
        if 'mean_percentage_error' in dic.keys():
            dic['mpe'] = dic.pop('mean_percentage_error')
        if 'val_mean_percentage_error' in dic.keys():
            dic['val_mpe'] = dic.pop('val_mean_percentage_error')
    print(str(i)+' files imprediction_timeported')
    mydataframe = pd.DataFrame.from_dict(mylist)
    return mydataframe

def get_model_files_list(folders):
    mylist = []
    i = 0
    for thisdir in folders:
        for r, _, f in os.walk(thisdir):
            for file in f:
                if "model.h5" in file:
                    current_file = os.path.join(r, file)
                    mylist.append(current_file.replace('\\', '/'))
                    i = i + 1
    print(str(i)+' files')
    return mylist

def import_model(folders):
    mylist = []
    i=0
    for thisdir in folders:
        for r, _, f in os.walk(thisdir):
            for file in f:
                if "model.json" in file:
                    current_file = os.path.join(r, file)
                    with open(current_file) as json_file: 
        #                print(json_file)
                        data = json.load(json_file)
                    mylist.append({**{'Scan': thisdir}, **{'Number': i}, **{'Name': current_file},**data})
                    i = i + 1
    print(str(i)+' files imported')
    mydataframe = pd.DataFrame.from_dict(mylist)
    return mydataframe

def import_models(folders, true_strings = [""], false_strings = ["-!-"], listonly=True):
    mylist = list()
    i=0
    for thisdir in folders:
        for r, d, f in os.walk(thisdir):
            for file in f:
                true_strings_bool = bool(np.prod([a in file for a in true_strings]))
                false_strings_bool = bool(np.prod([a not in file for a in false_strings]))
                if "model.h5" in file and true_strings_bool and false_strings_bool:
                    current_file = os.path.join(r, file)
                    if listonly:
                        print("Importing",current_file)
                        mylist.append(current_file)
                    else:
                        model = load_model(current_file, custom_objects={'R2_metric': R2_metric, 'Rt_metric': Rt_metric})
                        scalerX, scalerY = load_model_scaler(current_file)
                        mylist.append([model,scalerX,scalerY])
                    i = i + 1
    return mylist

## All these functions always take un-scaled X as inputs
def logprob_DNN_multi(samples,model,scalerX,scalerY,batch_size=1,threshold=-400):
    nnn = len(samples)
    logprob = model_predict(model,scalerX,scalerY,np.array(samples[0:nnn]),batch_size=batch_size)[0]
    if np.bool(np.prod(logprob<threshold)):
        logprob[logprob<threshold]=-np.inf
    if np.isnan(logprob).any():
        print("Warning: nan has been replaced with -np.inf.")
        logprob = np.nan_to_num(logprob)
        logprob[logprob==0]=-np.inf
        return logprob
    else:
        return logprob

def logprob_DNN(sample, model, scalerX, scalerY):
    return logprob_DNN_multi([sample], model, scalerX, scalerY, batch_size=1)[0]

def minus_logprob_DNN(sample, model, scalerX, scalerY):
    logprob = model_predict(model,scalerX,scalerY,np.array([sample]),batch_size=1)[0][0]
    return -logprob

def minus_logprob_delta_DNN(delta,mu,model, scalerX, scalerY):
    pars = np.concatenate((np.array([mu]),delta))
    logprob = model_predict(model,scalerX,scalerY,np.array([pars]),batch_size=1)[0][0]
    return -logprob

def tmu_DNN(mu,model, scalerX, scalerY):
    minimum_logprob_DNN = minimize(lambda x: minus_logprob_DNN(x,model,scalerX,scalerY), np.full(95,0),method='Powell')['x']
    L_muhat_deltahat_DNN = -minus_logprob_DNN(minimum_logprob_DNN,model,scalerX,scalerY)
    minimum_logprob_delta_DNN = np.concatenate((np.array([mu]),minimize(lambda x: minus_logprob_delta_DNN(x,mu,model,scalerX,scalerY), np.full(94,0),method='Powell')['x']))
    L_mu_deltahat_DNN = -minus_logprob_DNN(minimum_logprob_delta_DNN,model,scalerX,scalerY)
    return np.array([mu,L_muhat_deltahat_DNN,L_mu_deltahat_DNN,-2*(L_mu_deltahat_DNN-L_muhat_deltahat_DNN)])

def extend_corner_range(S1,S2,ilist,percent):
    res = []
    for i in ilist:
        minn = np.min([np.min(S1[:,i]),np.min(S2[:,i])])
        maxx = np.max([np.max(S1[:,i]),np.max(S2[:,i])])
        if minn<0:
            minn = minn*(1+percent/100)
        else:
            minn = minn*(1-percent/100)
        if maxx>0:
            maxx = maxx*(1+percent/100)
        else:
            maxx = maxx*(1-percent/100)
        res.append([minn,maxx])
    return res

def get_1d_hist(i_dim, xs, nbins=25, ranges=None, weights=None, intervals=None,normalize1d=False):
    """Assumes smooth1d = True
    """
    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"
    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")
    # Parse the parameter ranges.
    if ranges is None:
        if "extents" in hist2d_kwargs:
            logging.warn("Deprecated keyword argument 'extents'. "
                         "Use 'range' instead.")
            ranges = hist2d_kwargs.pop("extents")
        else:
            ranges = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in ranges], dtype=bool)
            if np.any(m):
                raise ValueError(("It looks like the parameter(s) in "
                                  "column(s) {0} have no dynamic range. "
                                  "Please provide a `range` argument.")
                                 .format(", ".join(map(
                                     "{0}".format, np.arange(len(m))[m]))))
    else:
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        ranges = list(ranges)
        for i, _ in enumerate(ranges):
            try:
                emin, emax = ranges[i]
            except TypeError:
                q = [0.5 - 0.5*ranges[i], 0.5 + 0.5*ranges[i]]
                ranges[i] = quantile(xs[i], q, weights=weights)
    if len(ranges) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")
    # Parse the bin specifications.
    try:
        bins = [int(nbins) for _ in ranges]
    except TypeError:
        if len(nbins) != len(ranges):
            raise ValueError("Dimension mismatch between bins and range")
    x = xs[i_dim]
    # Deal with masked arrays.
    if hasattr(x, "compressed"):
            x = x.compressed()
    # Get 1D curve.
    n, b = np.histogram(
        x, bins=bins[i_dim], weights=weights, range=np.sort(ranges[i_dim]))
    if normalize1d:
        n = n/n.sum()
    n = gaussian_filter(n, True)
    x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
    y0 = np.array(list(zip(n, n))).flatten()
    # Generate 1D curves in intervals.
    result = []
    if intervals is None:
        result.append([x0, y0])
    else:
        for interval in intervals:
            tmp = np.transpose(np.append(x0, y0).reshape([2, len(x0)]))
            tmp = tmp[(tmp[:, 0] >= interval[0])*(tmp[:, 0] <= interval[1])]
            result.append([tmp[:, 0], tmp[:, 1]])
    return result

def save_results(folder, model, scalerX, scalerY, title, summary_text, X_train, X_test, Y_train, Y_test, tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, pars=[0], labels='None', plot_coverage=False, plot_distr=True, plot_corners=True, plot_tmu=False, batch_size=1, verbose=True):
    jtplot.reset()
    folder = folder.rstrip('/')
    modname = title.replace(": ","_")
    try:
        plt.style.use('matplotlib.mplstyle')
    except:
        plt.style.use(r"%s" % ('/'.join(folder.rstrip('/').split('/')
                                        [:-1])+'/matplotlib.mplstyle'))
    
    if plot_coverage:
        for i in pars:
            figname = 'coverage_log_par_' + str(i)
            nnn = min(1000,len(X_train),len(X_test))
            x_coord = i
            idx_train = np.random.choice(np.arange(len(X_train)), nnn, replace=False)
            idx_test = np.random.choice(np.arange(len(X_test)), nnn, replace=False)
            Y_pred = model_predict(model,scalerX,scalerY,X_test[idx_test],batch_size=batch_size)[0]
            curve_train = np.array([X_train[idx_train,x_coord],Y_train[idx_train]]).transpose()
            curve_train = curve_train[curve_train[:,0].argsort()]
            curve_test = np.array([X_test[idx_test,x_coord],Y_test[idx_test]]).transpose()
            curve_test = curve_test[curve_test[:,0].argsort()]
            curve_pred = np.array([X_test[idx_test,x_coord],Y_pred]).transpose()
            curve_pred = curve_pred[curve_pred[:,0].argsort()]
            plt.plot(curve_train[:,0], curve_train[:,1], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=3, label=r"train")
            plt.plot(curve_test[:,0], curve_test[:,1], color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=3, label=r"test")
            plt.plot(curve_pred[:,0], curve_pred[:,1], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=3, label=r"pred")
            plt.xlabel(r"%s"%(labels[i]))
            plt.ylabel(r"logprob ($\log\mathcal{L}+\log\mathcal{P}$)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf"))
            if verbose:
                print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                      " created and saved."))
            plt.close()
            
            figname = 'coverage_par_' + str(i)
            curve_train = np.array([X_train[idx_train,x_coord],np.exp(Y_train[idx_train])]).transpose()
            curve_train = curve_train[curve_train[:,0].argsort()]
            curve_test = np.array([X_test[idx_test,x_coord],np.exp(Y_test[idx_test])]).transpose()
            curve_test = curve_test[curve_test[:,0].argsort()]
            curve_pred = np.array([X_test[idx_test,x_coord],np.exp(Y_pred)]).transpose()
            curve_pred = curve_pred[curve_pred[:,0].argsort()]
            plt.plot(curve_train[:,0], curve_train[:,1], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=3, label=r"train")
            plt.plot(curve_test[:,0], curve_test[:,1], color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=3, label=r"test")
            plt.plot(curve_pred[:,0], curve_pred[:,1], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=3, label=r"pred")
            plt.xlabel(r"%s" % (labels[i]))
            plt.ylabel(r"prob ($\mathcal{L}\cdot\mathcal{P}$)")
            plt.legend()
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf"))
            if verbose:
                print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                      " created and saved."))
            plt.close()
    
    if plot_distr:
        figname = 'logprob_distr'
        nnn = min(10000,len(X_train),len(X_test))
        idx_train = np.random.choice(np.arange(len(X_train)), nnn, replace=False)
        idx_test = np.random.choice(np.arange(len(X_test)), nnn, replace=False)
        Y_pred = model_predict(model,scalerX,scalerY,X_test[idx_test],batch_size=batch_size)[0]
        bins = np.histogram(Y_train[idx_train], 50)[1]
        counts, _ = np.histogram(Y_train[idx_train], bins)
        integral = 1
        plt.step(bins[:-1], counts/integral, where='post',color = 'green', label=r"train")
        counts, _ = np.histogram(Y_test[idx_test], bins)
        integral = 1
        plt.step(bins[:-1], counts/integral, where='post',color = 'blue', label=r"val")
        counts, _ = np.histogram(Y_pred, bins)
        integral = 1
        plt.step(bins[:-1], counts/integral, where='post',color = 'red', label=r"pred")
        plt.xlabel(r"logprob ($\log\mathcal{L}+\log\mathcal{P}$)")
        plt.ylabel(r"counts")
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname +".pdf"))
        if verbose:
            print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                  " created and saved."))
        plt.close()

        figname = 'prob_distr'
        bins = np.exp(np.histogram(Y_train[idx_train], 50)[1])
        counts, _ = np.histogram(np.exp(Y_train[idx_train]), bins)
        integral = 1
        plt.step(bins[:-1], counts/integral, where='post',color = 'green', label=r"train")
        counts, _ = np.histogram(np.exp(Y_test[idx_test]), bins)
        integral = 1
        plt.step(bins[:-1], counts/integral, where='post',color = 'blue', label=r"val")
        counts, _ = np.histogram(np.exp(Y_pred), bins)
        integral = 1
        plt.step(bins[:-1], counts/integral, where='post',color = 'red', label=r"pred")
        plt.xlabel(r"prob ($\mathcal{L}\cdot\mathcal{P}$)")
        plt.ylabel(r"number of samples")
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname +".pdf"))
        if verbose:
            print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                  " created and saved."))
        plt.close()
    
    if plot_corners:
        figname = 'corner_pars_train'
        nnn = len(X_train)
        ilist = np.array(pars)
        nndim = len(ilist)
        idx_train = np.random.choice(np.arange(len(X_train)), nnn, replace=False)
        Y_pred = logprob_DNN_multi(X_train[idx_train], model, scalerX, scalerY, batch_size=batch_size)
        if np.amax(Y_pred)>0:
            print('There are positive values of log-likelihood')
        weights_DNN = np.exp(Y_pred)/np.exp(Y_train[idx_train])
        samp_lik_train = X_train[idx_train][:, ilist]
        samp_DNN_weights = weights_DNN/np.sum(weights_DNN)*len(samp_lik_train)
        value1 = np.mean(samp_lik_train,0)
        value2 = np.average(samp_lik_train, 0, weights=samp_DNN_weights)
        fig, axes = plt.subplots(nndim, nndim, figsize=(3*nndim, 3*nndim))
        figure1 = corner(samp_lik_train, bins = 50, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color='green', plot_contours=True, smooth=True, smooth1d=True, range = np.full(nndim,0.9999),
                        hist_kwargs={'color': 'green', 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18})
        _ = corner(samp_lik_train, bins = 50, weights=samp_DNN_weights, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color='red', plot_contours=True, smooth=True, smooth1d=True, range = np.full(nndim,0.9999),
                        hist_kwargs={'color': 'red', 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18})
        axes = np.array(figure1.axes).reshape((nndim, nndim))
        for i in range(nndim):
                ax = axes[i, i]
                ax.axvline(value1[i], color="green",alpha=1)
                ax.axvline(value2[i], color="red",alpha=1)
                ax.grid(True, linestyle='--', linewidth=1)
                ax.tick_params(axis='both', which='major', labelsize=16)
        for yi in range(nndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(value1[xi], color="green",alpha=1)
                ax.axvline(value2[xi], color="red",alpha=1)
                ax.axhline(value1[yi], color="green",alpha=1)
                ax.axhline(value2[yi], color="red",alpha=1)
                ax.plot(value1[xi], value1[yi], color="green",alpha=1)
                ax.plot(value2[xi], value2[yi], color="red",alpha=1)
                ax.grid(True, linestyle='--', linewidth=1)
                ax.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname +".pdf"))
        if verbose:
            print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                  " created and saved."))
        plt.close()

        figname = 'corner_pars_test'
        nnn = len(X_test)
        ilist = np.array(pars)
        nndim = len(ilist)
        idx_test = np.random.choice(np.arange(len(X_test)), nnn, replace=False)
        Y_pred = logprob_DNN_multi(X_test[idx_test], model, scalerX, scalerY, batch_size=batch_size)
        if np.amax(Y_pred) > 0:
            print('There are positive values of log-likelihood')
        weights_DNN = np.exp(Y_pred)/np.exp(Y_test[idx_test])
        samp_lik_test = X_test[idx_test][:, ilist]
        samp_DNN_weights = weights_DNN/np.sum(weights_DNN)*len(samp_lik_test)
        value1 = np.mean(samp_lik_test, 0)
        value2 = np.average(samp_lik_test,0,weights=samp_DNN_weights)
        fig, axes = plt.subplots(nndim, nndim, figsize=(3*nndim, 3*nndim))
        figure1 = corner(samp_lik_test, bins = 50, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color='green', plot_contours=True, smooth=True, smooth1d=True, range = np.full(nndim,0.9999),
                        hist_kwargs={'color': 'green', 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18})
        _ = corner(samp_lik_test, bins=50, weights=samp_DNN_weights, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color='red', plot_contours=True, smooth=True, smooth1d=True, range=np.full(nndim, 0.9999),
                        hist_kwargs={'color': 'red', 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18})
        axes = np.array(figure1.axes).reshape((nndim, nndim))
        for i in range(nndim):
                ax = axes[i, i]
                ax.axvline(value1[i], color="green",alpha=1)
                ax.axvline(value2[i], color="red",alpha=1)
                ax.grid(True, linestyle='--', linewidth=1)
                ax.tick_params(axis='both', which='major', labelsize=16)
        for yi in range(nndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(value1[xi], color="green",alpha=1)
                ax.axvline(value2[xi], color="red",alpha=1)
                ax.axhline(value1[yi], color="green",alpha=1)
                ax.axhline(value2[yi], color="red",alpha=1)
                ax.plot(value1[xi], value1[yi], color="green",alpha=1)
                ax.plot(value2[xi], value2[yi], color="red",alpha=1)
                ax.grid(True, linestyle='--', linewidth=1)
                ax.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname +".pdf"))
        if verbose:
            print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                  " created and saved."))
        plt.close()

        figname = 'corner_pars_train_vs_test'
        value1 = np.mean(samp_lik_train,0)
        value2 = np.mean(samp_lik_test,0)
        fig, axes = plt.subplots(nndim, nndim, figsize=(3*nndim, 3*nndim))
        figure1 = corner(samp_lik_train, bins = 50, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color='green', plot_contours=True, smooth=True, smooth1d=True, range = np.full(nndim,0.9999),
                        hist_kwargs={'color': 'green', 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18})
        _ = corner(samp_lik_test, bins=50, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color='red', plot_contours=True, smooth=True, smooth1d=True, range=np.full(nndim, 0.9999),
                        hist_kwargs={'color': 'red', 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18})
        axes = np.array(figure1.axes).reshape((nndim, nndim))
        for i in range(nndim):
                ax = axes[i, i]
                ax.axvline(value1[i], color="green",alpha=1)
                ax.axvline(value2[i], color="red",alpha=1)
                ax.grid(True, linestyle='--', linewidth=1)
                ax.tick_params(axis='both', which='major', labelsize=16)
        for yi in range(nndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(value1[xi], color="green",alpha=1)
                ax.axvline(value2[xi], color="red",alpha=1)
                ax.axhline(value1[yi], color="green",alpha=1)
                ax.axhline(value2[yi], color="red",alpha=1)
                ax.plot(value1[xi], value1[yi], color="green",alpha=1)
                ax.plot(value2[xi], value2[yi], color="red",alpha=1)
                ax.grid(True, linestyle='--', linewidth=1)
                ax.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname +".pdf"))
        if verbose:
            print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                  " created and saved."))
        plt.close()

    if plot_tmu:
        figname = 'freq_tmu'
        plt.plot(tmuexact[:,0],tmuexact[:,-1])
        plt.plot(tmuDNN[:,0],tmuDNN[:,-1])
        plt.plot(tmusample001[:,1],tmusample001[:,-1])
        plt.plot(tmusample005[:,1],tmusample005[:,-1])
        plt.plot(tmusample01[:,1],tmusample01[:,-1])
        plt.plot(tmusample02[:,1],tmusample02[:,-1])
        plt.legend(['Exact','DNN','0.01','0.05','0.1','0.2'])
        x1,x2,_,_ = plt.axis()
        plt.axis([x1, x2, -0.5, 7.9])
        plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf"))
        if verbose:
            print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                  " created and saved."))
        plt.close()

def get_CI_from_sigma(sigma):
    np.array(sigma)
    return 2*stats.norm.cdf(sigma)-1

def get_sigma_from_CI(CI):
    np.array(CI)
    return stats.norm.ppf(CI/2+1/2)

def get_delta_chi2_from_CI(CI, dof = 1):
    np.array(CI)
    return stats.chi2.ppf(CI,dof)

def ks_w(data1, data2, wei1=False, wei2=False):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    n1 = len(data1)
    n2 = len(data2)
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side='right')]]
    d = np.max(np.abs(cdf1we - cdf2we))
    en = np.sqrt(n1 * n2 / (n1 + n2))
    prob = stats.distributions.kstwobign.sf(en * d)
    return [d, prob]

def sort_consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def HPD_intervals(data, intervals=0.68, weights=None, nbins=25, print_hist=False, reduce_binning=True):
    intervals = np.array([intervals]).flatten()
    if weights is None:
        weights = np.ones(len(data))
    weights = np.array(weights)
    counter = 0
    results = []
    for interval in intervals:
        hist = np.histogram(data, nbins, weights=weights, density=True)
        counts, bins = hist
        nbins_val = len(counts)
        if print_hist:
            integral = counts.sum()
            plt.step(bins[:-1], counts/integral, where='post',
                     color='green', label=r"train")
            plt.show()
        binwidth = bins[1]-bins[0]
        arr0 = np.transpose(np.concatenate(
            ([counts*binwidth], [(bins+binwidth/2)[0:-1]])))
        arr0 = np.transpose(np.append(np.arange(nbins_val),
                                      np.transpose(arr0)).reshape((3, nbins_val)))
        arr = np.flip(arr0[arr0[:, 1].argsort()], axis=0)
        q = 0
        bin_labels = np.array([])
        for i in range(nbins_val):
            if q <= interval:
                q = q + arr[i, 1]
                bin_labels = np.append(bin_labels, arr[i, 0])
            else:
                bin_labels = np.sort(bin_labels)
                result = [[arr0[tuple([int(k[0]), 2])], arr0[tuple([int(k[-1]), 2])]]
                          for k in sort_consecutive(bin_labels)]
                result_previous = result
                binwidth_previous = binwidth
                if reduce_binning:
                    while (len(result) == 1 and nbins_val+nbins < np.sqrt(len(data))):
                        nbins_val = nbins_val+nbins
                        result_previous = result
                        binwidth_previous = binwidth
                        nbins_val_previous = nbins_val
                        with BlockPrints():
                            HPD_int_val = HPD_intervals(data, intervals=interval, weights=weights, nbins=nbins_val, print_hist=False)
                        result = HPD_int_val[0][1]
                        binwidth = HPD_int_val[0][3]
                break
        results.append([interval, result_previous, nbins_val, binwidth_previous])
        counter = counter + 1
    return results

def HPD_quotas(data, intervals=0.68, weights=None, nbins=25, from_top=True):
    hist2D = np.histogram2d(data[:,0], data[:,1], bins=nbins, range=None, normed=None, weights=weights, density=None)
    intervals = np.array([intervals]).flatten()
    counts, binsX, binsY = hist2D
    integral = counts.sum()
    counts_sorted = np.flip(np.sort(flatten_list(counts)))
    quotas = intervals
    q = 0
    j = 0
    for i in range(len(counts_sorted)):
        if q < intervals[j] and i<len(counts_sorted)-1:
            q = q + counts_sorted[i]/integral
        elif q >= intervals[j] and i<len(counts_sorted)-1:
            if from_top:
                quotas[j] = 1-counts_sorted[i]/counts_sorted[0]
            else:
                quotas[j] = counts_sorted[i]/counts_sorted[0]
            j = j + 1
        else:
            for k in range(j,len(intervals)):
                quotas[k] = 0
            j = len(intervals)
        if j == len(intervals):
            return quotas

def weighted_quantiles(data, quantiles, weights=None,
                       data_sorted=False, onesided=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param data: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param weights: array-like of the same length as `array`
    :param data_sorted: bool, if True, then will avoid sorting of
        initial array
    :return: numpy.array with computed quantiles.
    """
    if onesided:
        data = np.array(data[data > 0])
    else:
        data = np.array(data)
    quantiles = np.array([quantiles]).flatten()
    if weights is None:
        weights = np.ones(len(data))
    weights = np.array(weights)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not data_sorted:
        sorter = np.argsort(data)
        data = data[sorter]
        weights = weights[sorter]

    w_quantiles = np.cumsum(weights) - 0.5 * weights
    w_quantiles -= w_quantiles[0]
    w_quantiles /= w_quantiles[-1]
    result = np.transpose(np.concatenate((quantiles, np.interp(
        quantiles, w_quantiles, data))).reshape(2, len(quantiles))).tolist()
    return result

def weighted_central_quantiles(data, intervals=0.68, weights=None, onesided=False):
    intervals = np.array([intervals]).flatten()
    if not onesided:
        return [[i, [weighted_quantiles(data, (1-i)/2, weights), weighted_quantiles(data, 0.5, weights), weighted_quantiles(data, 1-(1-i)/2, weights)]] for i in intervals]
    else:
        data = data[data > 0]
        return [[i, [weighted_quantiles(data, (1-i)/2, weights), weighted_quantiles(data, 0.5, weights), weighted_quantiles(data, 1-(1-i)/2, weights)]] for i in intervals]

def metric_name_unabbreviate(name):
    name_dict = {"acc": "accuracy", "me": "mean_error", "mpe": "mean_percentage_error", "mse": "mean_squared_error", "mae": "mean_absolute_error", "mape": "mean_absolute_percentage_error", "msle": "mean_squared_logarithmic_error"}
    for key in name_dict:
        name = name.replace(key, name_dict[key])
    return name

def metric_name_abbreviate(name):
    name_dict = {"accuracy": "acc", "mean_error": "me", "mean_percentage_error": "mpe", "mean_squared_error": "mse", "mean_absolute_error": "mae", "mean_absolute_percentage_error": "mape", "mean_squared_logarithmic_error": "msle"}
    for key in name_dict:
        name = name.replace(key, name_dict[key])
    return name

def compute_predictions(model, scalerX, scalerY, X_train, X_val, X_test, Y_train, Y_val, Y_test, LOSS, NEVENTS_TRAIN, BATCH_SIZE, FREQUENTISTS_RESULTS):
    print('Computing predictions')
    start_global = timer()
    start = timer()
    metrics_names = model.metrics_names
    #Choose NEVENTS_TRAIN random indices to pick data
    [idx_train, idx_val, idx_test] = [np.random.choice(np.arange(len(X)), min(
        int(NEVENTS_TRAIN), len(X)), replace=False) for X in [X_train, X_val, X_test]]
    #Redefine train/val/test data
    X_train = X_train[idx_train]
    X_val = X_val[idx_val]
    X_test = X_test[idx_test]
    Y_train = Y_train[idx_train]
    Y_val = Y_val[idx_val]
    Y_test = Y_test[idx_test]
    #Get logprobabilities and prediction time for selected subset of data
    Y_pred_train, _ = model_predict(model, scalerX, scalerY, X_train, batch_size=BATCH_SIZE)
    Y_pred_val, _ = model_predict(model, scalerX, scalerY, X_val, batch_size=BATCH_SIZE)
    Y_pred_test, prediction_time_test = model_predict(model, scalerX, scalerY, X_test, batch_size=BATCH_SIZE)
    prediction_time = prediction_time_test
    #Get probabilities exponentiating logprobabilities for both data and prediction
    [Y_train_exp, Y_val_exp, Y_test_exp] = [np.exp(Y_train), np.exp(Y_val), np.exp(Y_test)]
    [Y_pred_train_exp, Y_pred_val_exp, Y_pred_test_exp] = [np.exp(Y_pred_train), np.exp(Y_pred_val), np.exp(Y_pred_test)]
    #Evaluate the metrics on final best model
    metrics_names_train = [i+"_best" for i in model.metrics_names]
    metrics_names_val = ["val_"+i+"_best" for i in model.metrics_names]
    metrics_names_test = ["test_"+i+"_best" for i in model.metrics_names]
    metrics_train = model_evaluate(model, scalerX, scalerY, X_train, Y_train, batch_size=BATCH_SIZE)[0:len(metrics_names)]
    metrics_val = model_evaluate(model, scalerX, scalerY, X_val, Y_val, batch_size=BATCH_SIZE)[0:len(metrics_names)]
    metrics_test = model_evaluate(model, scalerX, scalerY, X_test, Y_test, batch_size=BATCH_SIZE)[0:len(metrics_names)]
    metrics_true = {**dict(zip(metrics_names_train,metrics_train)),**dict(zip(metrics_names_val,metrics_val)),**dict(zip(metrics_names_test,metrics_test))}
    print(metrics_true)
    #Evaluate min loss scaled
    metrics_names_train = [i+"_best_scaled" for i in model.metrics_names]
    metrics_names_val = ["val_"+i+"_best_scaled" for i in model.metrics_names]
    metrics_names_test = ["test_"+i+"_best_scaled" for i in model.metrics_names]
    metrics_train_scaled = [keras.losses.deserialize(l)(tf.convert_to_tensor(Y_train), tf.convert_to_tensor(Y_pred_train)).eval(session=tf.Session()) for l in [s.replace("loss", LOSS) for s in model.metrics_names]]
    metrics_val_scaled = [keras.losses.deserialize(l)(tf.convert_to_tensor(Y_val), tf.convert_to_tensor(Y_pred_val)).eval(session=tf.Session()) for l in [s.replace("loss", LOSS) for s in model.metrics_names]]
    metrics_test_scaled = [keras.losses.deserialize(l)(tf.convert_to_tensor(Y_test), tf.convert_to_tensor(Y_pred_test)).eval(session=tf.Session()) for l in [s.replace("loss", LOSS) for s in model.metrics_names]]
    metrics_scaled = {**dict(zip(metrics_names_train,metrics_train_scaled)),**dict(zip(metrics_names_val,metrics_val_scaled)),**dict(zip(metrics_names_test,metrics_test_scaled))}
    end = timer()
    print('Prediction on', NEVENTS_TRAIN,
          'points done form training, validation, and test data in', end-start, 's.')
    print('Estimating Bayesian inference')
    start = timer()
    #Computing and normalizing weights
    [Weights_train, Weights_val, Weights_test] = [Y_pred_train_exp / Y_train_exp, Y_pred_val_exp/Y_val_exp, Y_pred_test_exp/Y_test_exp]
    [Weights_train, Weights_val, Weights_test] = [Weights_train/np.sum(Weights_train)*len(Weights_train), Weights_val/np.sum(Weights_val)*len(Weights_val), Weights_test/np.sum(Weights_test)*len(Weights_test)]
    #Choosing probability intervals from gaussian sigmas
    quantiles = get_CI_from_sigma([get_sigma_from_CI(0.5), 1, 2, 3, 4, 5])
    [HPI_train, HPI_val, HPI_test] = [HPD_intervals(X_train[idx_train, 0], quantiles), HPD_intervals(
        X_val[idx_val, 0], quantiles), HPD_intervals(X_test[idx_test, 0], quantiles)]
    [HPI_pred_train, HPI_pred_val, HPI_pred_test] = [HPD_intervals(X_train[idx_train, 0], quantiles, Weights_train), HPD_intervals(
        X_val[idx_val, 0], quantiles, Weights_val), HPD_intervals(X_test[idx_test, 0], quantiles, Weights_test)]
    [one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test] = [((HPI_train[1][1][0][1]-HPI_train[1][1][0][0]) - (HPI_pred_train[1][1][0][1]-HPI_pred_train[1][1][0][0]))/(HPI_train[1][1][0][1]-HPI_train[1][1][0][0]),
                                                                                            ((HPI_val[1][1][0][1]-HPI_val[1][1][0][0]) - (HPI_pred_val[1][1][0][1]-HPI_pred_val[1][1][0][0]))/(HPI_val[1][1][0][1]-HPI_val[1][1][0][0]),
                                                                                            ((HPI_test[1][1][0][1]-HPI_test[1][1][0][0]) - (HPI_pred_test[1][1][0][1]-HPI_pred_test[1][1][0][0]))/(HPI_test[1][1][0][1]-HPI_test[1][1][0][0]), 
                                                                                            ((HPI_train[1][1][0][1]-HPI_train[1][1][0][0]) - (HPI_test[1][1][0][1]-HPI_test[1][1][0][0]))/(HPI_train[1][1][0][1]-HPI_train[1][1][0][0])]
    KS_test_pred_train = [[ks_w(X_test[idx_test,q],X_train[idx_train,q], np.ones(len(idx_test)), Weights_train)] for q in range(len(X_train[0]))]
    KS_test_pred_val = [[ks_w(X_test[idx_test,q],X_val[idx_val,q], np.ones(len(idx_test)), Weights_val)] for q in range(len(X_train[0]))]
    KS_val_pred_test = [[ks_w(X_val[idx_val,q],X_test[idx_test,q], np.ones(len(idx_val)), Weights_test)] for q in range(len(X_train[0]))]
    KS_train_test = [[ks_w(X_train[idx_train,q],X_test[idx_test,q], np.ones(len(idx_train)), np.ones(len(idx_test)))] for q in range(len(X_train[0]))]
    KS_test_pred_train_median = np.median(np.array(KS_test_pred_train)[:,0][:,1])
    KS_test_pred_val_median = np.median(np.array(KS_test_pred_val)[:, 0][:, 1])
    KS_val_pred_test_median = np.median(np.array(KS_val_pred_test)[:,0][:,1])
    KS_train_test_median = np.median(np.array(KS_train_test)[:, 0][:, 1])
    end = timer()
    print('Bayesian inference done in', end-start, 's.')
    [tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean] = ["None", "None", "None", "None", "None", "None", "None"]
    if FREQUENTISTS_RESULTS:
         print('Estimating frequentist inference')
         start_tmu = timer()
         blst = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
         tmuexact = np.array(list(map(tmu, blst)))
         tmuDNN = np.array(list(map(lambda x: tmu_DNN(x, model, scalerX, scalerY), blst)))
         [tmusample001, tmusample005, tmusample01, tmusample02] = [np.array(list(map(lambda x: tmu_sample(x, X_train, Y_train, binsize), blst))) for binsize in [0.01, 0.05, 0.1, 0.2]]
         tmu_err_mean = np.mean(np.abs(tmuexact[:, -1]-tmuDNN[:, -1]))
         end_tmu = timer()
         print('Frequentist inference done in', start_tmu-end_tmu, 's.')
    end_global = timer()
    print('Total time for predictions:',end_global-start_global,'s')
    return [metrics_true, metrics_scaled, HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, 
            one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
            KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median,
            tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time]

def generate_summary_log(model,now,FILE_SAMPLES,NDIM,NEVENTS_TRAIN,NEVENTS_VAL,NEVENTS_TEST,WEIGHT_SAMPLES,SCALE_X,SCALE_Y,LOSS,HID_LAYERS,DROPOUT_RATE,EARLY_STOPPING,REDUCE_LR_PATIENCE,MIN_DELTA,ACT_FUNC_OUT_LAYER,BATCH_NORM,
                         LEARNING_RATE,BATCH_SIZE,EXACT_EPOCHS,GPU_names,N_GPUS,training_time, metrics, metrics_scaled, 
                         HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                         KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, prediction_time, FREQUENTISTS_RESULTS,
                         tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean):
    summary_log = {**metrics,**metrics_scaled}
    summary_log['Date time'] = str(now)
    summary_log['Samples file'] = FILE_SAMPLES
    summary_log['Ndim'] = NDIM
    summary_log['Nevt-train'] = NEVENTS_TRAIN
    summary_log['Nevt-val'] = NEVENTS_VAL
    summary_log['Nevt-test'] = NEVENTS_TEST
    summary_log['Weighted'] = WEIGHT_SAMPLES
    summary_log['Scaled X'] = SCALE_X
    summary_log['Scaled Y'] = SCALE_Y
    if type(LOSS) == str:
        summary_log['Loss'] = LOSS
    else:
        summary_log['Loss'] = str(LOSS).split(' ')[1]
    summary_log['Hidden layers'] = HID_LAYERS
    summary_log['Params'] = model_params(model)
    summary_log['Trainable params'] = model_trainable_params(model)
    summary_log['Non-trainable params'] = model_non_trainable_params(model)
    summary_log['Dropout'] = DROPOUT_RATE
    summary_log['Early stopping'] = EARLY_STOPPING
    summary_log['Reduce LR patience'] = REDUCE_LR_PATIENCE
    summary_log['Min_delta'] = MIN_DELTA
    summary_log['AF out'] = ACT_FUNC_OUT_LAYER
    summary_log['Batch norm'] = BATCH_NORM
    summary_log['Optimizer'] = 'Adam (LR' + str(LEARNING_RATE)+')'
    summary_log['Batch size'] = BATCH_SIZE
    summary_log['Epochs'] = EXACT_EPOCHS
    summary_log['GPU(s)'] = GPU_names[:N_GPUS]
    summary_log['Training time'] = training_time
    summary_log['HPI train'] = HPI_train
    summary_log['HPI val'] = HPI_val
    summary_log['HPI test'] = HPI_test
    summary_log['HPI pred train'] = HPI_pred_train
    summary_log['HPI pred val'] = HPI_pred_val
    summary_log['HPI pred test'] = HPI_pred_test
    summary_log['1$\sigma$ HPI rel err train'] = one_sigma_HPI_rel_err_train
    summary_log['1$\sigma$ HPI rel err val'] = one_sigma_HPI_rel_err_val
    summary_log['1$\sigma$ HPI rel err test'] = one_sigma_HPI_rel_err_test
    summary_log['1$\sigma$ HPI rel err train-test'] = one_sigma_HPI_rel_err_train_test
    summary_log['KS test-pred_train'] = KS_test_pred_train
    summary_log['KS test-pred_val'] = KS_test_pred_val
    summary_log['KS val-pred_test'] = KS_val_pred_test
    summary_log['KS train-test'] = KS_train_test
    summary_log['KS test-pred_train median'] = KS_test_pred_train_median
    summary_log['KS test-pred_val median'] = KS_test_pred_val_median
    summary_log['KS val-pred_test median'] = KS_val_pred_test_median
    summary_log['KS train-test median'] = KS_train_test_median
    summary_log['Prediction time'] = prediction_time
    if FREQUENTISTS_RESULTS:
        summary_log['Frequentist tmu exact'] = tmuexact.tolist()
        summary_log['Frequentist tmu DNN'] = tmuDNN.tolist()
        summary_log['Frequentist tmu sample 0.01'] = tmusample001.tolist()
        summary_log['Frequentist tmu sample 0.05'] = tmusample005.tolist()
        summary_log['Frequentist tmu sample 0.1'] = tmusample01.tolist()
        summary_log['Frequentist tmu sample 0.2'] = tmusample02.tolist()
        summary_log['Frequentist mean error on tmu'] = tmu_err_mean.tolist()
    for key in list(summary_log.keys()):
        summary_log[metric_name_abbreviate(key)] = summary_log.pop(key)
    return summary_log

def generate_title(date_time,ndim,nevt,layers,loss):
    title = date_time + " - "
    title = title + "Ndim: " + str(ndim) + " - "
    title = title + "Nevt: " + '%.E' % Decimal(str(nevt)) + " - "
    title = title + "Layers: " + str(len(layers)) + " - "
    title = title + "Nodes: " + str(layers[0][0]) + " - "
    title = title.replace("+","") + "Loss: " + str(loss)
    return title

def generate_title_from_log(summary_log):
    title = summary_log['Date time'] + " - "
    title = title + "Ndim: " + str(summary_log['Ndim']) + " - "
    title = title + "Nevt: " + '%.E' % Decimal(str(summary_log['Nevt-train'])) + " - "
    title = title + "Layers: " + str(len(summary_log['Hidden layers'])) + " - "
    title = title + "Nodes: " + str(summary_log['Hidden layers'][0][0]) + " - "
    title = title.replace("+","") + "Loss: " + str(summary_log['Loss'])
    return title

def generate_title_from_log_reduced(summary_log):
    title = "Nevt: " + '%.E' % Decimal(str(summary_log['Nevt-train'])) + " - "
    title = title + "Hid Layers: " + str(len(summary_log['Hidden layers'])) + " - "
    title = title + "Nodes: " + str(summary_log['Hidden layers'][0][0]) + " - "
    title = title.replace("+", "") + "Loss: " + str(summary_log['Loss'])
    return title
    
def generate_summary_text(summary_log,history,FREQUENTISTS_RESULTS):
    summary_text = "Samples file: " + \
        str(os.path.split(summary_log['Samples file'])[
            1].replace("_", "$\_$")) + "\n"
    summary_text = summary_text + "Layers: " + str(summary_log['Hidden layers']) + "\n"
    summary_text = summary_text + "Pars: " + str(summary_log['Params']) + "\n"
    summary_text = summary_text + "Trainable pars: " + str(summary_log['Trainable params']) + "\n"
    summary_text = summary_text + "Non-trainable pars: " + str(summary_log['Non-trainable params']) + "\n"
    summary_text = summary_text + "Scaled X: " + str(summary_log['Scaled X']) + "\n"
    summary_text = summary_text + "Scaled Y: " + str(summary_log['Scaled Y']) + "\n"
    summary_text = summary_text + "Dropout: " + str(summary_log['Dropout']) + "\n"
    summary_text = summary_text + "Early stopping: " + str(summary_log['Early stopping']) + "\n"
    summary_text = summary_text + "Reduce LR patience: " + str(summary_log['Reduce LR patience']) + "\n"
    summary_text = summary_text + "Min-delta " + str(summary_log['Min_delta']) + "\n"
    summary_text = summary_text + "AF out: " + str(summary_log['AF out']) + "\n"
    summary_text = summary_text + "Batch norm: " + str(summary_log['Batch norm']) + "\n"
    summary_text = summary_text + "Loss: " + str(summary_log['Loss']) + "\n"
    summary_text = summary_text + "Optimizer: " + summary_log['Optimizer'] + "\n"
    summary_text = summary_text + "Batch size: " + str(summary_log['Batch size']) + "\n"
    summary_text = summary_text + "Epochs: " + str(summary_log['Epochs']) + "\n"
    summary_text = summary_text + "GPU(s): " + str(summary_log['GPU(s)']) + "\n"
    summary_text = summary_text + "Best losses: " + '[' + '{0:1.2e}'.format(summary_log[summary_log['Loss']+'_best']) + ',' '{0:1.2e}'.format(summary_log["val_"+summary_log['Loss']+'_best']) + ',' '{0:1.2e}'.format(summary_log["test_"+summary_log['Loss']+'_best']) + ']' + "\n"
    summary_text = summary_text + "Best losses scaled: " + '[' + '{0:1.2e}'.format(summary_log[summary_log['Loss']+'_best_scaled']) + ',' '{0:1.2e}'.format(summary_log["val_"+summary_log['Loss']+'_best_scaled']) + ',' '{0:1.2e}'.format(summary_log["test_"+summary_log['Loss']+'_best_scaled']) + ']' + "\n"
    summary_text = summary_text + "Pred. mean error: " + '[' + '{0:1.2e}'.format(summary_log['me_best']) + ',' + '{0:1.2e}'.format(summary_log['val_me_best']) + ',' + '{0:1.2e}'.format(summary_log['test_me_best']) +  ']'  + "\n"
    summary_text = summary_text + "1$\sigma$ HPI rel err: " + '[' + '{0:1.2e}'.format(summary_log['1$\sigma$ HPI rel err train']) + ',' + '{0:1.2e}'.format(summary_log['1$\sigma$ HPI rel err val']) + ',' + '{0:1.2e}'.format(summary_log['1$\sigma$ HPI rel err test']) + ',' + '{0:1.2e}'.format(summary_log['1$\sigma$ HPI rel err train-test']) + ']'  + "\n"
    summary_text = summary_text + "KS $p$-median: " + '[' + '{0:1.2e}'.format(summary_log['KS test-pred_train median']) + ',' + '{0:1.2e}'.format(summary_log['KS test-pred_val median']) + ',' + '{0:1.2e}'.format(summary_log['KS val-pred_test median']) + ',' + '{0:1.2e}'.format(summary_log['KS train-test median']) + ']' + "\n"
    if FREQUENTISTS_RESULTS:
        summary_text = summary_text + "Mean error on tmu: "+ str(summary_log['Frequentist mean error on tmu']) + "\n"
    summary_text = summary_text + "Train time: " + str(round(summary_log['Training time'],1)) + "s" + "\n"
    summary_text = summary_text + "Pred time: " + str(round(summary_log['Prediction time'],1)) + "s"
    return summary_text
    
def generate_summary_text_reduced(summary_log,FREQUENTISTS_RESULTS):
    history = summary_log
    summary_text = "Trainable pars: " + str(summary_log['Trainable params']) + "\n"
    summary_text = summary_text + "Scaled X: " + str(summary_log['Scaled X']) + "\n"
    summary_text = summary_text + "Scaled Y: " + str(summary_log['Scaled Y']) + "\n"
    summary_text = summary_text + "Act func hid layers: " + str(summary_log["Hidden layers"][0][1]) + "\n"
    summary_text = summary_text + "Act func out layer: " + str(summary_log['AF out']) + "\n"
    summary_text = summary_text + "Dropout: " + str(summary_log['Dropout']) + "\n"
    summary_text = summary_text + "Early stopping: " + str(summary_log['Early stopping']) + "\n"
    summary_text = summary_text + "Reduce LR patience: " + str(summary_log['Reduce LR patience']) + "\n"
    summary_text = summary_text + "Batch norm: " + str(summary_log['Batch norm']) + "\n"
    summary_text = summary_text + "Optimizer: " + summary_log['Optimizer'] + "\n"
    summary_text = summary_text + "Batch size: " + str(summary_log['Batch size']) + "\n"
    summary_text = summary_text + "Epochs: " + str(summary_log['Epochs']) + "\n"
    summary_text = summary_text + "GPU: " + summary_log['GPU(s)'][0] + "\n"
    summary_text = summary_text + "Min losses: " + '[' + '{0:1.2e}'.format(min(history['loss'])) + ',' '{0:1.2e}'.format(min(history['val_loss'])) + ']' + "\n"
    if FREQUENTISTS_RESULTS:
        summary_text = summary_text + "Mean error on tmu: "+ str(summary_log['Frequentist mean error on tmu']) + "\n"
    summary_text = summary_text + "Training time: " + str(round(summary_log['Training time'],1)) + "s" + "\n"
    summary_text = summary_text + "Prediction time: " + str(round(summary_log['Prediction time'],1)) + "s"
    return summary_text

def generate_training_data(GENERATE_DATA, GENERATE_DATA_ON_THE_FLY, LOAD_MODEL, allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, FILE_SAMPLES, LOGPROB_THRESHOLD_INDICES_TRAIN,LOGPROB_THRESHOLD_INDICES_TEST, NEVENTS_TRAIN, NEVENTS_VAL, NEVENTS_TEST, WEIGHT_SAMPLES, SCALE_X, SCALE_Y):
    if GENERATE_DATA:
        print('Generating training data')
        if len(LOGPROB_THRESHOLD_INDICES_TRAIN) < int(NEVENTS_TRAIN+NEVENTS_VAL):
            print(
                'Please increase LOGPROB_THRESHOLD or reduce NEVENTS. There are not enough training (and validation) samples with the requires LOGPROB_THRESHOLD.')
            CONTINUE = False
        if len(LOGPROB_THRESHOLD_INDICES_TEST) < int(NEVENTS_TEST):
            print(
                'Please increase LOGPROB_THRESHOLD or reduce NEVENTS. There are not enough test samples with the requires LOGPROB_THRESHOLD.')
            CONTINUE = False
        else:
            CONTINUE = True
        if CONTINUE:
            rnd_indices = np.random.choice(LOGPROB_THRESHOLD_INDICES_TRAIN, size=int(NEVENTS_TRAIN+NEVENTS_VAL), replace= False)
            rnd_indices_test = LOGPROB_THRESHOLD_INDICES_TEST[:NEVENTS_TEST]
            [rnd_indices_train, rnd_indices_val] = train_test_split(rnd_indices, train_size=NEVENTS_TRAIN, test_size=NEVENTS_VAL)
            if GENERATE_DATA_ON_THE_FLY:
                [X_train, Y_train] = import_XY_train(FILE_SAMPLES, rnd_indices_train)
                [X_val, Y_val] = import_XY_train(FILE_SAMPLES, rnd_indices_val)
                [X_test, Y_test] = import_XY_test(FILE_SAMPLES, rnd_indices_test)
            else:
                [X_train, Y_train] = [allsamples_train[rnd_indices_train],logprob_values_train[rnd_indices_train]]
                [X_val, Y_val] = [allsamples_train[rnd_indices_val],logprob_values_train[rnd_indices_val]]
                [X_test, Y_test] = [allsamples_test[rnd_indices_test],logprob_values_test[rnd_indices_test]]
            if WEIGHT_SAMPLES:
                W_train = compute_sample_weights(X_train, 500,power=1/1.3)
                W_val = compute_sample_weights(X_val, 500,power=1/1.3)
                W_test = compute_sample_weights(X_test, 500,power=1/1.3)
            else:
                W_train = np.full(len(X_train), 1)
                W_val = np.full(len(X_val), 1)
                W_test = np.full(len(X_test), 1)
            if SCALE_X:
                scalerX = StandardScaler(
                    with_mean=True, with_std=True)
                scalerX.fit(X_train)
            else:
                scalerX = StandardScaler(
                    with_mean=False, with_std=False)
                scalerX.fit(X_train)
            if SCALE_Y:
                scalerY = StandardScaler(
                    with_mean=True, with_std=True)
                scalerY.fit(Y_train.reshape(-1, 1))
            else:
                scalerY = StandardScaler(
                    with_mean=False, with_std=False)
                scalerY.fit(Y_train.reshape(-1, 1))
    elif GENERATE_DATA == False and LOAD_MODEL != 'None':
        print('Loading data for model',LOAD_MODEL)
        [rnd_indices_train, rnd_indices_val,rnd_indices_test] = load_data_indices(LOAD_MODEL)
        if GENERATE_DATA_ON_THE_FLY:
            [X_train, Y_train]= import_XY_train(FILE_SAMPLES, rnd_indices_train)
            [X_val, Y_val]= import_XY_train(FILE_SAMPLES, rnd_indices_val)
            [X_test, Y_test]= import_XY_test(FILE_SAMPLES, rnd_indices_test)
        else:
            [X_train, Y_train] = [allsamples_train[rnd_indices_train], logprob_values_train[rnd_indices_train]]
            [X_val, Y_val] = [allsamples_train[rnd_indices_val], logprob_values_train[rnd_indices_val]]
            [X_test, Y_test] = [allsamples_train[rnd_indices_test], logprob_values_train[rnd_indices_test]]
        if WEIGHT_SAMPLES:
            W_train = compute_sample_weights(X_train, 500, power=1/1.3)
            W_val = compute_sample_weights(X_val, 500, power=1/1.3)
            W_test = compute_sample_weights(X_test, 500, power=1/1.3)
        else:
            W_train= np.full(len(X_train), 1)
            W_val= np.full(len(X_val), 1)
            W_test= np.full(len(X_test), 1)
        [scalerX, scalerY]=load_model_scaler(LOAD_MODEL)
    try:
        [X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, rnd_indices_train, rnd_indices_val, rnd_indices_test, scalerX, scalerY]
        CONTINUE = True
        return [X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, rnd_indices_train, rnd_indices_val, rnd_indices_test, scalerX, scalerY, CONTINUE]
    except:
        print("No training data have been generated (GENERATE_DATA=False) and LOAD_MODEL = 'None'. Please change your selection to continue training.")
        CONTINUE = False
        return ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', CONTINUE]

def model_training_scan(N_RUNS,ACT_FUNC_OUT_LAYER_LIST,BATCH_NORM_LIST,BATCH_SIZE_LIST,CONTINUE_TRAINING,DROPOUT_RATE_LIST,EARLY_STOPPING,FILE_SAMPLES_LIST,
                        FOLDER,FREQUENTISTS_RESULTS,GENERATE_DATA,GENERATE_DATA_ON_THE_FLY,GPU_NAMES,HID_LAYERS_LIST,LABELS,LEARNING_RATE_LIST,LOAD_MODEL,
                        LOGPROB_THRESHOLD, LOGPROB_THRESHOLD_INDICES_TRAIN, LOGPROB_THRESHOLD_INDICES_TEST, LOSS_LIST, METRICS, MIN_DELTA_LIST, MODEL_CHEKPOINT, MONITORED_METRIC, MULTI_GPU, N_EPOCHS, NEVENTS_TRAIN_LIST, PARS, PLOTLOSSES, REDUCE_LR, REDUCE_LR_PATIENCE_LIST,
                        SCALE_X, SCALE_Y, TEST_FRACTION, VALIDATION_FRACTION, WEIGHT_SAMPLES_LIST,
                        allsamples_train='None',logprob_values_train='None',allsamples_test='None',logprob_values_test='None',
                        rnd_indices_train='None', rnd_indices_val='None', rnd_indices_test='None', X_train='None', X_val='None', X_test='None',
                        Y_train='None',Y_val='None',Y_test='None',W_train='None',W_val='None',scalerX='None',scalerY='None',
                        model='None', training_model='None',summary_log='None',history='None', training_time='None'):
    start = timer()
    overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={'width': '500px', 'height': '14px', 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
    display(overall_progress)
    iterator = 0
    for FILE_SAMPLES in FILE_SAMPLES_LIST:
        if GENERATE_DATA_ON_THE_FLY:
            print('Training and test data will be generated on-the-fly for each run to save RAM')
            logprob_values_train = import_Y_train(FILE_SAMPLES,'all')
            LOGPROB_THRESHOLD_INDICES_TRAIN = np.nonzero(
                logprob_values_train >= LOGPROB_THRESHOLD)[0]
            logprob_values_train = 'None'
            logprob_values_test = import_Y_test(FILE_SAMPLES, 'all')
            LOGPROB_THRESHOLD_INDICES_TEST = np.nonzero(
                logprob_values_test >= LOGPROB_THRESHOLD)[0]
            logprob_values_test = 'None'
        elif GENERATE_DATA and GENERATE_DATA_ON_THE_FLY == False:
            if [allsamples_train, logprob_values_train] != ['None', 'None']:
                print('Training data already loaded')
            else:
                print('Loading training data')
                allsamples_train, logprob_values_train = import_XY_train(FILE_SAMPLES,'all')
            if [allsamples_test, logprob_values_test] != ['None', 'None']:
                print('Test data already loaded')
            else:
                print('Loading test data')
                allsamples_test, logprob_values_test = import_XY_test(FILE_SAMPLES,'all')
            LOGPROB_THRESHOLD_INDICES_TRAIN = np.nonzero(
                logprob_values_train >= LOGPROB_THRESHOLD)[0]
            LOGPROB_THRESHOLD_INDICES_TEST = np.nonzero(
                logprob_values_test >= LOGPROB_THRESHOLD)[0]
        elif GENERATE_DATA == False and CONTINUE_TRAINING:
            CONTINUE = True
        else:
            try:
                LOGPROB_THRESHOLD_INDICES_TRAIN = np.nonzero(
                    logprob_values_train >= LOGPROB_THRESHOLD)[0]
                LOGPROB_THRESHOLD_INDICES_TEST = np.nonzero(
                    logprob_values_test >= LOGPROB_THRESHOLD)[0]
                print('Training and test data already loaded')
            except:
                print("No training data available, please generate them by setting GENERATE_DATA=True")
                CONTINUE = False
        if LOAD_MODEL != 'None':
            if CONTINUE_TRAINING == False:
                print('When loading model CONTINUE_TRAINING flag needs to be set to True, while it was set to False. Flag automatically changed to True.')
                CONTINUE_TRAINING = True
        if len(K.tensorflow_backend._get_available_gpus()) <= 1:
            MULTI_GPU = False
            N_GPUS = len(K.tensorflow_backend._get_available_gpus())
        if MULTI_GPU:
            N_GPUS = len(K.tensorflow_backend._get_available_gpus())
        else:
            N_GPUS = 1
        if MULTI_GPU:
            BATCH_SIZE_LIST = [i*N_GPUS for i in BATCH_SIZE_LIST]
        try:
            LABELS
            CONTINUE = True
        except:
            print("Please provide labels for the features")
            CONTINUE = False
        if CONTINUE:    
            for run in range(N_RUNS):
                for NEVENTS_TRAIN in NEVENTS_TRAIN_LIST:
                    NEVENTS_VAL = int(NEVENTS_TRAIN*VALIDATION_FRACTION)
                    NEVENTS_TEST = int(NEVENTS_TRAIN*TEST_FRACTION)
                    for WEIGHT_SAMPLES in WEIGHT_SAMPLES_LIST:
                        if CONTINUE_TRAINING == False or GENERATE_DATA or LOAD_MODEL != 'None':
                            [X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, rnd_indices_train, rnd_indices_val, rnd_indices_test, scalerX, scalerY, CONTINUE] = generate_training_data(
                                GENERATE_DATA, GENERATE_DATA_ON_THE_FLY, LOAD_MODEL, allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, FILE_SAMPLES, LOGPROB_THRESHOLD_INDICES_TRAIN,LOGPROB_THRESHOLD_INDICES_TEST, NEVENTS_TRAIN, NEVENTS_VAL, NEVENTS_TEST, WEIGHT_SAMPLES, SCALE_X, SCALE_Y)
                        if X_train != "None":
                            CONTINUE = True
                        else:
                            print("No training data available, please change flags to ensure data generation.")
                            CONTINUE = False
                        if CONTINUE:
                            for LOSS in LOSS_LIST:
                                for HID_LAYERS in HID_LAYERS_LIST:
                                    for ACT_FUNC_OUT_LAYER in ACT_FUNC_OUT_LAYER_LIST:
                                        for DROPOUT_RATE in DROPOUT_RATE_LIST:
                                            for BATCH_SIZE in BATCH_SIZE_LIST:
                                                for BATCH_NORM in BATCH_NORM_LIST:
                                                    for LEARNING_RATE in LEARNING_RATE_LIST:
                                                        for REDUCE_LR_PATIENCE in REDUCE_LR_PATIENCE_LIST:
                                                            for MIN_DELTA in MIN_DELTA_LIST:
                                                                now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                                                                NDIM = len(X_train[0])
                                                                #Model title
                                                                title = generate_title(now,NDIM,NEVENTS_TRAIN,HID_LAYERS,LOSS)
                                                                if LABELS == 'None':
                                                                    LABELS = [r"$x_{%d}$"%i for i in range(NDIM)]
                                                                OPTIMIZER = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.95, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                                                                if CONTINUE_TRAINING == False:
                                                                    if model == "None":
                                                                        model = model_define(NDIM,HID_LAYERS,DROPOUT_RATE,ACT_FUNC_OUT_LAYER,BATCH_NORM,verbose=1)
                                                                    elif type(model.input_shape) == tuple:
                                                                        model = model_define(NDIM,HID_LAYERS,DROPOUT_RATE,ACT_FUNC_OUT_LAYER,BATCH_NORM,verbose=1)
                                                                    model = model_compile(model,LOSS,OPTIMIZER,METRICS,False)
                                                                    training_model = model_compile(model,LOSS,OPTIMIZER,METRICS,MULTI_GPU)
                                                                    [history, training_time] = [{}, 0]
                                                                #Model training
                                                                if LOAD_MODEL != 'None':
                                                                    print('Loading model',LOAD_MODEL)
                                                                    model = load_model(LOAD_MODEL, custom_objects={'R2_metric': R2_metric, 'Rt_metric':Rt_metric})
                                                                    model = model_compile(model,LOSS,OPTIMIZER,METRICS,False)
                                                                    training_model = model_compile(model,LOSS,OPTIMIZER,METRICS,MULTI_GPU)
                                                                    LOAD_MODEL = 'None'
                                                                if CONTINUE_TRAINING:
                                                                    print('Continue training of loaded model')
                                                                else:
                                                                    if LOAD_MODEL != 'None':
                                                                        print('Continue training of loaded model')
                                                                    else:
                                                                        print('Start training of new model')
                                                                [h_run, training_time_run] = model_train(training_model, X_train, Y_train, X_val, Y_val, scalerX, scalerY, N_EPOCHS, BATCH_SIZE,
                                                                                                         sample_weights=W_train, folder=FOLDER, title=title, monitored_metric=MONITORED_METRIC, 
                                                                                                         plotlosses = PLOTLOSSES, model_checkpoint=MODEL_CHEKPOINT, early_stopping=EARLY_STOPPING,restore_best_weights=True,
                                                                                                         reduceLR=REDUCE_LR,reduce_LR_patience=REDUCE_LR_PATIENCE, min_delta=MIN_DELTA, verbose=2)
                                                                if CONTINUE_TRAINING == False:
                                                                    [history, training_time] = [h_run.history, training_time_run]
                                                                else:
                                                                    if LOAD_MODEL != 'None':
                                                                        with open(LOAD_MODEL.replace('model.h5','history.json')) as json_file:
                                                                            history = json.load(json_file)
                                                                        history_full = {}
                                                                        history_run = h_run.history
                                                                        for key in history_run.keys():
                                                                            history_full[key] = history[key] + history_run[key]
                                                                        [history, training_time] = [history_full, training_time + training_time_run]
                                                                        del(history_full,history_run)
                                                                    else:    
                                                                        history_full = {}
                                                                        history_run = h_run.history
                                                                        for key in history_run.keys():
                                                                            history_full[key] = history[key] + history_run[key]
                                                                        [history, training_time] = [history_full, training_time + training_time_run]
                                                                        del(history_full,history_run)
                                                                # compute predictions
                                                                [metrics, metrics_scaled,
                                                                HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                                                                KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median,
                                                                tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time] = compute_predictions(model, scalerX, scalerY, X_train, X_val, X_test, Y_train, Y_val, Y_test, LOSS, NEVENTS_TRAIN, BATCH_SIZE, FREQUENTISTS_RESULTS)
                                                                EXACT_EPOCHS = len(history['loss'])
                                                                #Model log
                                                                summary_log = generate_summary_log(model,now, FILE_SAMPLES, NDIM, NEVENTS_TRAIN, NEVENTS_VAL, NEVENTS_TEST, WEIGHT_SAMPLES, SCALE_X, SCALE_Y, LOSS, HID_LAYERS, DROPOUT_RATE, EARLY_STOPPING, REDUCE_LR_PATIENCE, MIN_DELTA, ACT_FUNC_OUT_LAYER, BATCH_NORM,
                                                                                                   LEARNING_RATE, BATCH_SIZE, EXACT_EPOCHS, GPU_NAMES, N_GPUS, training_time, metrics, metrics_scaled,
                                                                                                   HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                                                                                                   KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, prediction_time, FREQUENTISTS_RESULTS,
                                                                                                   tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean)
                                                                #Summary
                                                                summary_text = generate_summary_text(
                                                                    summary_log, history, FREQUENTISTS_RESULTS)
                                                                #Summary figure saving 
                                                                print('Saving model')
                                                                model_save_fig(FOLDER,history,title,summary_text,metrics=[LOSS,MONITORED_METRIC],yscale='log')
                                                                model_store(FOLDER, rnd_indices_train, rnd_indices_val, rnd_indices_test,
                                                                            model, scalerX, scalerY, history, title, summary_log)
                                                                print('Saving results')
                                                                save_results(FOLDER, model, scalerX, scalerY, title, summary_text, X_train, X_test, Y_train, Y_test, tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, pars=PARS, 
                                                                        labels=LABELS, plot_coverage=False, plot_distr=True, plot_corners=True, plot_tmu=FREQUENTISTS_RESULTS, batch_size=BATCH_SIZE, verbose=True)
                                                                iterator = iterator + 1
                                                                overall_progress.value = float(iterator)/(len(FILE_SAMPLES_LIST)*len(NEVENTS_TRAIN_LIST)*len(LEARNING_RATE_LIST)*len(BATCH_NORM_LIST)*len(
                                                                    LOSS_LIST)*len(HID_LAYERS_LIST)*len(ACT_FUNC_OUT_LAYER_LIST)*len(DROPOUT_RATE_LIST)*len(BATCH_SIZE_LIST)*len(REDUCE_LR_PATIENCE_LIST)*len(MIN_DELTA_LIST)*N_RUNS)
                                                                print("Processed NN:" + summary_text.replace("\n"," / "))
        end = timer()
        if CONTINUE:
            print("Processed " + str(len(FILE_SAMPLES_LIST)*len(NEVENTS_TRAIN_LIST)*len(LEARNING_RATE_LIST)*len(BATCH_NORM_LIST)*len(LOSS_LIST)*len(HID_LAYERS_LIST) *
                                     len(ACT_FUNC_OUT_LAYER_LIST)*len(DROPOUT_RATE_LIST)*len(BATCH_SIZE_LIST)*len(REDUCE_LR_PATIENCE_LIST)*len(MIN_DELTA_LIST)*N_RUNS) + " models in " + str(int(end-start)) + " s")
    return [allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, LOGPROB_THRESHOLD_INDICES_TRAIN, LOGPROB_THRESHOLD_INDICES_TEST, rnd_indices_train, rnd_indices_val, rnd_indices_test, X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, scalerX, scalerY, model, training_model, summary_log, history, training_time]

def plot_corners(ilist, nbins, samp1, samp2, w1=None, w2=None, levels1=None, levels2=None, HPI_intervals1=None, HPI_intervals2=None, ranges=None, title1=None, title2=None, color1='green', color2='red', plot_title= "Params contours", legend_labels=None, figdir=None, figname=None):
    jtplot.reset()
    plt.style.use('matplotlib.mplstyle')

    start = timer()
    linewidth = 1.3
    nndim = len(ilist)
    if ilist[0] == 0:
        labels = ['$\mu$']
        for i in ilist[1:]:
            labels = np.append(labels, ['$\delta_{'+str(i+1)+'}$'])
    else:
        labels = ['$\delta_{'+str(ilist[0])+'}$']
        for i in ilist[1:]:
            labels = np.append(labels, ['$\delta_{'+str(i+1)+'}$'])
    fig, axes = plt.subplots(nndim, nndim, figsize=(3*nndim, 3*nndim))
    figure1 = corner(samp1, bins=nbins, weights=w1, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color=color1, plot_contours=True, smooth=True, smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False, fill_contours=False, normalize1d=True,
                     hist_kwargs={'color': color1, 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels1, data_kwargs={"alpha": 1}, contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(HPI_intervals1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPI_intervals1[0])]},
                     no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color1], "alpha": 1})
    figure2 = corner(samp2, bins=nbins, weights=w2, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color=color2, plot_contours=True, smooth=True, range=ranges, smooth1d=True, plot_datapoints=True, plot_density=False, fill_contours=False, normalize1d=True,
                     hist_kwargs={'color': color2, 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels2, data_kwargs={"alpha": 1}, contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][0:len(HPI_intervals1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPI_intervals1[0])]},
                     no_fill_contours=False, contourf_kwargs={"colors": ["white", "tomato", color2], "alpha": 1})
    axes = np.array(figure1.axes).reshape((nndim, nndim))
    for i in range(nndim):
        ax = axes[i, i]
        title = ""
        ax.grid(True, linestyle='--', linewidth=1, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        HPI681 = HPI_intervals1[i][0][1]
        HPI951 = HPI_intervals1[i][1][1]
        HPI3s1 = HPI_intervals1[i][2][1]
        HPI682 = HPI_intervals2[i][0][1]
        HPI952 = HPI_intervals2[i][1][1]
        HPI3s2 = HPI_intervals2[i][2][1]
        hists_1d_1 = get_1d_hist(i, samp1, nbins=nbins, ranges=ranges,
                                 weights=w1, normalize1d=True)[0]
        hists_1d_2 = get_1d_hist(i, samp2, nbins=nbins, ranges=ranges,
                                 weights=w2, normalize1d=True)[0]
        for j in HPI3s1:
            ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                       color=color1, alpha=1, linestyle=":", linewidth=linewidth)
            ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                       color=color1, alpha=1, linestyle=":", linewidth=linewidth)
        for j in HPI3s2:
            ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                       color=color2, alpha=1, linestyle=":", linewidth=linewidth)
            ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                       color=color2, alpha=1, linestyle=":", linewidth=linewidth)
        for j in HPI951:
            ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                       color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
            ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                       color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
        for j in HPI952:
            ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                       color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
            ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                       color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
        for j in HPI681:
            ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                       color=color1, alpha=1, linestyle="--", linewidth=linewidth)
            ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                       color=color1, alpha=1, linestyle="--", linewidth=linewidth)
            title = title+title1 + \
                ": ["+'{0:1.2e}'.format(j[0])+","+'{0:1.2e}'.format(j[1])+"]"
        title = title+"\n"
        for j in HPI682:
            ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                       color=color2, alpha=1, linestyle="--", linewidth=linewidth)
            ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                       color=color2, alpha=1, linestyle="--", linewidth=linewidth)
            title = title+title2 + \
                ": ["+'{0:1.2e}'.format(j[0])+","+'{0:1.2e}'.format(j[1])+"]"
        if i == 0:
            x1, x2, _, _ = ax.axis()
            ax.set_xlim(x1*1.3, x2)
        ax.set_title(title, fontsize=12)
    for yi in range(nndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            if xi == 0:
                x1, x2, _, _ = ax.axis()
                ax.set_xlim(x1*1.3, x2)
            ax.grid(True, linestyle='--', linewidth=1)
            ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    fig.text(0.53,0.97,r'%s'%plot_title, fontsize=26)
    colors = [color1,color2,'black', 'black', 'black']
    red_patch = matplotlib.patches.Patch(color=colors[0])
    blue_patch = matplotlib.patches.Patch(color=colors[1])
    line1 = matplotlib.lines.Line2D([0], [0], color=colors[0], lw=12)
    line2 = matplotlib.lines.Line2D([0], [0], color=colors[1], lw=12)
    line3 = matplotlib.lines.Line2D([0], [0], color=colors[2], linewidth=3, linestyle='--')
    line4 = matplotlib.lines.Line2D([0], [0], color=colors[3], linewidth=3, linestyle='-.')
    line5 = matplotlib.lines.Line2D([0], [0], color=colors[4], linewidth=3, linestyle=':')
    lines = [line1,line2,line3,line4,line5]
    fig.legend(lines, legend_labels, fontsize=26, loc=(0.53,0.8))
    plt.savefig(figdir + figname, dpi=50)
    plt.show()
    end = timer()
    print("Plot done and saved in", end-start, "s.")
