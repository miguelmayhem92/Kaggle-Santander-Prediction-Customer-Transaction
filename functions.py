import pandas as pd
import numpy as np
import random
import scipy as sp
from operator import attrgetter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import seaborn as sns; sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
    
def scaler(dataset, scaler=None):
    if scaler:
        df = dataset.iloc[:,1:]
        df_scaled = scaler.transform(df)
        dataset_scaled = pd.DataFrame(df_scaled, columns = df.columns)
        df_result = pd.concat([dataset.iloc[:,0:1],dataset_scaled], axis = 1)
        return df_result
    
    else:
        df  = dataset.iloc[:,1:]
        scaler = StandardScaler()
        scaler.fit(df)

        dataset_scaled = scaler.transform(df)
        dataset_scaled = pd.DataFrame(dataset_scaled, columns = df.columns)
        df_result = pd.concat([dataset.iloc[:,0:1],dataset_scaled], axis = 1)
        return df_result, scaler
    
def partitions(data, limits = [1.269331,-0.253846]):

    data_section_1 = data[data.mahalanobis > limits[0]]
    data_section_2 = data[(data.mahalanobis > limits[1]) & (data.mahalanobis <= limits[0])]
    data_section_3 = data[data.mahalanobis <= limits[1]]
    
    return data_section_1, data_section_2, data_section_3

def threeD_plot(dataset,variables,fraq = 0.10):
    variables = variables + ["target"]
    fig = plt.figure(figsize=(13,13))
    df  = dataset.sample(frac = fraq, replace = False)
    ax = Axes3D(fig)
    categories = list(df["target"].unique())
    categories.sort()
    for category in categories:
        df_plot = df[df["target"] == category]
        countx = len(df_plot)
        alpha = 0.7
        if category != 1:
            alpha = 0.4
        sc = ax.scatter(df_plot[variables[0]], df_plot[variables[1]], df_plot[variables[2]], marker='o', alpha=alpha, label= f'{category}-{countx}' )
        
    ax.legend()
    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[0])
    ax.set_zlabel(variables[0])
    ax.set_facecolor("grey")
    plt.show()
    

def mahalanobis(x=None, data=None,cov_mu = [False,False], cov=None, mu = None):
    if cov_mu[0] == False:
        cov = np.cov(data.values.T)
    elif cov_mu[0] == True:
        cov = cov
        
    if cov_mu[1] == False:
        x_mu = x - np.mean(data)
    elif cov_mu[1] == True:
        x_mu = x - mu

        
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

def mahalanobis_vector_plot(data):
    df = data.copy()
    df = df.sort_values('mahalanobis').reset_index(drop = True)
    fig = plt.figure(figsize=(10,7))
    plt.scatter(df.index, df['mahalanobis'],marker='o')
    plt.xlabel('order')
    plt.ylabel('Mahalanobis distance')
    fig.show()
    
def compute_mahalanobis_parts(data,numerical_features, chunk_size, cov_mu, mu, cov):
    
    chunks = list()
    num_chunks = len(data) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(data[i*chunk_size:(i+1)*chunk_size])
    
    treated_dfs = list()
    for chunk in chunks:
        chunk['mahalanobis'] = mahalanobis( x=chunk[numerical_features], data=chunk[numerical_features], cov_mu = cov_mu, mu = mu, cov = cov)
        treated_dfs.append(chunk)
    result_df = pd.concat(treated_dfs)
    #return chunks
    return result_df

def normal_augmentation(data, columns, n, pop_mean, pop_cov ,range_distance = [None,None], label_target = 2, rate = 0):
    calc = 0
    var_limits = dict()
    for var in columns:
        var_limits[var] = {'mu': np.mean(data[var]),'std': np.std(data[var])}
    new_data = list()
    while calc < n:
        data_dict = dict()
        data_dict['target'] = label_target
        for var in columns:
            mu, std = var_limits[var]['mu'], var_limits[var]['std'], 
            value = np.random.normal(mu, std*(1 + rate), 1)[0]
            data_dict[var] = value
        data = pd.DataFrame(data_dict,index = [calc])
        data['mahalanobis'] = mahalanobis( x=data[columns], data=data[columns], cov_mu = [True,True], mu = pop_mean, cov = pop_cov)
        if data['mahalanobis'].values > range_distance[0] and data['mahalanobis'].values < range_distance[1]:
            new_data.append(data)
            calc = calc + 1
    new_data = pd.concat(new_data)
    return new_data

def concentrated_augmentation(data, columns, n, category, label_target = 3, rate = 0.015):
    data_selection = data[data.target == category]
    lenght = len(data_selection)
    new_data_result = list()
    for i in range(n):
        new_data_dict = dict()
        new_data_dict['target'] = label_target
        index = np.random.randint(low = 0, high=lenght, size=1)[0]
        data_dict = data_selection.iloc[index,:].copy().to_dict()
        for var in columns:
            q75, q25 = np.percentile(data_selection[var], [75 ,25])
            iqr = q75 - q25
            dist_param = iqr * rate
            dist = np.random.uniform(-dist_param,dist_param,1)[0]

            new_value = data_dict[var] + dist
            new_data_dict[var] = new_value
            #print(f'var{var} before {data_dict[var]} new:{new_data_dict[var]}')
        new_data_dict['mahalanobis'] = data_dict['mahalanobis']
        #print(data_dict,new_data_dict)
        new_data_df = pd.DataFrame(new_data_dict, index = [i])
        new_data_result.append(new_data_df)
    new_data_result = pd.concat(new_data_result)
    return new_data_result

def get_dictionary_count_lables(data,variables,bins):
    labels_dictionary = dict()

    for column in variables:
        high_limits = list(pd.cut(data[column], bins= bins).map(attrgetter('right')).astype(float).sort_values().unique())
        low_limits = list(pd.cut(data[column], bins= bins).map(attrgetter('left')).astype(float).sort_values().unique())
        labels_list = list()
        ## Geting the labels
        for low,high in zip(low_limits, high_limits):
            countx = len(data[(data[column] >= low) & (data[column] <= high) & (data.target == 1)] )
            labels_list.append(countx)

        labels_dictionary[column] = {'low': low_limits, 'high': high_limits, 'label': labels_list}
    return labels_dictionary

def count_encoding(data, dictionary_lables, variables):
    data_result = data.copy()
    for variable in variables:
        lowers = dictionary_lables[variable]['low']
        highers = dictionary_lables[variable]['high']
        labels = dictionary_lables[variable]['label']
        data_result[f'count{variable}'] = 0
        for low,high,label in zip(lowers,highers,labels):
            data_result[f'count{variable}'] = np.where((data_result[variable] >= low) & (data_result[variable] <= high), label, data_result[f'count{variable}'] )
    return data_result


def augmentation_strategy(data, dict_1,dict_2, dict_3,gen_mean, gen_cov, columns_features ):
    aug_normal = normal_augmentation(data = data , columns = columns_features, n = dict_1['n'],
                          pop_mean = gen_mean, pop_cov = gen_cov, range_distance = dict_1['range'], label_target = dict_1['label'], rate = dict_1['rate'])
    
    aug_conditioned_1 = concentrated_augmentation(data = data, columns = columns_features, n = dict_2['n'], category = dict_2['category'], label_target = dict_2['label'])
    
    aug_conditioned_2 = concentrated_augmentation(data = aug_normal, columns = columns_features, n = dict_3['n'], category = dict_3['category'], label_target = dict_3['label'])
    
    augmentation_result = pd.concat([data, aug_normal,  aug_conditioned_1, aug_conditioned_2])
    
    return augmentation_result 

def augmentation_selection_rates(data, rate = 0.0, sample = 1.0, reduction_falses = 0):
    data1 = data[data.target.isin([0,1])].copy() 
    data2 = data[~data.target.isin([0,1])].sample(frac = rate).copy() ### For the augmentation
    data2['target'] = 1
    
    data_concat = pd.concat([data1,data2]).sample(frac = sample).copy()
    
    array_index = np.array(data_concat[data_concat.target.isin([0])].index)
    np.random.shuffle(array_index)
    array_index = list(array_index)
    array_todrop = array_index[0:reduction_falses]
    dataresult = data_concat[~data_concat.index.isin(array_todrop)]
    
    return dataresult

def metrics_train_validation(model, X_train, Y_train, X_val, Y_val):
    predictions_train, predictions_probas = model.predict(X_train), model.predict_proba(X_train)[:,1]
    
    metrics = {'Train' : {'precision': precision_score(Y_train.values,predictions_train),
          'Accuracy' : accuracy_score(Y_train.values,predictions_train),
          'AUC' : roc_auc_score(Y_train.values,predictions_probas)}}

    predictions_validation, predictions_probas = model.predict(X_val),  model.predict_proba(X_val)[:,1]

    metrics['Validation'] = {'precision': precision_score(Y_val.values,predictions_validation),
              'Accuracy' : accuracy_score(Y_val.values,predictions_validation),
              'AUC' : roc_auc_score(Y_val.values,predictions_probas)}

    to_conf_mat = pd.DataFrame({'True': Y_val, 'Pred' : predictions_validation})
    conf_mat = to_conf_mat.assign(ones = 1).pivot_table(index = 'True', columns = 'Pred',values = 'ones', aggfunc = 'count')
    
    return metrics, conf_mat, predictions_validation

def my_calibration_plot(model, calibrated_model, X_data, Y_data):
    
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(1,1)
    ax_calibration_curve = fig.add_subplot(gs[0])
    calibration_displays = {}
    display = CalibrationDisplay.from_estimator(model,X_data,Y_data,n_bins=10,name ='No calibrated', ax = ax_calibration_curve, color = 'red')
    calibration_displays['No calibrated'] = display
    
    
    ax = fig.add_subplot(gs[0])
    probs_calib = calibrated_model.predict_proba(X_data)[:, 1]
    fop_calib, mpv_calib = calibration_curve(Y_data, probs_calib, n_bins=10, normalize=True)
    
    ax.plot(mpv_calib, fop_calib, marker='.', color = 'blue', label = 'Calibrated')
    plt.legend()
    plt.show()

def balance_validation(data, additional = 0):
    ones = len(data[data.target == 1]) + additional
    ones_list  = list(data[data.target == 1].index)
    selected = list(data[data.target == 0].sample(n = ones, replace = False).copy().index)
    
    result_data = data[data.index.isin(ones_list + selected)]
    
    return result_data

def k_folds_indexs(data, folds = 5, balanced = False, validation_additional_false = 500, validation_reduce_false = 700 ):

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
    index = data[data.target.isin([0,1])].index
    array_index = np.array(index)
    np.random.shuffle(array_index)
    list_index = list(array_index)
    result_split = list(split(list_index, folds))
    
    result_validation_indexs = result_split
    
    if balanced:
        result_validation_indexs = list()
        for fold in result_split:
            temp_val = data[data.index.isin(fold)]
            temp_val =  balance_validation(temp_val, validation_additional_false )
            array = list(temp_val.index)
            result_validation_indexs.append(array)
    
    result_train_indexs = list()
    for fold in result_validation_indexs:
        train_data_raw = data[~data.index.isin(fold)]
        if validation_reduce_false:
            indexs = np.array(train_data_raw[train_data_raw.target.isin([0])].index)
            np.random.shuffle(indexs)
            indexs_selected = list(indexs)[0:validation_reduce_false]
        train_data_indexs = list(train_data_raw[~train_data_raw.index.isin(indexs_selected)].index)
        result_train_indexs.append(train_data_indexs)
            
    data_indexs = dict()
    for fold in range(1, folds+1):
        data_indexs[f'fold {fold}'] = { 'train index': result_train_indexs[fold-1], 'val index': result_validation_indexs[fold-1], }
    
    return data_indexs


def model_fitting_kfold(models, indexes_kfolds, features, train_data, rate_aug, sample_aug ):
    indexm = 1
    model_results = dict()
    for model,rate_param, sample_param in zip(models,rate_aug, sample_aug) :
        fold_result = dict()
        fold_i = 1
        for fold in indexes_kfolds.keys():
            
            train_index, val_index = indexes_kfolds[fold]['train index'] , indexes_kfolds[fold]['val index']

            validation_data = train_data[train_data.index.isin(val_index)]
            train_data_tomodel = train_data[train_data.index.isin(train_index)]
        
            train_data_tomodel_aug = augmentation_selection_rates(train_data_tomodel, rate = rate_param, sample = sample_param)

            X_train = train_data_tomodel_aug[features]
            Y_train = train_data_tomodel_aug['target']

            X_val = validation_data[features]
            Y_val = validation_data['target']
            
            #### MODEL TRAINING:
            my_model = model.fit(X_train, Y_train)
            result_metrics, _, _ = metrics_train_validation(my_model, X_train, Y_train, X_val, Y_val)
            
            fold_result[f'fold-{fold_i}'] = {'metrics': result_metrics}
            #print(f'fold {fold_i} done')
            fold_i = fold_i + 1
        model_results[f'model-{indexm}'] = fold_result
        print(f'done machine {indexm}')
        indexm = indexm + 1
    return model_results

def jsontotable(json, typex = 'Validation'):
    list_machines, list_folds, list_acc, list_auc = list(), list(), list(),list()
    
    for machine in json.keys():
        machine_json = json[machine]
        for fold in machine_json.keys():
            mets = machine_json[fold]['metrics'][typex]
            Acc, AUC = mets['Accuracy'], mets['AUC']
            list_machines.append(machine), list_folds.append(fold), list_acc.append(Acc), list_auc.append(AUC)
    data = {'machine': list_machines, 'fold' : list_folds , 'Accuracy' : list_acc, 'AUC': list_auc}
    result_data = pd.DataFrame(data)
    return result_data

def plot_results(data):
    fig, axs = plt.subplots(nrows=2, ncols=1,figsize=(15,10))

    ax = sns.swarmplot(ax = axs[0] , data = data , x="machine", y="AUC", hue = 'fold')
    ax = sns.swarmplot(ax = axs[1] , data = data , x="machine", y="Accuracy", hue = 'fold')

    fig.show()
    
    
def unscaler(dataset, scalerx):
    df = dataset.iloc[:,1:]
    df_unscaled = scalerx.inverse_transform(df)
    dataset_unscaled = pd.DataFrame(df_unscaled , columns = df.columns,index = df.index)
    df_result = pd.concat([dataset.iloc[:,0:1],dataset_unscaled], axis = 1)
    return df_result


def k_folds_indexs(data, folds = 5, balanced = False, validation_additional_false = 500, validation_reduce_false = 700 ):

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
    index = data[data.target.isin([0,1])].index
    array_index = np.array(index)
    np.random.shuffle(array_index)
    list_index = list(array_index)
    result_split = list(split(list_index, folds))
    
    result_validation_indexs = result_split
    
    if balanced:
        result_validation_indexs = list()
        for fold in result_split:
            temp_val = data[data.index.isin(fold)]
            temp_val =  balance_validation(temp_val, validation_additional_false )
            array = list(temp_val.index)
            result_validation_indexs.append(array)
    
    result_train_indexs = list()
    for fold in result_validation_indexs:
        train_data_raw = data[~data.index.isin(fold)]
        if validation_reduce_false:
            indexs = np.array(train_data_raw[train_data_raw.target.isin([0])].index)
            np.random.shuffle(indexs)
            indexs_selected = list(indexs)[0:validation_reduce_false]
        train_data_indexs = list(train_data_raw[~train_data_raw.index.isin(indexs_selected)].index)
        result_train_indexs.append(train_data_indexs)
            
    data_indexs = dict()
    for fold in range(1, folds+1):
        data_indexs[f'fold {fold}'] = { 'train index': result_train_indexs[fold-1], 'val index': result_validation_indexs[fold-1], }
    
    return data_indexs

def model_fitting_kfold(models, indexes_kfolds, features, train_data, rate_aug, sample_aug, save_model = False, save_nro_machine = 1 ):
    indexm = 1
    model_results = dict()
    model_saved = list()
    for model,rate_param, sample_param in zip(models,rate_aug, sample_aug) :
        fold_result = dict()
        fold_i = 1
        for fold in indexes_kfolds.keys():
            
            train_index, val_index = indexes_kfolds[fold]['train index'] , indexes_kfolds[fold]['val index']

            validation_data = train_data[train_data.index.isin(val_index)]
            train_data_tomodel = train_data[train_data.index.isin(train_index)]
        
            train_data_tomodel_aug = augmentation_selection_rates(train_data_tomodel, rate = rate_param, sample = sample_param)

            X_train = train_data_tomodel_aug[features]
            Y_train = train_data_tomodel_aug['target']

            X_val = validation_data[features]
            Y_val = validation_data['target']
            
            #### MODEL TRAINING:
            my_model = model.fit(X_train, Y_train)
            if save_model and indexm == save_nro_machine:
                model_saved.append(my_model)
                
            result_metrics, _, _ = metrics_train_validation(my_model, X_train, Y_train, X_val, Y_val)
            
            fold_result[f'fold-{fold_i}'] = {'metrics': result_metrics}
            #print(f'fold {fold_i} done')
            fold_i = fold_i + 1
            
        model_results[f'model-{indexm}'] = fold_result
        print(f'done machine {indexm}')
        indexm = indexm + 1
    return model_results, model_saved

def weighting_models(models, features, data, trained_model = None):
    X_train = data[data.target.isin([0,1])][features]
    Y_train = data[data.target.isin([0,1])]['target']
    probas_result = dict()
    i = 1
    for model in models:    
        probas = model.predict_proba(X_train)[:,1]
        probas_result[i] = probas
    probas_result['target'] = Y_train
    result_df = pd.DataFrame(probas_result)
    X_train = result_df.iloc[:,0:-1]
    Y_train = result_df['target']
    
    if trained_model:
        probas = trained_model.predict_proba(X_train)[:,1]
        return probas
    else:
        weight_model = LogisticRegression().fit(X_train, Y_train)
        return weight_model

def jsontotable(json, typex = 'Validation'):
    list_machines, list_folds, list_acc, list_auc = list(), list(), list(),list()
    
    for machine in json.keys():
        machine_json = json[machine]
        for fold in machine_json.keys():
            mets = machine_json[fold]['metrics'][typex]
            Acc, AUC = mets['Accuracy'], mets['AUC']
            list_machines.append(machine), list_folds.append(fold), list_acc.append(Acc), list_auc.append(AUC)
    data = {'machine': list_machines, 'fold' : list_folds , 'Accuracy' : list_acc, 'AUC': list_auc}
    result_data = pd.DataFrame(data)
    return result_data

def plot_results(data):
    fig, axs = plt.subplots(nrows=2, ncols=1,figsize=(15,10))

    ax = sns.swarmplot(ax = axs[0] , data = data , x="machine", y="AUC", hue = 'fold')
    ax = sns.swarmplot(ax = axs[1] , data = data , x="machine", y="Accuracy", hue = 'fold')

    fig.show()
    
def reconstructing_data(list_data, list_predictions, threshold = 0.15 ):
    list_data_result = list()
    for data,vector in zip(list_data, list_predictions):
        data['target'] = np.where(vector > threshold, 1,0)
        list_data_result.append(data)
    return pd.concat(list_data_result).sort_index()