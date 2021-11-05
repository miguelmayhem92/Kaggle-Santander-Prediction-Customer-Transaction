import pandas as pd
import numpy as np
import random
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import seaborn as sns; sns.set()

from sklearn.preprocessing import StandardScaler

def scaler(dataset):
    df  = dataset.iloc[:,2:]
    scaler = StandardScaler()
    scaler.fit(df)

    dataset_scaled = scaler.transform(df)
    dataset_scaled = pd.DataFrame(dataset_scaled, columns = df.columns)
    df_result = pd.concat([dataset.iloc[:,0:2],dataset_scaled], axis = 1)
    return df_result, scaler

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