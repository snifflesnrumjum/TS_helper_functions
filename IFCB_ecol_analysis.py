# this script provides helper functions/wrappers for Imaging FlowCytobot (IFCB) time series for further analysis
# coded for python2.7 
# originally coded by Darren Henrichs unless otherwise specified

import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import random

from skbio.stats.ordination import PCoA, CA
from skbio import DistanceMatrix

from sklearn import manifold
from sklearn.decomposition import PCA as PCA_sklearn



def run_PCA(in_data, num_dimen=2, distance_measure='braycurtis'):
    """Run a principal components analysis from a pandas dataframe 
       (e.g. species x dates). Will calculate the distance based on the given 
       distance metric.
       
       Input:
           in_data (df) = a pandas dataframe of several time series
           num_dimen (int) = number of dimensions in which to conduct the 
                       PCA (default: 2)
           distance_measure (str) = which distance metric should be used to 
                                    calculate the distance between groups 
                                    (default: 'braycurtis')
       Returns:
           pca (ndarray) = array of shape (num_dates x num_dimen)
    """
    dist = sp.spatial.distance.pdist(in_data, distance_measure)
    dist = sp.spatial.distance.squareform(dist)
    pca = PCA_sklearn(n_components=num_dimen).fit_transform(dist)
    return pca

def run_PCoA(in_data):
    dist = sp.spatial.distance.pdist(in_data, 'braycurtis')
    dist = sp.spatial.distance.squareform(dist)
    pcoa = PCoA(DistanceMatrix(dist))
    return pcoa.scores()

def normalize_cc(indata, ver=1):
    #this will normalize the indata so that when calculating a cross correlation using the np.correlate function
    #the output values will be between -1 and 1; checked with R and values match
    #input data should be a pandas series
    #use this on the inputs to np.correlate -> i.e. np.correlate(normalize_cc(in1), normalize_cc(in2))
    #to get the normalized (between -1,1) values after the np.correlate you must do the following:
    #cross_corr_result /= np.sqrt(np.dot(in1, in1) * np.dot(in2, in2))
    #to get all this in one function use matplotlib.pyplot.xcorr
    #however, the xcorr function requires both inputs to be the same length
    if ver ==1:
        return (indata - np.mean(indata)) / (np.std(indata) * len(indata))
    else:
        return (indata - np.mean(indata)) / np.std(indata)

def cross_corr(in1, in2, max_lag=None):
    #this function will take two input sequences and return the normalized cross correlation
    #9/21/2015 - input data should be in the right sampling freq (i.e. hourly, daily) and include all timepoints including NA
    #in1, in2 = align_indexes(in1, in2)
    #in1_norm = normalize_cc(in1)
    #in2_norm = normalize_cc(in2)
    if not max_lag:
        max_lag = len(in1.index) - 1
    #temp_cc = np.correlate(in1_norm, in2_norm, 2)  #the 2 here just means "full" mode
    #norm_cc = temp_cc / np.sqrt(np.dot(in1_norm, in1_norm) * np.dot(in2_norm, in2_norm)) #this line of code is based on the normalization in the code from the xcorr function
    #if len(norm_cc) % 2 == 0:
    #    xaxis_data = [0 - len(norm_cc)/2, len(norm_cc)/2]
    #else:
    #    xaxis_data = [0 - len(norm_cc)/2, len(norm_cc)/2 + 1]
    xaxis_data = range(-max_lag, max_lag+1, 1)
    #corr_values = [0] * len(xaxis_data)
    #for x, y in enumerate(xaxis_data):
    #    corr_values[x] = in1.corr(in2.shift(y), min_periods=len(in1.index)/10)
    corr_values = [in1.corr(in2.shift(y), min_periods=len(in1.index)/10) for y in xaxis_data]
    return xaxis_data, pd.Series(corr_values, index=xaxis_data)

def align_indexes(in1, in2):
    #this function will take both indexes and make the aligned to each other
    in2_temp = in2.reindex(in1.dropna().index)
    in1_temp = in1.reindex(in2_temp.dropna().index)
    in2_temp = in2_temp.reindex(in1_temp.dropna().index)
    print in1_temp.index
    print in2_temp.index
    return in1_temp, in2_temp


def permutation_test(in1, in2, permutations=999):
    #this will calculate the pandas.corr function; then permute the values of one variable and recalculate the pandas.corr to determine significance
    #expects two pandas frames
    initial_corr = in1.corr(in2)
    num_success = 1
    in2_copy = in2.copy()
    for x in range(permutations):
        random.shuffle(in2_copy)
        temp_corr = in1.corr(in2_copy)
        if abs(temp_corr) >= abs(initial_corr): #a one-sided test
            num_success += 1
    return initial_corr, (num_success / float(permutations + 1))
    


def rotate_current_velocity(u, v, theta):
    #this will take the u, v current velocity values and rotate them theta degrees
    #this is from the book: Data Analysis Methods in Physical Oceanography eqs. 5.50b
    theta_rads = math.radians(theta)    
    u_prime = (u * np.cos(theta_rads)) + (v * np.sin(theta_rads))
    v_prime = (-u * np.sin(theta_rads)) + (v * np.cos(theta_rads))
    return u_prime, v_prime

def run_NMDS(indata, num_dimen=5, max_iter=100, metric=False, n_init=10, n_jobs=1, init=None, eps=1e-55, random_state=None):
    #this function will run an NMDS analysis
    #input should be a pandas dataframe of data, already transformed/standardized
    #requires a precomputed distance matrix as input so I'll just do it here
    #will return the NMDS results and a list of index names
    dist = sp.spatial.distance.pdist(indata, 'braycurtis')
    dist = sp.spatial.distance.squareform(dist)
    nmds = manifold.MDS(n_components=num_dimen, metric=metric, max_iter=max_iter, eps=eps, dissimilarity='precomputed', n_jobs=n_jobs, verbose=1, n_init=n_init, random_state=random_state)
    result = nmds.fit_transform(dist, init=init) #run the nmds analysis
    #result *= np.sqrt((dist**2).sum()) / np.sqrt((result**2).sum()) #rescale the data
    result_rotated = PCA_sklearn(n_components=num_dimen).fit_transform(result)
    sample_names = []
    for name in indata.index.ravel():
        sample_names.append(str(name)[:-19])
    print "Final stress:", nmds.stress_
    return result_rotated, sample_names, result, nmds

def plot_NMDS(indata, label_list=None, s=40, c='b', alpha=1.0):
    #indata should be a np.array(output) from run_NMDS function
    #if labels are desired then send a list of the labels as label_list
    plt.scatter(indata[:, 0], indata[:, 1], s=s, c=c, alpha=alpha)
    if label_list:
        for name_loc, name in enumerate(label_list):
            plt.text(indata[name_loc,0], indata[name_loc,1], name)
    

def plot_env_vars(inenv, innmds, corr_value=0.10, scale=1, c='k', label_arrows=False, single_var=False, plot=True):
    #will take a pandas dataframe as input for inenv (usually the envir_data), and an np.array for innmds (usually from the NMDS analysis)
    #will assume they are in the same frequency (e.g. daily, monthly, etc.)
    complete = inenv.join(pd.DataFrame(innmds[:, :2], columns=['NMDS1', 'NMDS2'], index=inenv.index))
    results = []
    if not single_var:
        for indiv_var in complete.keys():
            if complete.NMDS1.corr(complete[indiv_var]) > corr_value or complete.NMDS1.corr(complete[indiv_var]) < -corr_value:
                results.append([0,0,complete.NMDS1.corr(complete[indiv_var]), complete.NMDS2.corr(complete[indiv_var]), indiv_var])
            elif complete.NMDS2.corr(complete[indiv_var]) > corr_value or complete.NMDS2.corr(complete[indiv_var]) < -corr_value:
                results.append([0,0,complete.NMDS1.corr(complete[indiv_var]), complete.NMDS2.corr(complete[indiv_var]), indiv_var])
        results = results[:-2]
    else:
        results.append([0,0,complete.NMDS1.corr(complete[single_var]), complete.NMDS2.corr(complete[single_var]), single_var])
    X, Y, U, V, names = zip(*results)
    if plot:
        plt.quiver(X, Y, U, V, scale=scale, scale_units='xy', angles='xy', color=c, edgecolor='k')
    if plot and label_arrows:
        for indiv_index, indiv_arrow in enumerate(names):
            plt.text(U[indiv_index], V[indiv_index], indiv_arrow)
    return results

def plot_cluster_NMDS(indata, cluster_data, label_list=None, s=40, scale_size = False, **kwargs):
    #indata should be a np.array(output) from run_NMDS function
    #if labels are desired then send a list of the labels as label_list
    color_library = {0: 'b',
                     1: 'g',
                     2: 'r',
                     3: 'm',
                     4: 'k',
                     5: 'c',
                     6: 'y',
                     7: 'w',
                     8: 'darkblue',
                     9: 'lime',
                     10: 'darkred',
                     11: 'darkmagenta',
                     12: 'lightgrey',
                     13: 'darkcyan',
                     14: 'darkyellow',
                     15: 'beige'}
    
    for datapt in range(indata.shape[0]):
        if not scale_size:
            plt.scatter(indata[datapt, 0], indata[datapt, 1], s=s, c=color_library[cluster_data[datapt]], **kwargs)
        else:
            pt_size = s[datapt]
            plt.scatter(indata[datapt, 0], indata[datapt, 1], s=pt_size, c=color_library[cluster_data[datapt]], **kwargs)
    if label_list:
        for name_loc, name in enumerate(label_list):
            plt.text(indata[name_loc,0], indata[name_loc,1], name)
 
def partial_corr(in_data, cat1, cat2):
    #in_data should be a pandas dataframe with all the factors to analyze included and those that are not wanted already removed
    #cat1, cat2 are the two categories to be analyzed: cat1 is the target/response variable, cat2 is the category to correlate with cat1
    #this function uses the matrix method shown in Chp 4 of the Numerical Ecology book; pgs 175-178
    new_cols = list(in_data.columns)
    new_cols.pop(new_cols.index(cat1))
    new_cols.pop(new_cols.index(cat2))
    temp = pd.DataFrame(in_data, columns = [cat1, cat2] + new_cols)
    corr_matrix = np.matrix(temp.corr())
    mat_cond_corr = corr_matrix[:2, :2] - (corr_matrix[:2, 2:] * corr_matrix[2:, 2:]**-1 * corr_matrix[2:, :2])
    diag_sqrt = np.diag((mat_cond_corr.diagonal().A**-.5)[0])
    big_r = diag_sqrt * mat_cond_corr * diag_sqrt
    return big_r
    

def calculate_monthly_anomaly(in_data):
    """Calculate the anomaly values for monthly data
    Inputs:
        in_data (Series): the time series of data to be used for anomalies
        
    Returns:
        out_data (Series): the time series of data but subtracted from the monthly mean 
    """
    annual_data = in_data.groupby(in_data.index.month)
    out_data = in_data.copy()
    for x in range(len(in_data)):
        mean_data = annual_data.get_group(in_data.index.month[x]).mean()
        out_data[x] = in_data[x] - mean_data
    return out_data
    
