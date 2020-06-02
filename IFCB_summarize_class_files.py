# -*- coding: utf-8 -*-
"""
Originally created on Tue Feb 17 2015
This script has been repurposed to provide summary counts from class files. May 2020

This script will grab the biovolume feature data from extracted feature files 
for all images in an automated class file.
Can bin data by category or leave each image separate.
@author: Darren Henrichs
"""

# script to extract the biovolume estimates from IFCB V2 feature files
# and sum them per category for class files
# this will read a directory of class files and search the feature path for
# those files, pulling the biovolume from them

# 06/13/2017 DWH
# this script is a modified version of the biovolume grabber script
# this script will take the biovolume value for each cell, convert it to
# units of carbon following formulas from Menden_Deuer and Lessard 2000
# then sum the total carbon per category

from scipy.io import loadmat
import os
import pandas as pd
import numpy as np
from datetime import datetime

__author__ = 'dhenrichs'

# path to the feature files
#feature_path = '/data4/processed_data/extracted_features/2014/'
#feature_path = '/data4/Cruise_data/HRR_cruise/processed_data/extracted_features/'

# path to the class files for binning the biovolumes into categories
#class_path = '/data4/manual_classify_results/temp_alyssa_manual/' #processed_data/class_files/2014/'
#class_path = '/data4/Cruise_data/HRR_cruise/manual_corrections_cruise_data_31Jan2019/'
#class_path = '/data4/Cruise_data/HRR_cruise/class_files_cruise_data_CNN/'

# path to where the outfiles with biovolume will be located
#outpath = '/data4/test_biovolume/'

# limit files to one particular month/day   IMPORTANT: if you don't want to limit the date, just put None
date_limiter = None # a string  (e.g. 'D20170404') or None (you literally have to type None)

#are you using automated class files or manually corrected mat files?
#automated_or_manual = 'automated'  #can be 'automated' or 'manual'

#are these CNN results
#CNN = True


def grab_biovolume(in_feature, in_class, automated):
    '''this function is designed to return the total sum of carbon per category.
       this will NOT return the carbon for each image'''
    feature_data = load_feature_file(in_feature)
    if automated == 'automated':
        feature_data['Biovolume'] = convert_biovolume_pixels_to_microns(feature_data['Biovolume'])
        category_list, class_data = load_class_file_automated(in_class)
        if 'unclassified' not in category_list:
            category_list.append('unclassified')
        outdata = pd.DataFrame([0]*len(category_list), index=category_list, columns=['Biovolume']).T
        for image_cat, feat_size in zip(class_data, feature_data['Biovolume']):
            carbon_value = calculate_carbon_from_biovolume(feat_size, image_cat)
            outdata[image_cat] += carbon_value
        return outdata.T
 
    elif automated == 'manual':
        category_list, class_data, roinums = load_class_file_manual(in_class)
        converted_data = pd.DataFrame(index=roinums)
        converted_data['Category'] = class_data
        converted_data = converted_data.dropna()
        b = list(map(lambda x: category_list[int(x-1)], converted_data['Category']))
        converted_data['Category'] = b
        outdata = pd.DataFrame([0.]*len(category_list), index=category_list, columns=['Biovolume'])
        converted_data['Biovolume'] = convert_biovolume_pixels_to_microns(feature_data['Biovolume'])
        skipped_imgs = 0
        for image_cat, feat_size in zip(class_data, converted_data['Biovolume']):
            try:
        #if not np.isnan(image_cat):
                carbon_value = calculate_carbon_from_biovolume(feat_size, category_list[int(image_cat)])
                outdata.T[category_list[int(image_cat)-1]] += carbon_value
            #print "after", outdata.T[category_list[int(image_cat)-1]]
            #print 'CARBON:',carbon_value
        #print 'FEAT_SIZE:', feat_size
            except:
        #print 'Error occurred, skipping image:', image_cat
                skipped_imgs += 1
        #raise
        print('skipped_images:', skipped_imgs, end = ' ') 
        
        #now get the image counts for the categories
        counts = {cat:class_data.count(cat) for cat in sorted(category_list)}
        
        return outdata, counts
        #else:
        #    return None

def grab_class_counts(in_class, automated, main_category_list=None):
    
    if automated == 'automated':
        category_list, class_data = load_class_file_automated(in_class)
    else:
        category_list, class_data, roinums = load_class_file_manual(in_class)
        
    #now get the image counts for the categories
    if main_category_list:
        counts = {cat:class_data.count(cat) for cat in main_category_list}
    else:
        counts = {cat:class_data.count(cat) for cat in category_list}
    return counts
        
def convert_biovolume_pixels_to_microns(in_value):
    '''The biovolume values given from the IFCB data processing
        are in pixel units. Need to convert pixels to microns.
        Will use a calculated value from a beads file.'''
    conversion = 0.2 #this is the number of microns per pixel; this value calculated from 6um beads on IFCB130
    new_value = in_value * (conversion**2)  #this assumes the incoming value is biovolume 
    return new_value

def calculate_carbon_from_biovolume(invalue, category):
    """Calculate the cellular carbon from the given biovolume value based on  
       what category the image is assigned and how large it is. Conversion 
       formulas are from Table 4 in Menden-Deuer and Lessard (2000).
       
       inputs:
            invalue (float) = the biovolume value from the features file converted to microns

            category (str) = the category to which the image was assigned 
    
       returns:
            carbon_value (float) = the carbon calculated from the formulas
    """
    diatoms = ['Asterionellopsis', 'Centric', 'Ch_simplex', 'Chaetoceros', 'Corethron', 'Cylindrotheca',
               'Cymatosira', 'DactFragCeratul', 'Ditlyum', 'Eucampia', 'Eucampiacornuta', 'Guinardia',
               'Hemiaulus', 'Leptocylindrus', 'Licmophora', 'Melosira', 'Odontella', 'Pleurosigma', 'Pseudonitzschia',
               'Rhizosolenia', 'Skeletonema', 'Thalassionema', 'Thalassiosira', 'centric10', 'pennate', ]

    if category in diatoms:
        if invalue > 3000.: # diatoms > 3000 cubic microns (um**3)
            carbon_value = (10**(-0.933)) * (invalue ** 0.881)
        else:
            carbon_value = (10**(-0.541)) * (invalue ** 0.811)
    else:
        if invalue < 3000.: # protist plankton < 3000 cubic microns (um**3)
            carbon_value = (10**(-0.583)) * (invalue ** 0.860)
        else:
            carbon_value = (10**(-0.665)) * (invalue ** 0.939)

    return carbon_value


def load_class_file_automated(in_class):
    """Load the automated classifier results and list of class names.
    Returns:
            category_list = list of category names
            class_data = list classifications for each roi image
    """
    f = loadmat(class_path + in_class)
    category_list = f['class2useTB']
    class_data = f['TBclass_above_threshold'] #use this line for automated classifier results; can be 'TBclass_above_optthresh' if available
    
    if CNN:
        class_data = [category[0] for category in class_data[0]] #un-nest the MATLAB stuff #use this line for automated classifier results
        category_list = [category[0] for category in category_list[0]]
    else: #deal with the Python to MATLAB weirdness
        class_data = [category[0][0] for category in class_data] #un-nest the MATLAB stuff #use this line for automated classifier results
        category_list = [category[0][0] for category in category_list] #un-nest the MATLAB stuff
    return category_list, class_data


def load_class_file_manual(in_class):
    """Load the manually classified results and list of class names.
    Returns:
            category_list = list of category names
            class_data = list classifications for each roi image
            roinums = list of roi numbers 
    """
    #the structure of the mat file variable with the classes is slightly different in manual files
    #classlist is a table of shape (num_rois x 3) with the columns being: roinum, manual category, automated category
    f = loadmat(class_path + in_class)
    roinums = None
    class_data_manual = f['classlist']
    class_data = f['classlist'][:,2]
    roinums = f['classlist'][:,0]
    for index, value in enumerate(class_data):
        if not np.isnan(class_data_manual[index, 1]):
            class_data[index] = class_data_manual[index,1]
    category_list = f['class2use_manual']
    try:
        category_list = [category[0] for category in category_list[0]] #this works with some of the files
    except:
        category_list = [category[0] if len(category) > 0 else '' for category in category_list[0]] #this works with the others
    return category_list, class_data, roinums


def load_feature_file(in_feature):
    f = pd.read_csv(feature_path + in_feature, index_col=0)
    return f


if __name__ == '__main__':
    
    #read in config file
    with open('./IFCB_summary.cfg') as f:
    
        config_in = {line[:-1].replace(' ', '').split('=')[0]: line[:-1].replace(' ', '').split('=')[1] 
                 for line in f if len(line) > 1 if line[0] != '#'}

    feature_path = config_in['path_to_features']
    class_path = config_in['path_to_class_files']
    outpath = config_in['path_to_place_output_files']
    automated_or_manual = config_in['automated_or_manual']
    
    #have to convert the string into a boolean
    if config_in['CNN'] == 'True':
        CNN = True
    else:
        CNN = False
    
    if feature_path[-1] != '/':
        feature_path += '/'
            
    if class_path[-1] != '/':
        class_path += '/'
    
    if outpath[-1] != '/':
        outpath += '/'
        
    
    # grab the list of files from each directory
    list_of_feature_files = os.listdir(feature_path)
    list_of_class_files = os.listdir(class_path)
    print("Feature files: {}".format(len(list_of_feature_files)))
    print("Class files  : {}".format(len(list_of_class_files)))
    
    #create the output dataframe for counts summary
    #first get the category list
    
                    
    temp = grab_class_counts(list_of_class_files[0], automated_or_manual)
    category_list = sorted(temp.keys())
    #now use the index from the first class file year to start the master list of counts
    all_counts = pd.DataFrame(index=category_list)
    all_biovolumes = pd.DataFrame(index=category_list)
    
    # start working through the class files individually
    for class_index, indiv_file in enumerate(list_of_class_files):
        if indiv_file[-3:] == 'mat':
            if not date_limiter or date_limiter == indiv_file[:len(date_limiter)]:
                print("Processing {}...".format(indiv_file), end=' ')
                
                features_found = True
               # try:
                if 1:
                    feature_index = 0
                    while list_of_feature_files[feature_index][:21] != indiv_file[:21]:
                        feature_index += 1
                        if feature_index >= len(list_of_feature_files)-1:
                            #raise ValueError("The feature file was not found") #this will error out and stop the program
                            print("feature file not found.")
                            features_found = False
                            print(list_of_feature_files[feature_index][:21], indiv_file[:21] )
                            continue
                    if features_found:
                        temp_biovolumes = grab_biovolume(list_of_feature_files.pop(feature_index), list_of_class_files[class_index], automated_or_manual)
                        #temp_biovolumes.to_csv(outpath + indiv_file[:-3] + 'csv')
                        
                        print("done!")
                    temp_counts = grab_class_counts(list_of_class_files[class_index], automated_or_manual, category_list)
                    temp_timestamp = list_of_class_files[class_index]
                    if temp_timestamp[0] == 'D':
                        timestamp = datetime.strptime(temp_timestamp[:16], 'D%Y%m%dT%H%M%S')
                    else:
                        timestamp = datetime.strptime(temp_timestamp[:21], 'IFCB3_%Y_%j_%H%M%S')
                    
                    temp_counts_df = pd.DataFrame.from_dict(temp_counts, orient='index')
                    all_counts[timestamp] = temp_counts_df
                    all_biovolumes[timestamp] = temp_biovolumes.Biovolume
                    
                    
                #except:
                 #   print "something went wrong."
                    #break
                    #while list_of_feature_files[feature_index][:21] != indiv_file[:21]:
                    #    feature_index += 1
                    #    if feature_index >= len(list_of_feature_files)-1:
                    #        #raise ValueError("The feature file was not found") #this will error out and stop the program
                    #        print "feature file not found."; print list_of_feature_files[feature_index][:21], indiv_file[:21] 
                    #        features_found = False
                    #if features_found:
                    #    temp_biovolumes = grab_biovolume(list_of_feature_files.pop(feature_index), list_of_class_files[class_index], automated_or_manual)
                    #    temp_biovolumes.to_csv(outpath + indiv_file[:-3] + 'csv')
                    #    print "done!"
        else:
            continue
    
    #now output the final files
    all_counts.T.to_csv(outpath+'summary_counts.csv')
    all_biovolumes.T.to_csv(outpath+'summary_biovolumes.csv')
