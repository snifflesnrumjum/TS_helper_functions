#script to load the new style RF results into python for analysis

import scipy.io
import pandas
from matplotlib.dates import num2date
from datetime import timedelta
#from datetime import datetime
import numpy as np
#import copy

#load the config file; presume it's in the same directory as this script
#with open('D:/Python27/TestProgs/IFCB_analysis/IFCB_analysis_config.cfg') as f:
with open('./IFCB_analysis_config.cfg') as f:
    
    config_in = {line[:-1].replace(' ', '').split('=')[0]: line[:-1].replace(' ', '').split('=')[1] 
                 for line in f if len(line) > 1 if line[0] != '#'}

path_to_summary_files = config_in['summary_directory']
summary_file_prefix = config_in['summary_prefix']
summary_file_years = list(config_in['years'].split(','))
path_to_ml_files = config_in['ml_analyzed_directory']
ml_analyzed_ifcb3_prefix = config_in['ml_analyzed_prefix']

#convert string to boolean
if config_in['bin_files'] == 'True':
    bin_files = True
else:
    bin_files = False
    
bin_frequency = config_in['bin_frequency']

#have to convert the string into a boolean
if config_in['CNN'] == 'True':
    CNN = True
else:
    CNN = False

outfile_directory = config_in['outfile_directory']

if path_to_summary_files[-1] != '/':
    path_to_summary_files += '/'
        
if path_to_ml_files[-1] != '/':
    path_to_ml_files += '/'

if outfile_directory[-1] != '/':
    outfile_directory += '/'

def what_month_is_it(year, julian_day):
    if year % 4 == 0:
        month_ranges = [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    else:
        month_ranges = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    month_to_return = 0
    while julian_day > month_ranges[month_to_return]:
        month_to_return += 1

    #adding 1 to the return value for non zero index used in datetime module for the month
    return month_to_return + 1

def get_manual_data(manual_dates, raw_data):
    #this will grab the raw data for the manual files provided
    #global merged_dates, mls_analyzed
    data_return = []
    mls_return = []
    for matdate in range(len(merged_dates)):
        if merged_dates[matdate] in manual_dates:
            data_return.append(raw_data.T[matdate])
            mls_return.append(mls_analyzed[matdate])
    return [np.array(data_return), np.array(mls_return)]

    
#load the raw data files
print "Loading ml_analyzed data...",
ifcb3_ml_analyzed = []
for ml_analy in summary_file_years:
    print ml_analy,
    ifcb3_ml_analyzed.append(scipy.io.loadmat(path_to_ml_files + ml_analyzed_ifcb3_prefix + ml_analy + '.mat'))
print
raw_summary = []
######use this section to load individual years of summary files
print "Loading summary data...",

for summary in summary_file_years:
    print summary,
    raw_summary.append(scipy.io.loadmat(path_to_summary_files + summary_file_prefix + summary + '.mat'))
#######
print

print "loaded!"
#end of data loading

#find the list of dates in both datasets
print "Creating merged lists..."
merged_date_list = []
merged_indexes = []
all_mls = []
all_dates = []
for indiv_file in ifcb3_ml_analyzed:
    
    all_mls.extend(list(indiv_file['ml_analyzed'][0]))
    all_dates.extend(list(indiv_file['time'][0]))

            
number_mls_files_removed = 0
missing_matdates = []
missing_matdates_raw_summary_1 = [rawsum['mdateTB'][0] for rawsum in raw_summary]
missing_matdates_raw_summary = []

for x in range(len(missing_matdates_raw_summary_1)):
    missing_matdates_raw_summary.extend(list(missing_matdates_raw_summary_1[x]))
last_good_index = 0

for x in range(len(all_dates)):
    keep=False
    low_mls = False
    year_index = []
    for year in range(len(raw_summary)):
        if all_dates[x] in raw_summary[year]['mdateTB']:
            if 0.25 < all_mls[x] < 5.5:
                keep = True
                year_index = [year, np.where(raw_summary[year]['mdateTB'] == all_dates[x])[0][0]]
                last_good_index = year_index[1]
            else:
                number_mls_files_removed += 1
                low_mls = True
            break
    if keep:
        merged_date_list.append(all_dates[x])
        merged_indexes.append([x, year_index])
    else:
        if not low_mls:
            missing_matdates.append(all_dates[x])
           
missing_matdates_raw_summary.sort()

print "Grabbed data indexes..."
        
filenumber = 1
rf_classcountTB = []
rf_classcountTB_optthresh = []

mls_analyzed = []
for x in merged_indexes:
    if filenumber % 5000 == 0:
        print filenumber,
    
    rf_classcountTB.append(raw_summary[x[1][0]]['classcountTB'][x[1][1]])
    rf_classcountTB_optthresh.append(raw_summary[x[1][0]]['classcountTB_above_optthresh'][x[1][1]])
    
    if not np.isnan(all_mls[x[0]]):
        mls_analyzed.append(all_mls[x[0]])
    else:
        try:
            if not np.isnan(raw_summary[x[1][0]]['ml_analyzedTB'][x[1][1]]):
                mls_analyzed.append(raw_summary[x[1][0]]['ml_analyzedTB'][x[1][1]])
            else:
                mls_analyzed.append(np.nan)
        except:
            print "didn't get ", x
    filenumber += 1

rf_classcountTB = np.array(rf_classcountTB).T
rf_classcountTB_optthresh = np.array(rf_classcountTB_optthresh).T


rf_classcounts = []
rf_classcounts.append(rf_classcountTB / mls_analyzed)
rf_classcounts.append([])
rf_classcounts.append(rf_classcountTB_optthresh / mls_analyzed)
merged_dates = merged_date_list

#finished with loading and appending the data 
print "done!"


#####
#####
#making the time series variables
print "Binning the data...",
binned_counts = []
binned_concentrations = []
binned_counts_opt = []
binned_concentrations_opt = []

binned_dates = []
for x in range(len(merged_dates)):
    binned_dates.append(num2date(merged_dates[x]) - timedelta(days=366))

categories = []
if CNN:
    for x in range(len(raw_summary[0]['class2useTB'][0])):
        categories.append(str(raw_summary[0]['class2useTB'][0][x][0]))
else:    
    for x in range(len(raw_summary[0]['class2useTB'])):
        categories.append(str(raw_summary[0]['class2useTB'][x][0][0]))
    

temp_binned_mls = pandas.Series(mls_analyzed, index = binned_dates)
temp_counts = pandas.DataFrame(rf_classcountTB, index = categories, columns = binned_dates).T
temp_counts_optthresh = pandas.DataFrame(rf_classcountTB_optthresh, index = categories, columns = binned_dates).T
binned_mls = temp_binned_mls.resample('1h').sum().tz_convert('UTC')  #this line groups the data by index [in this case time], then sums the data from the repeated indexes, then resamples the data to hourly [filling in gaps], then gives it a timezone so we can use this in later calculations
binned_mls[binned_mls < 1] = np.nan
for x in range(len(categories)):
    
    binned_counts.append(temp_counts[categories[x]].resample('1h').sum().tz_convert('UTC'))
    binned_concentrations.append(temp_counts[categories[x]].resample('1h').sum().tz_convert('UTC') / binned_mls)
    binned_counts_opt.append(temp_counts_optthresh[categories[x]].resample('1h').sum().tz_convert('UTC'))
    binned_concentrations_opt.append(temp_counts_optthresh[categories[x]].resample('1h').sum().tz_convert('UTC') / binned_mls)

#unbinned_concentrations = temp_counts.T / temp_binned_mls 
#unbinned_concentrations = unbinned_concentrations.T
    
binned_counts = pandas.DataFrame(binned_counts, index = categories).T
binned_concentrations = pandas.DataFrame(binned_concentrations, index = categories).T
binned_counts_opt = pandas.DataFrame(binned_counts_opt, index = categories).T
binned_concentrations_opt = pandas.DataFrame(binned_concentrations_opt, index = categories).T

binned_concentrations.resample(bin_frequency).sum().to_csv(outfile_directory+'IFCB_binned_concentrations.csv')
binned_concentrations_opt.resample(bin_frequency).sum().to_csv(outfile_directory+'IFCB_binned_concentrations_opt_thresh.csv')


print "Done!"

