# -*- coding: utf-8 -*-
# @Author: yll
# @Date:   2018-09-12 09:43:49
# @Last Modified by:   yll
# @Last Modified time: 2018-09-13 11:22:33

import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def data_load(file_name):
	df = pandas.read_csv(file_name, names=None)
	values = df.values
	label = values[:,-1] # the last column is the label 
	del df

	ind_fea = list(range(4,5))
	ind_fea.extend(list(range(6,40)))
	ind_fea.extend(list(range(42,51)))
	fea = values[:,ind_fea] # fea: n_trials * n_features (44)
	del values

	# remove the trials with nan 
	ind_nan_label = np.where(np.isnan(label))[0]
	ind_nan_fea = np.where(np.isnan(fea))[0]
	ind_nan = np.concatenate((ind_nan_label, ind_nan_fea))
	ind_del = np.unique(ind_nan)
	label = np.delete(label, ind_del, 0)
	fea = np.delete(fea, ind_del, 0)

	# converts all values in the columns from 0 to 1
	mn = MinMaxScaler()
	fea = mn.fit_transform(fea)


	return fea, label
