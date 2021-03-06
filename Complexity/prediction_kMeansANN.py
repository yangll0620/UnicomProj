# -*- coding: utf-8 -*-
# @Author: yangll0620
# @Date:   2018-09-27 13:45:52
# @Last Modified by:   yangll0620
# @Last Modified time: 2018-09-27 15:43:00

from sklearn.cluster import KMeans
from loadData import data_load
import numpy as np
import time
import csv

from model_Dense3Layers import model_Dense3Layers_training, model_Dense3Layers_prediction
from evaluation import evaluate_prediction

def prediction_perpair(file_train, file_test):
# return precision, recall, TNR, F_measure, G_mean, time_train
	
	print("training data is ")
	print(file_train)
	print( "test data is  ")
	print(file_test)
	print("loading data.......")
	[x_train, y_train] = data_load(file_train)
	[x_test, y_test] = data_load(file_test)

	# use k-Means to select the two largest clusters
	time_trainstr = time.time()
	n_clusters  = 5
	print("...........Running kMeans...............")
	kmeans = KMeans(n_clusters=n_clusters, random_state = 0).fit(x_train)
	[labels_unique,counts_unique] = np.unique(kmeans.labels_,return_counts = True)
	ind = np.argsort(counts_unique)
	label_remain = labels_unique[ind[-2:]]
	tag = np.in1d(kmeans.labels_,label_remain)
	x_train_sub = x_train[tag,:]
	y_train_sub= y_train[tag]

	# ANN train
	model = model_Dense3Layers_training(x_train_sub, y_train_sub)
	time_trainend = time.time()
	time_train = time_trainend - time_trainstr
	y_pred = model_Dense3Layers_prediction(model, x_test)

	# evaluation
	[precision, recall, TNR, F_measure, G_mean] = evaluate_prediction(y_test, y_pred)

	del x_train, y_train, x_test, y_test
	return precision, recall, TNR, F_measure, G_mean, time_train


def prediction_kMeansANN():
	path_Dropbox = '/home/yll/Dropbox/'
	path_Proj = path_Dropbox + 'workSpace/SYSU/Projects/UnicomProject/Manuscripts/JournalJason/Complexity/'
	folder_load = path_Proj + 'Lidongyang/OriginalData/'
	file_07 =  folder_load + 'ZDJM_4G_XQ12_201507_T2_YUAN.csv'
	file_08 =  folder_load + 'ZDJM_4G_XQ12_201508_T2_YUAN.csv'
	file_09 =  folder_load + 'ZDJM_4G_XQ12_201509_T2_YUAN.csv'
	file_10 =  folder_load + 'ZDJM_4G_XQ12_201510_T2_YUAN.csv'

	path_save = path_Proj + 'Major Revision/Results/'
	savefile_name = 'results_kMeansANN.csv'
	with open(path_save+savefile_name,'w') as csv_file:
		fieldnames = ['type','precision','recall','TNR','F-measure','G-mean','train time']
		writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
		writer.writeheader() # write the header

		# 7 to 8
		[precision, recall, TNR, F_measure, G_mean, time_train] = prediction_perpair(file_07, file_08)
		writer.writerow({'type':'7 to 8','precision':precision,'recall':recall
			,'TNR':TNR,'F-measure':F_measure,'G-mean':G_mean,'train time':time_train})
		del precision, recall, TNR, F_measure, G_mean, time_train

		# 8 to 9
		[precision, recall, TNR, F_measure, G_mean, time_train] = prediction_perpair(file_08, file_09)
		writer.writerow({'type':'8 to 9','precision':precision,'recall':recall
			,'TNR':TNR,'F-measure':F_measure,'G-mean':G_mean,'train time':time_train})
		del precision, recall, TNR, F_measure, G_mean, time_train

		# 9 to 10
		[precision, recall, TNR, F_measure, G_mean, time_train] = prediction_perpair(file_09, file_10)
		writer.writerow({'type':'9 to 10','precision':precision,'recall':recall
			,'TNR':TNR,'F-measure':F_measure,'G-mean':G_mean,'train time':time_train})
		del precision, recall, TNR, F_measure, G_mean, time_train

if __name__ == "__main__":
	prediction_kMeansANN()