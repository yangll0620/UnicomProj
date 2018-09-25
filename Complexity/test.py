# -*- coding: utf-8 -*-
# @Author: yll
# @Date:   2018-09-14 11:34:38
# @Last Modified by:   yll
# @Last Modified time: 2018-09-14 11:39:50

import csv
savefile_name = 'test.csv'
csv_file = open(savefile_name,mode = 'w')

# write the header
try:
	fieldnames = ['type','precision','recall','TNR','F-measure','G-mean','train time']
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
	writer.writeheader()
finally:
	csv_file.close()	
# write 7 to 8
csv_file = open(savefile_name,mode = 'a')
try:
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
	writer.writerow({'type':'7 to 8','precision':1,'recall':2
		,'TNR':3,'F-measure':4,'G-mean':5,'train time':6})
finally:
	csv_file.close()

# write 
csv_file = open(savefile_name,mode = 'a')
try:
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
	writer.writerow({'type':'8 to 9','precision':7,'recall':8
		,'TNR':9,'F-measure':10,'G-mean':11,'train time':12})
finally:
	csv_file.close()