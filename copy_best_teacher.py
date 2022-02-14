import os
import numpy as np
import pandas as pd
from distutils.dir_util import copy_tree

root_path = 'results/'
path_teacher = root_path + 'teacher/'

for d in os.listdir(path_teacher + 'UCRArchive_2018/'):
	if d[0]=='.':
		continue

	teacher_acc=[]

	for i in range(5):
		if i==0:
			dir_ucr = 'UCRArchive_2018/'
		else:
			dir_ucr = 'UCRArchive_2018_itr_' + str(i) + '/'

		df = pd.read_csv(path_teacher + dir_ucr + d + '/historyteacher.csv')
		df = df.sort_values('loss')
		teacher_acc.append(df['categorical_accuracy'].iloc[0])

	maxi = max(teacher_acc)
	index_maxi = teacher_acc.index(max(teacher_acc))
	if index_maxi==0:
		best_teacher = 'UCRArchive_2018/'
	else:
		best_teacher = 'UCRArchive_2018_itr_' + str(index_maxi) + '/'
	path_in = path_teacher + best_teacher + d
	path_out = path_teacher + 'UCRArchive_2018_best_teacher/' + d
	copy_tree(path_in,path_out)