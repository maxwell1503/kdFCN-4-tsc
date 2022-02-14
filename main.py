from utils.utils import create_directory
from utils.utils import read_dataset
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS, ARCHIVE_NAMES, ITERATIONS, ITERATIONS_STUDENT_ALONE, FILTERS, FILTERS2, ALPHALIST, TEMPERATURELIST, PATH_OUT, BEST_TEACHER_ONLY, LAYERS, SEPARABLE_CONV
from utils.utils import read_all_datasets

def callfit (tmp_output_directory, filters, filters2, alpha = None, temperature = None):
    output_directory = tmp_output_directory + dataset_name + '/'

    if os.path.exists(output_directory + 'DONE'):
        print('already computed')
    else:

        create_directory(output_directory)
        if alpha or temperature:
            fit_classifier(output_directory, filters, filters2, alpha, temperature)
        else:
            fit_classifier(output_directory, filters, filters2)    

        print('\t\t\t\tDONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')        

def fit_classifier(output_directory, filters, filters2, alpha=None, temperature=None):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    if alpha or temperature:
        create_classifier(classifier_name, x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2, alpha, temperature)
    else:
        create_classifier(classifier_name, x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2)    


def create_classifier(classifier_name, x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2, alpha = None, temperature = None):
    if classifier_name == 'teacher':
        from teacher import create_teacher
        return create_teacher(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, 128, 256)
    if classifier_name == 'StudentAlone':
        from StudentAlone import create_StudentAlone
        return create_StudentAlone(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2, layers=LAYERS, separable_conv=SEPARABLE_CONV)
    if classifier_name == 'Student':
        from Student import create_Student
        return create_Student(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2, alpha, temperature, layers=LAYERS, separable_conv=SEPARABLE_CONV)


############################################### main

# change this directory path in utils.constants for your machine
root_dir = PATH_OUT

for archive_name in ARCHIVE_NAMES:
    print('\tarchive_name', archive_name)

    datasets_dict = read_all_datasets()
    
    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:

            i = 0
            j = 0
            for filters in FILTERS:
                i = i + 1
                filters2 = FILTERS2[i-1]
                
                for classifier_name in CLASSIFIERS:
                    if classifier_name == 'teacher':
                        j = j + 1
                        if j >= 2 :
                           continue
                    print('classifier_name', classifier_name)   

                    if classifier_name == 'StudentAlone':
                        iterations = ITERATIONS_STUDENT_ALONE
                    else:
                        iterations = ITERATIONS
           
                    for iter in range(iterations):
                        print('\t\titer', iter)                        
                        trr = ''
                        if iter != 0:
                            trr = '_itr_' + str(iter)
                    
                        if classifier_name == 'teacher':
                            tmp_output_directory = root_dir + '/results/'  + classifier_name + '/' + archive_name + trr + '/'  
                            callfit (tmp_output_directory, filters, filters2)
                        elif classifier_name == 'StudentAlone' :
                            tmp_output_directory = root_dir + '/results/'  + '/results_f1_' + str(filters) + '_f2_' + str(filters2) + '/' + classifier_name + '/' + archive_name + trr + '/'
                            callfit (tmp_output_directory, filters, filters2)
                        else: #student
                            for alpha in ALPHALIST:
                              for temperature in TEMPERATURELIST:
                                  if BEST_TEACHER_ONLY:
                                    tmp_output_directory = root_dir + '/results/'  + '/results_f1_' + str(filters) + '_f2_' + str(filters2) + '/' + classifier_name + '/alpha' + str(alpha) + '/' + '/temperature' + str(temperature)+  '/' + archive_name + '_best_teacher' + trr + '/'
                                  else:   
                                    tmp_output_directory = root_dir + '/results/'  + '/results_f1_' + str(filters) + '_f2_' + str(filters2) + '/' + classifier_name + '/alpha' + str(alpha) + '/' + '/temperature' + str(temperature)+  '/' + archive_name + trr + '/'
                                  callfit (tmp_output_directory, filters, filters2, alpha, temperature)
                            if BEST_TEACHER_ONLY:
                                break #a single teacher                         


