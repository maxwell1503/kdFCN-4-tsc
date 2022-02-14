# kdFCN-4-tsc

This is the companion repository for our paper entitled "A study of Knowledge Distillation in Fully Convolutional Network for Time Series Classification", submitted to [IJCNN 2022](https://wcci2022.org). 

## Datasets used

All our experiments were performed on the widely used time series classification benchmark [UCR/UEA archive](http://timeseriesclassification.com/index.php). 
We used the latest version (2018) and considered 112 datasets by discarding datasets containing series of unequal length or missing values, as well as the Fungi dataset, which only provides a single train case for each class label. The UCR archive is necesary to run the code.

## Code

The code is divided as follows:

* The [main.py](main.py) python file contains the necessary code to run the experiments.
* The [teacher.py](teacher.py) python file contains the teacher implementation.
* The [student.py](student.py) python file contains the student implementation.
* The [studentAlone.py](studentAlone.py) python file contains the studentAlone implementation.
* The [utils.py](utils/utils.py) python file contains the necessary functions to read the datasets.
* The [constants.py](utils/constants.py) python file contains the definition of various constants.
* The [copy_best_teacher.py](copy_best_teacher.py) python file contains the code to copy the results of the best teacher (in training)

### Run the experiments

To simply run the experiments, you can type:

``python3 main.py``

It will run the experiments according to the [constants.py](utils/constants.py) configuration

### Configuration

You can setup the configuration in the [constants.py](utils/constants.py) python file.
In particular, you may need to modify:
* PATH_DATA: path to the folder where the UCR Archive is
* PATH_OUT: path to the folder where results will be saved
* CLASSIFIERS: list of classifiers to consider among 'teacher', 'StudentAlone' and 'Student'
* BEST_TEACHER_ONLY: set to True if you want to consider only the best teacher (in training) in while distilling to the student
* ALPHALIST: list of different alpha hyper-parameter to consider
* TEMPERATURELIST: list of different temperature hyper-parameter to consider
* FILTERS: list of number of filters to consider in first and third layers of student architectures
* FILTERS2: list of number of filters to consider in second layer of student architectures
* LAYER: number of layers to consider in student architectures
* SEPARABLE_CONV: set to True if you want to use depthwise separable convolutions

### Steps
If you want to reproduce the experiments of the paper for a particular configuration:
1. Run the [main.py](main.py) script for 'teacher' and 'studentAlone' only
2. Run the [copy_best_teacher.py](copy_best_teacher.py) script to copy the best teacher models
3. Run again the [main.py](main.py) script for 'student' model
