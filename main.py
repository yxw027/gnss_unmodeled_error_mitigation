from time import time


from config.configutil import getpath
from preproc.preprocessor import  countprefile
from proc.cnn_processor import cnn_predict, cnn_train


if __name__ == '__main__':
    sts = ['alic', 'bake', 'bjfs', 'meli', 'penc', 'picl', 'pova',  'sch2','nril', 'tixi', 'yakt', 'hkws']
    for st in sts:
        tstart = time()
        fnum = countprefile(getpath('preprocpath', st))#countprefile: get the total number of training data
        train_range = range(fnum - 10, fnum - 2)
        test_range = range(fnum - 2, fnum)
        cnn_train(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range)#getpath: get data directory
        cnn_predict(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range, test_range, True)
