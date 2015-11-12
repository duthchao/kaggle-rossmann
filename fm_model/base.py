import subprocess
import os
import pandas as pd

def run_cmd(cmd):
    print (cmd)
    process = subprocess.Popen(cmd, shell=True,
                       stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    for t, line in enumerate(iter(process.stdout.readline,'')):
        if line == b'':
            break
        line = line.rstrip()
        print (line.decode('utf-8'))
    process.communicate()
    return process.returncode

class FM(object):
    '''
----------------------------------------------------------------------------
libFM
  Version: 1.4.2
  Author:  Steffen Rendle, srendle@libfm.org
  WWW:     http://www.libfm.org/
This program comes with ABSOLUTELY NO WARRANTY; for details see license.txt.
This is free software, and you are welcome to redistribute it under certain
conditions; for details see license.txt.
----------------------------------------------------------------------------
-cache_size     cache size for data storage (only applicable if data is
                in binary format), default=infty
-dim            'k0,k1,k2': k0=use bias, k1=use 1-way interactions,
                k2=dim of 2-way interactions; default=1,1,8
-help           this screen
-init_stdev     stdev for initialization of 2-way factors; default=0.1
-iter           number of iterations; default=100
-learn_rate     learn_rate for SGD; default=0.1
-load_model     filename for reading the FM model
-meta           filename for meta information about data set
-method         learning method (SGD, SGDA, ALS, MCMC); default=MCMC
-out            filename for output
-regular        'r0,r1,r2' for SGD and ALS: r0=bias regularization,
                r1=1-way regularization, r2=2-way regularization
-relation       BS: filenames for the relations, default=''
-rlog           write measurements within iterations to a file;
                default=''
-save_model     filename for writing the FM model
-seed           integer value, default=None
-task           r=regression, c=binary classification [MANDATORY]
-test           filename for test data [MANDATORY]
-train          filename for training data [MANDATORY]
-validation     filename for validation data (only for SGDA)
-verbosity      how much infos to print; default=0

    '''
    def __init__(self,model='./libFM',task='c',train='train.txt',test='test.txt',dim='1,1,8',save_model='fm'):
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        self.model=os.path.join(curr_path,model)
        self.task=task
        self.train=train
        self.test=test
        self.dim=dim
        self.save_model = save_model
    def fit(self):
        cmd = ' '.join(['-'+item[0]+' '+item[1] for item in zip(['task','train','test','dim','save_model'],
                                                                [self.task,self.train,self.test,self.dim,self.save_model])])
        cmd = self.model+' '+cmd
        run_cmd(cmd)
        
    def predict(self,load_model='fm',test='X_pred.libfm',out='y_pred'):
        cmd = ' '.join(['-'+item[0]+' '+item[1] for item in zip(['task','train','test','dim','load_model','iter','out'],
                                                                [self.task,self.train,test,self.dim,load_model,'1',out])]) 
        cmd = self.model+' '+cmd
        run_cmd(cmd)
        
    def get_param(self,fm,num_features,k):
        '''
        get W V from saved fm model
        
        Parameters
        ----------
        fm : saved fm model file path
        
        num_features: fm model num_features
        
        k : fm model V 's dim
        
        return
        ---------
        W:W0,Wj
        
        V:Vj
        
        '''
        W = pd.read_csv(fm,encoding='utf8',sep=' ',names='W',skiprows=[0,2],nrows=num_features+1)
        V = pd.read_csv(fm,encoding='utf8',sep=' ',names=['v'+str(i) for i in range(k)],skiprows=range(num_features+4))
        return W,V