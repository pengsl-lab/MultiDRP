# MATDRP
MATDRP is a Multi-view Attention-based Deep Learning Framework for Anticancer Drug Response Prediction. MATDRP can integrate the internal correlations of features and external relationships of biomedical entities by proposing a multi-view attention-based deep learning framework, thus improving the performance of anticancer drug response prediction.

# Data directory
There are 3 datasets, i.e.,PRISM, GDSC, and TCGA. For each dataset, We provide drug data (i.e., physicochemical properties and molecular fingerprint), cell line data (i.e., gene expression profiles and copy number variation profiles), and all known drug response data.

# Requirements
MATDRP is tested to work under:
* Python 3.6
* torch 1.5.0
* numpy 1.19.1
* sklearn 0.23.2

# Quick start
* Create a directory "output".  
* Run python MATDRP.py to reproduce the cross validation results of MATDRP. Options are:  
-epochs: The Number of epochs to train, default: 15.  
-batch: Number of batch size, default: 127.  
-lr: Learning rate, default: 0.0001.  
-hidden: Number of hidden units: 64.  
-nb_heads: Number of head attentions, default: 4.  
-nb_heads: Alpha for the leaky_relu, default: 0.1.  
-patience: Patience in the early stop mechanism, default: 5.  
