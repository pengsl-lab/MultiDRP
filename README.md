# MultiDRP
MultiDRP is a hierarchical attention network for anticancer drug response prediction. MultiDRP can integrate both internal correlation of feature items and external relationship of biomedical entities by using graph attention and self-attention models to improve the anticancer drug response prediction.

# Data directory
In PRISM dataset, we provide drug data (i.e., physicochemical properties and molecular fingerprint), cell line data (i.e., gene expression profiles and copy number variation profiles), and all known drug response data.

# Requirements
MultiDRP is tested to work under:
* Python 3.6
* torch 1.5.0
* numpy 1.19.1
* sklearn 0.23.2

# Quick start
* Unzip "data.zip" in ./data
* Create a directory "output".  
* Run python MultiDRP.py to reproduce the cross validation results of MultiDRP. Options are:  
-epochs: The Number of epochs to train, default: 15.  
-batch: Number of batch size, default: 128.  
-lr: Learning rate, default: 0.0001.  
-hidden: Number of hidden units: 64.  
-nb_heads: Number of head attentions, default: 4.  
-nb_heads: Alpha for the leaky_relu, default: 0.2.  
-patience: Patience in the early stop mechanism, default: 5.  
