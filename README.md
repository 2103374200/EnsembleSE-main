# EnsembleSE:Identification of Super-Enhancers Based on Ensemble Learning

EnsembleSE is a model based on ensemble learning, using a deep hybrid model and traditional machine learning models to predict super-enhancers from typical enhancers. 

In this work, we complete the prediction task using only the sequence, while also considering the mining of deep abstract semantic relationships and shallow biological low-dimensional features of the sequence, providing effective predictive factors for the identification of super-enhancers for subsequent research.

## File descriptions

### feature/get_3_dna2vec.py

Encode the DNA sequences of typical enhancers (TEs) and super-enhancers (SEs) using the pre-trained 64-dimensional k-mer feature vectors in a tripartite manner.

### feature/3mer-third.py

Perform tripartite encoding of the DNA sequences for typical enhancers and super-enhancers using 3-mer frequency information.

### feature/3-pseEIIP.py

Perform tripartite encoding of the DNA sequences for typical enhancers and super-enhancers using pseEIIP features.

### feature_selection.py

Use the fscore method for feature selection on the encoded feature files.

### alstmSE.py

Deep learning model architecture for classifying deep features.

### se_train.py

The main file for training  on mouse and human cells.

### se_predict.py

The main file for predicting super-enhancers on mouse and human cells.

## Step by step for training model

### **Step 1**: Install third-party packages.

EnsembleSE requires the runtime enviroment with Python >=3.9. The modules requried by EnsembleSE is provied in the file requirements.txt, and you can install them by

```
pip install requirements.txt
```

### **Step 2**: Encode the DNA sequences of mouse and human cell lines.

This experiment involves two cell lines. You can replace the dataset path in the encoding file to complete the encoding of the training set and test set for both cell lines.

Please complete the separate encoding of the training and test sets for the two cell lines in the three encoding files.You can input the command lines as follows:

```
python get_3_dna2vec.py
```

```
python 3mer-third.py
```

```
python 3-pseEIIP.py
```

### Step 3: Train and predict the model.

Use "train.py" to train SENet model

```
python se_train.py
```

Use "predict.py" to predict super-enhancers

```
python se_predict.py
```

