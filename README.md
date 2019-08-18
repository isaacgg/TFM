## How to use it
### To train the LSTM
Download project and create a folder named 'timit-feats' (this folder name can be changed in Main.py) that contains X_train.npy, y_train.npy, X_test.npy and y_test.npy files. 
X files should be lists of np.arrays. This arrays has shape [n_windows, n_features] for each wav file.
y files are lists of lists with the same number of windows in each inner list as in each X file.

### To train RBM
Load the data already segmented in windows as in the previous step.
Preprocess it if necessary
Import the RBM architecture that best fits your data from commons/rbms
Instantiate the architecture like rbm = RBM(n_visible, n_hidden, k, logsdir) where k is the CD step parameter and logsdir is the output file.
Train it calling rbm.fit(...)
Backpropagate through the hidden bias with rbm.fine_tune(...)
