## How to use it
### To train the LSTM
1. Download project and create a folder named 'timit-feats' (this folder name can be changed in Main.py) that contains X_train.npy, y_train.npy, X_test.npy and y_test.npy files. 
2. X files should be lists of np.arrays. This arrays has shape [n_windows, n_features] for each wav file.
3. y files are lists of lists with the same number of windows in each inner list as in each X file.
4. Customize any parameter inside this file (epochs, lr, ...)
5. Run it

### To train RBM
1. Load the data already segmented in windows as in the previous step.
2. Preprocess it, if necessary.
3. Import the RBM architecture that best fits your data from commons/rbms.
4. Instantiate the architecture like rbm = RBM(n_visible, n_hidden, k, logsdir) where k is the CD step parameter and logsdir is the output file.
5. Train it calling rbm.fit(...).
6. Backpropagate through the hidden bias with rbm.fine_tune(...).
