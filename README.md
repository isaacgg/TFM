## How to use it
### To train the LSTM
\begin{itemize}
\item Download project and create a folder named 'timit-feats' (this folder name can be changed in Main.py) that contains X_train.npy, y_train.npy, X_test.npy and y_test.npy files. 
\item X files should be lists of np.arrays. This arrays has shape [n_windows, n_features] for each wav file.
\item y files are lists of lists with the same number of windows in each inner list as in each X file.
\end{itemize}

### To train RBM
\begin{itemize}
\item Load the data already segmented in windows as in the previous step.
\item Preprocess it, if necessary.
\item Import the RBM architecture that best fits your data from commons/rbms.
\item Instantiate the architecture like rbm = RBM(n_visible, n_hidden, k, logsdir) where k is the CD step parameter and logsdir is the output file.
\item Train it calling rbm.fit(...).
\item Backpropagate through the hidden bias with rbm.fine_tune(...).
\end{itemize}
