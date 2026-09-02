import os
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

DATA_DIR = "/home/dataset"
SAVE_PATH = "/home/Data/bci_2a"

os.makedirs(SAVE_PATH, exist_ok=True)

def load_mat_data(mat_path):
    fs = 250
    t1 = int(1.5 * fs)
    t2 = int(6.0 * fs)
    window_Length = 7 * fs

    a = sio.loadmat(mat_path)
    a_data = a['data']

    X_list = []
    y_list = []

    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]

        for trial in range(0, a_trial.size):
            start = int(a_trial[trial])
            end = start + window_Length
            trial_data = np.transpose(a_X[start:end, :22])
            X_list.append(trial_data)
            y_list.append(int(a_y[trial]) - 1)

    X = np.array(X_list)[:, :, t1:t2]
    y = np.array(y_list)
    return X, y

def run_pro_preprocessing():
    for sub in range(1, 10):
        train_mat = os.path.join(DATA_DIR, f"A0{sub}T.mat")
        test_mat = os.path.join(DATA_DIR, f"A0{sub}E.mat")

        X_train, y_train = load_mat_data(train_mat)
        X_test, y_test = load_mat_data(test_mat)

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

        X_train = X_train[:, np.newaxis, :, :].astype('float32')
        X_test = X_test[:, np.newaxis, :, :].astype('float32')

        for j in range(22):
            scaler = StandardScaler()
            scaler.fit(X_train[:, 0, j, :])
            X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
            X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

        save_name = os.path.join(SAVE_PATH, f"processed_BCI2a_subject_{sub}.npz")
        np.savez(save_name, 
                 X_train=X_train, y_train=y_train, 
                 X_test=X_test, y_test=y_test,
                 y_train_onehot=to_categorical(y_train, 4),
                 y_test_onehot=to_categorical(y_test, 4))

if __name__ == "__main__":
    run_pro_preprocessing()