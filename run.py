import os
import time
import random
import argparse
import numpy as np
import warnings
import subprocess
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score


logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

def get_free_gpu():
    try:
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip().split('\n')
        free_memory = [int(x) for x in result]
        best_gpu = int(np.argmax(free_memory))
        return str(best_gpu)
    except:
        return "0"

os.environ['CUDA_VISIBLE_DEVICES'] = get_free_gpu()
set_global_seed(42)

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from model.Net import build_ortho_model


class AsymRegTrainer(tf.keras.Model):
    def __init__(self, base_model, ortho_weight=0.13, reg_weight=0.01):
        super().__init__()
        self.base_model = base_model
        self.ortho_weight = ortho_weight
        self.reg_weight = reg_weight

    def compile(self, optimizer, loss_fn, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.loss_fn = loss_fn

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)[0]

    @tf.function(reduce_retracing=True)
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, e_f, t_f = self.base_model(x, training=True)
            cls_loss = self.loss_fn(y, y_pred)
            e_n, t_n = tf.nn.l2_normalize(e_f, axis=1), tf.nn.l2_normalize(t_f, axis=1)
            ortho_loss = tf.reduce_mean(tf.square(tf.reduce_sum(e_n * t_n, axis=1)))
            total_loss = cls_loss + self.ortho_weight * ortho_loss + self.reg_weight * (tf.reduce_mean(tf.abs(e_f)) + tf.reduce_mean(tf.square(t_f)) * 0.5)
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function(reduce_retracing=True)
    def test_step(self, data):
        x, y = data
        y_pred, _, _ = self.base_model(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

class WeightSmoothingCallback(tf.keras.callbacks.Callback):
    def __init__(self, smoothing_start):
        super().__init__()
        self.smoothing_start = smoothing_start
        self.weight_history = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.smoothing_start:
            self.weight_history.append(self.model.base_model.get_weights())


class Exporter:
    def __init__(self, net_name, log_folder):
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        self.txt_path = os.path.join(self.log_folder, f"results_{net_name}.txt")
    
    def export2txt(self, content, print_console=False):
        if print_console:
            print(content)
        with open(self.txt_path, 'a') as f:
            f.write(content + '\n')


class Runner:
    def __init__(self, configs, log_folder):
        self.configs = configs
        self.log_folder = log_folder
        self.net_name = configs['net_name']
        self.data_dir = configs['data_dir_name']
        self.lr = configs['lr']
        self.epochs = configs['NumOfEpoch']
        self.batch_size = configs['batchsize']
        self.train_strategy = configs['train_strategy']
        self.val_ratio = configs['validation_ratio']
        
        self.model_save_base = os.path.join(os.path.dirname(__file__), 'Saved_files', 'trained_model', self.net_name)
        os.makedirs(self.model_save_base, exist_ok=True)
        
        self.exporter = Exporter(self.net_name, self.log_folder)

    def ho_train(self, sub):
        K.clear_session()
        set_global_seed(42)
        sub_start = time.time()

        data_path = os.path.join(self.data_dir, f'processed_BCI2a_subject_{sub}.npz')
        data = np.load(data_path)
        X_tr, y_tr = data['X_train'].astype('float32'), data['y_train'].astype(int)
        X_te, y_te = data['X_test'].astype('float32'), data['y_test'].astype(int)

        if X_tr.shape[1] == 1:
            X_tr, X_te = np.transpose(X_tr, (0, 2, 3, 1)), np.transpose(X_te, (0, 2, 3, 1))
        if np.min(y_tr) == 1:
            y_tr -= 1
            y_te -= 1

        base_model = build_ortho_model()
        trainer = AsymRegTrainer(base_model, ortho_weight=self.configs['ortho_weight'], reg_weight=self.configs['reg_weight'])
        
        steps_per_epoch = len(X_tr) // self.batch_size
        lr_sched = tf.keras.optimizers.schedules.CosineDecay(self.lr, self.epochs * steps_per_epoch)
        trainer.compile(Adam(lr_sched), tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01), ['accuracy'])

        file_name = f'{self.net_name}_sub{sub}.h5'
        save_path = os.path.join(self.model_save_base, file_name)

        
        if self.train_strategy == 'standard':
            print(f"--- Sub {sub:02d} | 训练开始 ---")
            ws_cb = WeightSmoothingCallback(smoothing_start=self.epochs - 20)
            
            
            trainer.fit(X_tr, to_categorical(y_tr, 4), batch_size=self.batch_size, epochs=self.epochs, callbacks=[ws_cb], verbose=0)
            
            if ws_cb.weight_history:
                avg_w = [np.mean(np.array([w[i] for w in ws_cb.weight_history]), axis=0) for i in range(len(ws_cb.weight_history[0]))]
                base_model.set_weights(avg_w)
                
            base_model.save_weights(save_path)

        elif self.train_strategy == 'holdout':
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_tr, y_tr, test_size=self.val_ratio, random_state=42, stratify=y_tr
            )
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
            
            trainer.fit(
                X_train_split, to_categorical(y_train_split, 4),
                validation_data=(X_val_split, to_categorical(y_val_split, 4)),
                batch_size=self.batch_size, epochs=self.epochs, 
                callbacks=[early_stop], verbose=0
            )
            base_model.save_weights(save_path)

        y_pred = np.argmax(base_model.predict(X_te, verbose=0)[0], axis=1)
        acc = accuracy_score(y_te, y_pred) * 100
        kappa = cohen_kappa_score(y_te, y_pred)
        
        elapsed = time.time() - sub_start
        self.exporter.export2txt(f'Sub {sub:02d} 测试 -> Acc: {acc:.2f}% | Kappa: {kappa:.4f} | 耗时: {elapsed:.1f}s', print_console=True)
        
        return acc, kappa

    def run(self):
        all_accs = []
        all_kappas = []
        
        subjectNum = 9 if '2a' in self.data_dir.lower() else 0
        
        for i in range(1, subjectNum + 1):
            acc, kappa = self.ho_train(sub=i)
            all_accs.append(acc)
            all_kappas.append(kappa)

        mean_acc = np.mean(all_accs)
        mean_kappa = np.mean(all_kappas)
        
        summary = (
            f"\n======================================\n"
            f"总平均准确率: {mean_acc:.2f}%\n"
            f"Kappa: {mean_kappa:.4f}\n"
            f"======================================"
        )
        self.exporter.export2txt(summary, print_console=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Pipeline')
    
    
    parser.add_argument("--train_strategy", type=str, default='standard', choices=['standard', 'holdout'])
    parser.add_argument("--net_name", type=str, default='PhysOrthoNet')
    parser.add_argument("--data_dir_name", type=str, default="/home/Data/bci_2a")
    
    parser.add_argument("--NumOfEpoch", type=int, default=300)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--betatwo", type=float, default=0.999)
    parser.add_argument("--scheduler", type=bool, default=True)
    parser.add_argument("--validation_ratio", type=float, default=0.3)
    
    parser.add_argument("--ortho_weight", type=float, default=0.13)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    
   
    parser.add_argument("--earlystop", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--save_logs", type=bool, default=True)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--window_len", type=int, default=1125)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--drop_att", type=float, default=0.2)
    parser.add_argument("--isAttention", type=bool, default=True)
    parser.add_argument("--isGaussianNoise", type=bool, default=False)
    parser.add_argument("--p", type=int, default=3)

    args = parser.parse_args()
    configs = vars(args)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_str = str(random.randint(100, 999))
    log_folder = os.path.join(os.path.dirname(__file__), 'Saved_files', 'LOG', 
                              f"{configs['net_name']}_{configs['train_strategy']}_{timestamp}_{random_str}")
    
    runner_ = Runner(configs, log_folder)
    runner_.run()