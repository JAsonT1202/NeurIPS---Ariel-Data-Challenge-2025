import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm
from astropy.stats import sigma_clip
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip
import pandas as pd
import numpy as np

from tqdm import tqdm
from pqdm.threads import pqdm
import itertools

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

def get_data_2(PlanetIds, config, preprocessed_data, ROOT_PATH,MODE):
    class TransitModel2:
        def __init__(self, config):
            self.cfg = config
    
        def _phase_detector(self, signal):
            search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
            min_index = np.argmin(signal[search_slice]) + search_slice.start
            
            signal1 = signal[:min_index]
            signal2 = signal[min_index:]
    
            grad1 = np.gradient(signal1)
            grad1 /= grad1.max()
            
            grad2 = np.gradient(signal2)
            grad2 /= grad2.max()
    
            phase1 = np.argmin(grad1)
            phase2 = np.argmax(grad2) + min_index
    
            return phase1, phase2
        
        def _objective_function(self, s, signal, phase1, phase2):
            delta = self.cfg.MODEL_OPTIMIZATION_DELTA
            power = self.cfg.MODEL_POLYNOMIAL_DEGREE
    
            if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
                delta = 2
    
            y = np.concatenate([
                signal[: phase1 - delta],
                signal[phase1 + delta : phase2 - delta] * (1 + s),
                signal[phase2 + delta :]
            ])
            x = np.arange(len(y))
    
            coeffs = np.polyfit(x, y, deg=power)
            poly = np.poly1d(coeffs)
            error = np.abs(poly(x) - y).mean()
            
            return error
    
        def predict(self, single_preprocessed_signal):
            signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
            signal_1d = savgol_filter(signal_1d, 23, 2)
            
            phase1, phase2 = self._phase_detector(signal_1d)
    
            phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
            phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    
    
            result = minimize(
                fun=self._objective_function,
                x0=[0.0001],
                args=(signal_1d, phase1, phase2),
                method="Nelder-Mead"
            )
            base = result.x[0]
            
            array_length = 283  # 数组总长度
            window_size = 60    # 切片长度
            half_window = window_size // 2  # 25
            result_list = [base]
            cut = 70
            for i in range(283):
                if i<cut: 
                    #result_list.append(base)
                    continue 
                else:
                    j = array_length-i
                    if j - half_window <0:
                        continue
                    start = max(0, j - half_window)  # 初始起始位置
                    start = min(start, array_length - window_size)  # 限制最大起始位置
                    end = start + window_size  # 结束位置
    
                    #print(start,end)
                    signal_1d = single_preprocessed_signal[:, 1:][:,start:end].mean(axis=1)
                    signal_1d = savgol_filter(signal_1d, 23, 2)
                    
                    #phase1, phase2 = self._phase_detector(signal_1d)
            
                    #phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
                    #phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    
            
                    result = minimize(
                        fun=self._objective_function,
                        x0=[0.0001],
                        args=(signal_1d, phase1, phase2),
                        method="Nelder-Mead"
                    )
        
                    result_list.append(result.x[0])
            return result_list
    
        def predict_all(self, preprocessed_signals):
            predictions = [
                self.predict(preprocessed_signal)
                for preprocessed_signal in tqdm(preprocessed_signals)
            ]
            return np.array(predictions) * self.cfg.SCALE
    
    model2 = TransitModel2(config)
    predictions2 = model2.predict_all(preprocessed_data)
    predictions_df2 = pd.DataFrame({
        "planet_id": PlanetIds,
        **{f"pre_{i}": predictions2[:, i] for i in range(185)}
    })
    
    StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
    StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
    PlanetIds = StarInfo["planet_id"].tolist()
    StarInfo = StarInfo.set_index("planet_id")
    
    input_df2 = pd.merge(predictions_df2, StarInfo, on="planet_id", how="left")
    
    for i in range(185):
        input_df2[f"pre_{i}"] *= 10000
        
    features = [f"pre_{i}" for i in range(185)] + ['Rs', 'i']
    
    X2 = input_df2[features].values.astype(np.float32)
    X_tensor2 = torch.tensor(X2, dtype=torch.float32)
    return X_tensor2