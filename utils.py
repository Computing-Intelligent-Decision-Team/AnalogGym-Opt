import torch
import math
import warnings
from torch import Tensor
import numpy as np

import os

from dev_params import DeviceParams

PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ActionNormalizer():
    """重新缩放和调整动作范围。"""
    def __init__(self, action_space_low, action_space_high, action_space_step=None):
        self.action_space_low = action_space_low     
        self.action_space_high = action_space_high
        self.action_space_step = action_space_step

    def action(self, action: np.ndarray) -> np.ndarray:
        """将 [-1, 1] 的范围转换为真实的 [low, high]，如果提供了最小步长也会将其考虑在内。"""
        low, high = self.action_space_low, self.action_space_high
        # Rescale action from [-1, 1] to [low, high]
        action = low + (action + 1) * (high - low) / 2
        action = np.clip(action, low, high)
        
        # Apply minimum step constraint if provided and valid
        if self.action_space_step is not None:
            step_array = np.asarray(self.action_space_step)
            # Only apply step constraint where step is valid (>0)
            for i in range(len(action)):
                if step_array[i] is not None and step_array[i] > 0:
                    # Calculate and round steps, then reconstruct action
                    steps = np.round((action[i] - low[i]) / step_array[i])
                    action[i] = low[i] + steps * step_array[i]
                    action[i] = np.clip(action[i], low[i], high[i])
        
        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """将真实的 [low, high] 转换为算法所需的 [-1, 1] 范围。"""
        low, high = self.action_space_low, self.action_space_high
        
        # Rescale action from [low, high] to [-1, 1]
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, -1.0, 1.0)
        
        return action

class OutputParser2(DeviceParams):  
    
    def __init__(self, ckt_hierarchy, base_dir: str = None):
        self.ckt_hierarchy = ckt_hierarchy 
        # 初始化op字典
        self.op = {}
        for device in self.ckt_hierarchy:
            device_name = device[0]
            self.op[device_name] = {}
        # allow overriding base dir to support per-corner parsing
        self.base_dir = base_dir if base_dir is not None else SPICE_NETLIST_DIR

        super().__init__(self.ckt_hierarchy)       

    def ac(self, file_name):
        try:
            AMP_PFC_ac = open(f'{self.base_dir}/{file_name}', 'r')  
            lines_ac = AMP_PFC_ac.readlines()     
            freq = []                        
            cmrrdc_ac = []
            PSRP_ac = []
            PSRN_ac = []
            dcgain_ac = []
            for line in lines_ac:
                Vac = line.split(' ')                 
                Vac = [i for i in Vac if i != '']    
                freq.append(float(Vac[0]))           
                cmrrdc_ac.append(float(Vac[1]))
                PSRP_ac.append(float(Vac[3]))
                PSRN_ac.append(float(Vac[5]))
                dcgain_ac.append(float(Vac[7]))#除温度外添加的值均一样
                
            return freq, cmrrdc_ac, PSRP_ac, PSRN_ac, dcgain_ac
        except:
            print(f"Simulation errors in {self.base_dir}: no file: {file_name}.")   
            return None

    def GBW_PM(self, file_name):
        try:
            AMP_PFC_GBW_PM = open(f'{self.base_dir}/{file_name}', 'r') 
            lines_GBW_PM = AMP_PFC_GBW_PM.readlines()     
            freq = []                        
            GBW_ac = []
            phase_margin_ac = []
            for line in lines_GBW_PM:
                Vac = line.split(' ')                 
                Vac = [i for i in Vac if i != '']     
                freq.append(float(Vac[0]))            
                GBW_ac.append(float(Vac[1]))
                phase_margin_ac.append(float(Vac[3]))
                
            return freq, GBW_ac, phase_margin_ac
        except:
            print(f"Simulation errors in {self.base_dir}: no file: {file_name}.")
            return None
            
    def dc(self, file_name):
        try:
            AMP_PFC_dc = open(f'{self.base_dir}/{file_name}', 'r')
            lines_dc = AMP_PFC_dc.readlines()
            Temp_dc = []                     
            TC_dc = []
            Power_dc = []
            vos_dc = []
            for line in lines_dc:
                Vdc = line.split(' ')
                Vdc = [i for i in Vdc if i != '']
                Temp_dc.append(float(Vdc[0]))#DC文件第一列
                TC_dc.append(float(Vdc[1]))#DC文件第二列
                Power_dc.append(float(Vdc[3])) 
                vos_dc.append(float(Vdc[5]))
          #-40度到120度仿真的dc值，且加入列表的值除了温度其他几个数据都一样
            return Temp_dc, TC_dc, Power_dc, vos_dc
        except:
            print(f"Simulation errors in {self.base_dir}: no file: {file_name}.")
            return None
      
    def tran(self, file_name):
        try:
            AMP_PFC_tran = open(f'{self.base_dir}/{file_name}', 'r')
            lines_tran = AMP_PFC_tran.readlines()
            time = []                         
            sr_rise = []
            sr_fall = []
            for line in lines_tran:
                line = line.split(' ')
                line = [i for i in line if i != '']
                time.append(float(line[0]))
                sr_rise.append(float(line[1]))
                sr_fall.append(float(line[3]))

            return time, sr_rise, sr_fall
        except:
            print(f"Simulation errors in {self.base_dir}: no file: {file_name}.")
            return None

            
    def dcop(self, file_name):
        try:
            AMP_PFC_op = open(f'{self.base_dir}/{file_name}', 'r')
            
            lines_op = AMP_PFC_op.readlines()
            for index, line in enumerate(lines_op):
                if line == "Values:\n":       
                    start_idx = index
            # 提取VDD参数（Values:后第1行）
            vdd_line = lines_op[start_idx+1]
            self.op['VDD'] = {}
            self.op['VDD']['v'] = float(vdd_line.split('\n')[0].split('\t')[1])
            # 提取其他器件参数（Values:后第2行开始）
            _lines_op = lines_op[start_idx+2:-1]     
            lines_op = []
            for _line in _lines_op:            
                lines_op.append(float(_line.split('\n')[0].split('\t')[1]))   
            
            num_dev = len(self.ckt_hierarchy)
            num_dev_params_mos = len(self.params_mos)
            num_dev_params_r = len(self.params_r)
            num_dev_params_c = len(self.params_c)
            num_dev_params_i = len(self.params_i)
            num_dev_params_v = len(self.params_v)
            
            idx = 0         
            for i in range(num_dev):        
                dev_type = self.ckt_hierarchy[i][3]
                if dev_type == 'm' or dev_type == 'M':   
                    for j in range(num_dev_params_mos):
                        param = self.params_mos[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_mos
                elif dev_type == 'r' or dev_type == 'R':  
                    for j in range(num_dev_params_r):
                        param = self.params_r[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_r
                elif dev_type == 'c' or dev_type == 'C':
                    for j in range(num_dev_params_c):
                        param = self.params_c[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_c
                elif dev_type == 'i' or dev_type == 'I':
                    for j in range(num_dev_params_i):
                        param = self.params_i[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_i
                elif dev_type == 'v' or dev_type == 'V':
                    for j in range(num_dev_params_v):
                        param = self.params_v[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_v
                else:
                    None
            
            return self.op
        except:
            print(f"Simulation errors in {self.base_dir}: no file: {file_name}.")
            return None

    def analyze_amplifier_performance(self, vinp, vout, time, d0):
        vinp = np.array(vinp)  
        vout = np.array(vout)
        time = np.array(time)
        def get_step_parameters(vinp, time):
            dv = np.diff(vinp)
            t0 = time[np.where(dv > 0)[0][0]]
            t1 = time[np.where(dv < 0)[0][0]]
            v0 = np.median(vinp[time < t0])
            v1 = np.median(vinp[(time > t0) & (time < t1)])
            return v0, v1, t0, t1
        v0, v1, t0, t1 = get_step_parameters(vinp, time)#取电压上升和下降的时间为t0、t1；变化前后的中位数作为变化前后的电压值  
    
        pre_step_data = vout[time < t0]
        delta0 = (pre_step_data - v0) / v0
        d0_settle = np.mean(np.abs(delta0))
        stable = not np.any(np.abs(delta0) > d0)
    
        def find_settling_time_index(delta, d0):
            for i in range(len(delta)):
                if np.all(np.abs(delta[i:]) < d0):
                    return i
            return None
    
        def get_slope_and_settling_time(vout, time, v0, v1, start_t, end_t, d0, mode):
            idx = (time >= start_t) & (time <= end_t)
            vout_segment = vout[idx]
            time_segment = time[idx]
    
            target_value = v0 + (v1 - v0) / 2
            idx_target = np.where(vout_segment >= target_value)[0][0] if np.any(vout_segment >= target_value) else None
            if idx_target is None:
                SR = np.nan
            else:
                SR = np.gradient(vout_segment, time_segment)[idx_target]
    
            if mode == 'positive':
                delta = (vout_segment - v1) / v1
            else:
                delta = (vout_segment - v0) / v0

            idx_settle = find_settling_time_index(delta, d0)
            if idx_settle is None:
                settling_time = np.nan
                d_settle = np.mean(np.abs(delta))
            else:
                settling_time = time_segment[idx_settle] - start_t
                d_settle = np.mean(np.abs(delta[idx_settle:]))
            return SR, settling_time, d_settle

        SR_p, settling_time_p, d1_settle = get_slope_and_settling_time(vout, time, v0, v1, t0, t1, d0, 'positive')

        SR_n, settling_time_n, d2_settle = get_slope_and_settling_time(vout, time, v0, v1, t1, np.max(time), d0, 'negative')

        return d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n    

    def extract_tran_data(self,file_name):
        time_points = []
        raw_data = []
        vin_data = []
        vout_data = []
        time_data = []
        data_section = False
        with open(f'{self.base_dir}/{file_name}', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    if line.startswith('Values:'):
                        data_section = True
                        continue
                    if data_section:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            time_points.append(int(parts[0]))
                            raw_data.append(float(parts[1]))
                        else:
                            raw_data.append(float(parts[0]))  

        if len(time_points) != len(raw_data)/3:
            print('Error in extracting transient data')
            return None, None
        for i in time_points:
            time_data.append(raw_data[3*i])
            vin_data.append(raw_data[3*i+2])
            vout_data.append(raw_data[3*i+1])

        return time_data, vin_data, vout_data

# https://zhuanlan.zhihu.com/p/521318833
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / np.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std) 
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()  

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def trunc_normal(mean, std, a=-1, b=1):  
    
    """
    wrapper of <trunc_normal_> to work with np.array
    """
    output = np.zeros(len(mean))
    
    for i in range(len(mean)):
        output[i] = trunc_normal_(torch.empty(1), mean[i], std, a, b)[0]

    return output
    