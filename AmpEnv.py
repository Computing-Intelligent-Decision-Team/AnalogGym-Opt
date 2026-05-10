import numpy as np
import os
import math
import json
import shutil
from tabulate import tabulate
from reporting_metrics import build_reporting_table_rows
from itertools import product
import multiprocessing
import subprocess

from dev_params import DeviceParams
from utils import ActionNormalizer, OutputParser2

class AmpEnv(DeviceParams):
    """Ngspice-backed reinforcement-learning environment for amplifier sizing."""
    def __init__(self, circuit_config: dict):
        self.circuit_config = circuit_config
        self.ckt_hierarchy = tuple(tuple(item) for item in self.circuit_config.get('ckt_hierarchy', ()))
        self.device_params = DeviceParams(self.ckt_hierarchy)
        DeviceParams.__init__(self, self.ckt_hierarchy)
        # Initialize device parameters from configuration
        action_low, action_high, action_step, initial_params = [], [], [], []
        self.paraBound = {}
        self.device_count = {}
        for device, config in circuit_config['device'].items():
            self.device_count[device] = config['num']
            if 'range' in config:
                for param, bounds in config['range'].items():
                    # Collect range values
                    action_low.append(bounds[0])
                    action_high.append(bounds[1])
                    self.paraBound[f"{param}_{device}"] = {'min': float(bounds[0]), 'max': float(bounds[1])}
                    # Collect step values if defined, otherwise use None
                    if 'step' in config and param in config['step'] and config['step'][param] > 0:
                        action_step.append(config['step'][param])
                    else:
                        action_step.append(None)
            if 'init' in config:
                initial_params.extend(config['init'][param] for param in config['init'])
        # Convert to numpy arrays
        self.action_space_low = np.array(action_low)
        self.action_space_high = np.array(action_high)
        self.action_space_step = action_step if action_step else None
        self.initial_params = np.array(initial_params)
        self.action_dim = len(self.action_space_low)

        # Generated simulation workspace. Circuit templates are kept read-only.
        self.base_sim_dir = os.path.join(os.getcwd(), 'simulation_output')
        if os.path.exists(self.base_sim_dir):
            shutil.rmtree(self.base_sim_dir)
        os.makedirs(self.base_sim_dir, exist_ok=True)

        # Read-only simulation template directory resolved by circuit_config_loader.
        self.simulation_files_dir = os.path.dirname(circuit_config['paths']['ACDC_cir_path'])
        self.path = circuit_config['paths']
        self.post_target_bonus_config = self.circuit_config.get('post_target_bonus', {}) or {}
        
        # Small constant for numerical stability
        self.eps = 1e-8

    def _score_higher_better(self, value, target):
        """Return a non-positive score for a maximize objective."""
        if value is None or math.isnan(value) or math.isinf(value):
            return -1.0
        denominator = abs(value) + abs(target) + self.eps
        score = (value - target) / denominator
        return min(0.0, score)

    def _score_lower_better(self, value, target):
        """Return a non-positive score for a minimize objective."""
        if value is None or math.isnan(value) or math.isinf(value):
            return -1.0
        denominator = abs(value) + abs(target) + self.eps
        score = (target - value) / denominator
        return min(0.0, score)   

    def _bonus_higher_better(self, value, target, scale=1.0, max_bonus=1.0):
        """Positive bonus used only after the original all-satisfied state is reached."""
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        denominator = abs(value) + abs(target) + self.eps
        score = (value - target) / denominator
        return float(np.clip(score * scale, 0.0, max_bonus))

    def _bonus_lower_better(self, value, target, scale=1.0, max_bonus=1.0):
        """Positive bonus used only after the original all-satisfied state is reached."""
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        denominator = abs(value) + abs(target) + self.eps
        score = (target - value) / denominator
        return float(np.clip(score * scale, 0.0, max_bonus))
    
    def _generate_performance_table(self, corner_info):
        """Generate performance parameter table
        
        Args:
            corner_info: Information dictionary for a single corner
            
        Returns:
            Formatted performance parameter table
        """
        # Get performance parameters from circuit configuration
        performance = self.circuit_config['performance']
        
        table_data = build_reporting_table_rows(corner_info, performance)
        
        # Generate table
        return tabulate(
            table_data,
            headers=['param', 'num', 'target'], 
            tablefmt='orgtbl', 
            numalign='right', 
            floatfmt=".8f"
        )
        
    def _parse_action_to_params(self, action):
        """Parse action array into individual parameters dictionary
        
        Args:
            action: Action array containing all circuit parameters
        
        Returns:
            Dictionary of individual parameters with full parameter names
        """
        individual_params = {}
        param_index = 0
        for device, info in self.circuit_config['device'].items():
            for param in info['range'].keys():
                full_param_name = f"{param}_{device}"
                if param_index < len(action):
                    value = action[param_index]
                    # Convert to integer if parameter is M (multiplier)
                    if param == 'M':
                        value = int(value)
                    individual_params[full_param_name] = value
                    param_index += 1
        return individual_params

    def _calculate_area(self, individual_params):
        """Calculate circuit area from individual parameters
        
        Args:
            individual_params: Dictionary of individual parameters with full parameter names
        
        Returns:
            Calculated circuit area
        """
        # Initialize variables
        transistor_area = 0.0
        capacitor_area = 0.0
        resistor_area = 0.0
        transistor_params = {}
        
        # Process parameters
        for key, value in individual_params.items():
            if not isinstance(value, (int, float)):
                continue
            # Handle transistor parameters
            if '_M' in key:
                param_type, device_name = key.split('_', 1)
                if device_name not in transistor_params:
                    transistor_params[device_name] = {}
                transistor_params[device_name][param_type] = value
            # Handle capacitor parameters
            elif '_C' in key:
                cap_name = key.split('_')[1]
                count = self.device_count[f'{cap_name}']
                capacitor_area += value * count
            # Handle resistor parameters
            elif '_R' in key:
                resistor_area += value
        
        # Calculate transistor area
        for device_name, params in transistor_params.items():
            if all(k in params for k in ['W', 'L', 'M']):
                count = self.device_count[f'{device_name}']
                transistor_area += params['W'] * params['L'] * params['M'] * count * 1e-6
        
        # Calculate total area
        capacitor_area = capacitor_area * 1.823 * 1089
        resistor_area = resistor_area * 1e-3 * 5
        total_area = transistor_area + capacitor_area + resistor_area
        return total_area ** 0.5
    
    def _write_vars_file(self, sim_dir: str, individual_params: dict):
        """Write SPICE variables file
        
        Args:
            sim_dir: Simulation directory
            individual_params: Dictionary of individual parameters with full parameter names
        """
        vars_filename = os.path.basename(self.path['vars_path'])
        vars_file_path = os.path.join(sim_dir, vars_filename)
        
        # Write all parameters to vars file
        lines = []
        for key, value in individual_params.items():
            lines.append(f'.param {key}={value}\n')
        with open(vars_file_path, 'w') as f:
            f.writelines(lines)

    def _do_simulation(self, sim_dir: str):
        ACDC_cir_name = os.path.basename(self.path['ACDC_cir_path'])
        result = subprocess.run(['ngspice', '-b', '-o', 'ACDC.log', ACDC_cir_name],cwd=sim_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR during simulation in {sim_dir}: {result.returncode}")
            return None
        Tran_cir_name = os.path.basename(self.path['Tran_cir_path'])
        result = subprocess.run(['ngspice', '-b', '-o', 'Tran.log', Tran_cir_name],cwd=sim_dir, capture_output=True, text=True)  
        if result.returncode != 0:
            print(f"ERROR during simulation in {sim_dir}: {result.returncode}")
            return None
    
    def do_simulation(self, sim_dir: str, individual_params: dict):
        """Execute simulation for single simulation directory
        
        Args:
            sim_dir: Simulation directory
        """
        self._do_simulation(sim_dir)
        sim_results = OutputParser2(self.ckt_hierarchy, sim_dir)
        observation = self._get_obs(sim_results, individual_params)
        info = self._get_info(sim_results, individual_params)
        reward = info['reward']
        return observation, info, reward

    def reset(self):
        """Run the initial simulations."""
        sim_dir = os.path.join(self.base_sim_dir, 'reset')
        os.makedirs(sim_dir, exist_ok=True)
        files_to_copy = [os.path.basename(self.path['ACDC_cir_path']), os.path.basename(self.path['Tran_cir_path'])]
        self._prepare_action_files(sim_dir, files_to_copy)
        # Use initial parameters from configuration
        individual_params = self._parse_action_to_params(self.initial_params)
        self._write_vars_file(sim_dir, individual_params)
        observation, info, reward = self.do_simulation(sim_dir, individual_params)
        return observation, info
    
    def close(self):
        return None
    
    def step(self, action):
        """Perform simulation step for given action.
        
        Args:
            action: Simulation action parameters
        """
        action = ActionNormalizer(action_space_low=self.action_space_low, action_space_high=self.action_space_high, action_space_step=self.action_space_step).action(action)  # Convert [-1, 1] range back to normal range
        print(f"Action: {action}")
        sim_dir = os.path.join(self.base_sim_dir, 'step')
        os.makedirs(sim_dir, exist_ok=True)
        files_to_copy = [os.path.basename(self.path['ACDC_cir_path']), os.path.basename(self.path['Tran_cir_path'])]
        self._prepare_action_files(sim_dir, files_to_copy)
        # Run simulations
        individual_params = self._parse_action_to_params(action)
        self._write_vars_file(sim_dir, individual_params)
        observation, info, reward = self.do_simulation(sim_dir, individual_params)
        terminated = reward >= 0

        return observation, reward, terminated, False, info

    def parallel_step(self, actions, enable_pvt=False):
        """Parallel simulate multiple batches of actions, supporting single action simulation.
        
        Args:
            actions: Batch of simulation action parameters (2D array, shape [batch_size, action_shape]) or single action (1D array, shape [action_shape])
            enable_pvt: Optional, whether to enable PVT simulation (process, voltage, temperature).
        """
        if len(np.array(actions).shape) == 1:
            # Single action, convert to single-element batch
            actions = [actions]
        normalized_actions = [ActionNormalizer(action_space_low=self.action_space_low, action_space_high=self.action_space_high, action_space_step=self.action_space_step).action(action) for action in actions]

        pvt_conditions = None
        all_tasks = []
        for action_idx, action in enumerate(normalized_actions):
            # Create action directory
            action_dir = os.path.join(self.base_sim_dir, f'action_{action_idx}')
            os.makedirs(action_dir, exist_ok=True)
            
            # Copy and process simulation files
            files_to_copy = [os.path.basename(self.path['ACDC_cir_path']), os.path.basename(self.path['Tran_cir_path'])]
            self._prepare_action_files(action_dir, files_to_copy)
            individual_params = self._parse_action_to_params(action)
            self._write_vars_file(action_dir, individual_params)
            
            # Prepare simulation tasks
            if enable_pvt:
                # Define PVT conditions
                PROCESSES = ['ss', 'fs', 'tt', 'sf', 'ff']
                VOLTAGES = ['1v62', '1v98']
                TEMPERATURES = ['-25C', '125C']
                pvt_conditions = list(product(PROCESSES, VOLTAGES, TEMPERATURES))
                pvt_conditions.insert(0, ('tt', None, None))  # Add TT corner as first condition
                
                # Create tasks for each PVT condition
                for (proc, volt, temp) in pvt_conditions:
                    corner_dir = os.path.join(action_dir, f'{proc}_{volt}_{temp}' if volt and temp else 'tt')
                    os.makedirs(corner_dir, exist_ok=True)
                    self._prepare_pvt_files(action_dir, corner_dir, files_to_copy, proc, volt, temp)
                    all_tasks.append((corner_dir, individual_params, action_idx, (proc, volt, temp)))
            else:
                all_tasks.append((action_dir, individual_params, action_idx, None))
        
        num_processes = multiprocessing.cpu_count()
        task_results = []
        
        if enable_pvt:
            # Number of actions to process in each batch for PVT corner simulation
            batch_size = 2  
            action_tasks = {}
            for task in all_tasks:
                action_tasks.setdefault(task[2], []).append(task)
            
            action_indices = list(action_tasks.keys())
            for i in range(0, len(action_indices), batch_size):
                batch_action_indices = action_indices[i:i + batch_size]
                batch_tasks = sum([action_tasks[idx] for idx in batch_action_indices], [])
                print(f"Simulating in parallel: {', '.join(f'action_{idx}' for idx in batch_action_indices)} (21 PVT corners, total {len(action_indices)} actions)")
                
                with multiprocessing.Pool(processes=min(len(batch_tasks), num_processes)) as pool:
                    # task_results.extend([pool.apply_async(self.single_simulation, args=task).get() for task in batch_tasks])
                    task_results.extend(pool.starmap(self.single_simulation, batch_tasks))
        else:
            # When PVT conditions are not enabled, execute all tasks directly
            with multiprocessing.Pool(processes=num_processes) as pool:
                # task_results = [pool.apply_async(self.single_simulation, args=task).get() for task in all_tasks]
                task_results = pool.starmap(self.single_simulation, all_tasks)
        
        return self._aggregate_results(task_results, normalized_actions, enable_pvt, pvt_conditions)
    
    def _prepare_action_files(self, action_dir, files_to_copy):
        """Prepare action-level simulation files"""
        # Clear previous simulation files
        for filename in os.listdir(action_dir):
            file_path = os.path.join(action_dir, filename)
            if os.path.isfile(file_path) and filename not in files_to_copy:
                os.remove(file_path)
        
        # Copy and update files
        sky130_lib_path = os.path.abspath('simulation_files/sky130_pdk')
        for filename in files_to_copy:
            src_file = os.path.join(self.simulation_files_dir, filename)
            dest_file = os.path.join(action_dir, filename)
            
            shutil.copy(src_file, dest_file)
            
            with open(dest_file, 'r') as f:
                content = f.read()
            
            # Update file paths
            content = content.replace(f'./{os.path.basename(self.path["netlist_path"])}', self.path["netlist_path"].replace('\\', '/'))
            content = content.replace(f'./{os.path.basename(self.path["vars_path"])}', os.path.abspath(os.path.join(action_dir, os.path.basename(self.path["vars_path"]))).replace('\\', '/'))
            content = content.replace(f'./{os.path.basename(self.path["dev_params_path"])}', self.path["dev_params_path"].replace('\\', '/'))
            
            # Update SKY130 library paths
            content = content.replace('../sky130_pdk/libs.tech/ngspice/corners/tt.spice', 
                                     os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'corners', 'tt.spice').replace('\\', '/'))
            content = content.replace('../sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical.spice', 
                                     os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'r+c', 'res_typical__cap_typical.spice').replace('\\', '/'))
            content = content.replace('../sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical__lin.spice', 
                                     os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'r+c', 'res_typical__cap_typical__lin.spice').replace('\\', '/'))
            content = content.replace('../sky130_pdk/libs.tech/ngspice/corners/tt/specialized_cells.spice', 
                                     os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'corners', 'tt', 'specialized_cells.spice').replace('\\', '/'))
            
            with open(dest_file, 'w') as f:
                f.write(content)
    
    def _prepare_pvt_files(self, action_dir, corner_dir, files_to_copy, proc, volt, temp):
        """Prepare simulation files for PVT conditions"""
        # Clear previous simulation files
        for filename in os.listdir(corner_dir):
            file_path = os.path.join(corner_dir, filename)
            if os.path.isfile(file_path) and filename not in files_to_copy:
                os.remove(file_path)
        
        # Copy files from action directory
        sky130_lib_path = os.path.abspath('simulation_files/sky130_pdk')
        for filename in files_to_copy:
            src_file = os.path.join(action_dir, filename)
            dest_file = os.path.join(corner_dir, filename)
            shutil.copy(src_file, dest_file)
            
            with open(dest_file, 'r') as f:
                content = f.read()
            
            if proc != 'tt':
                # Modify PVT conditions
                tt_spice_path = os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'corners', 'tt.spice').replace('\\', '/')
                tt_specialized_path = os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'corners', 'tt', 'specialized_cells.spice').replace('\\', '/')
                
                content = content.replace(tt_spice_path, os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'corners', f'{proc}.spice').replace('\\', '/'))
                content = content.replace(tt_specialized_path, os.path.join(sky130_lib_path, 'libs.tech', 'ngspice', 'corners', proc, 'specialized_cells.spice').replace('\\', '/'))
                content = content.replace('.PARAM supply_voltage = 1.8', f'.PARAM supply_voltage = {float(volt.replace("v", ".").replace("V", "."))}')
                content = content.replace('*.temp', f'.temp {int(temp.replace("C", "").replace("c", ""))}')
            
            with open(dest_file, 'w') as f:
                f.write(content)
    
    def _aggregate_results(self, task_results, normalized_actions, enable_pvt, pvt_conditions):
        """Aggregate and organize simulation results"""
        # Group results by action index
        action_results = {}
        for result in task_results:
            if result is not None:
                action_idx = result['action_idx']
                if action_idx not in action_results:
                    action_results[action_idx] = {'observations': [], 'infos': [], 'rewards': []}
                action_results[action_idx]['observations'].append(result['observation'])
                action_results[action_idx]['infos'].append(result['info'])
                action_results[action_idx]['rewards'].append(result['reward'])
        
        # Aggregate results for each action
        final_observations = []
        final_rewards = []
        final_terminateds = []
        final_infos = []
        
        for action_idx in range(len(normalized_actions)):
            if action_idx in action_results:
                if enable_pvt:
                    # PVT simulation: select worst case (minimum reward)
                    rewards = np.array(action_results[action_idx]['rewards'])
                    worst_idx = np.argmin(rewards)
                    final_observations.append(action_results[action_idx]['observations'][worst_idx])
                    final_rewards.append(rewards[worst_idx])
                    final_terminateds.append(rewards[worst_idx] >= 0)
                    
                    # Build complete PVT information
                    info = action_results[action_idx]['infos'][worst_idx].copy()
                    
                    # Generate performance tables for each corner
                    tables = {i: self._generate_performance_table(info_i) for i, info_i in enumerate(action_results[action_idx]['infos'])}
                    
                    # Add detailed PVT conditions
                    pvt_info = {'index': worst_idx, **dict(zip(['proc', 'vdd', 'temp'], pvt_conditions[worst_idx]))}
                    info['pvt_selected_corner'] = pvt_info
                    
                    # Add PVT context information
                    info['pvt_context'] = {
                        'aggregation': 'worst',
                        'per_corner': {i: action_results[action_idx]['infos'][i] for i in range(len(action_results[action_idx]['infos']))},
                        'corners': list(range(len(action_results[action_idx]['infos']))),
                        'rewards_per_corner': list(action_results[action_idx]['rewards']),
                        'tables': tables
                    }
                    
                    # Add additional information
                    info['real_action'] = list(normalized_actions[action_idx])
                    info['extra_corner'] = {
                        'index': 0,
                        'folder': 'tt',
                        'reward': action_results[action_idx]['rewards'][0],
                        'performance': action_results[action_idx]['infos'][0]
                    }
                    
                    final_infos.append(info)
                else:
                    # Non-PVT simulation: take direct results
                    final_observations.append(action_results[action_idx]['observations'][0])
                    final_rewards.append(action_results[action_idx]['rewards'][0])
                    final_terminateds.append(action_results[action_idx]['rewards'][0] >= 0)
                    final_infos.append(action_results[action_idx]['infos'][0])
            else:
                # Handle failed tasks
                final_observations.append(None)
                final_rewards.append(-1.0)
                final_terminateds.append(False)
                final_infos.append({'error': 'All simulation tasks failed'})
        
        # Convert to numpy arrays and return
        return (
            np.array(final_observations),
            np.array(final_rewards),
            np.array(final_terminateds),
            np.array([False]*len(final_terminateds)),
            final_infos
        )
    
    def single_simulation(self, sim_dir, individual_params, action_idx, pvt_condition):
        """Execute single simulation, supporting parallel tasks.
        
        Args:
            sim_dir: Simulation directory
            individual_params: Action parameters
            action_idx: Action index
            pvt_condition: PVT condition (proc, volt, temp)
        """
        try:
            observation, info, reward = self.do_simulation(sim_dir, individual_params)
            return {
                'action_idx': action_idx,
                'observation': observation,
                'info': info,
                'reward': reward,
                'pvt_condition': pvt_condition
            }
        except Exception as e:
            print(f"Simulation error in {sim_dir}: {str(e)}")
            return None

    def _get_info(self, sim_results, individual_params):
        '''Evaluate the performance'''

        performance = self.circuit_config['performance']
        CL = self.circuit_config['PARAM_CLOAD']
        ''' DC '''
        dc_results = sim_results.dc(file_name=os.path.basename(self.path['dc_results_path']))
        TC = 10 if dc_results is None else dc_results[1][1]
        Power = 10 if dc_results is None else dc_results[2][1]
        vos_1 = 10 if dc_results is None else dc_results[3][1]
        vos = abs(vos_1)
             
        TC_score = -1.0 if TC < 0 else self._score_lower_better(TC, performance['TC']['target'])
        Power_score = -1.0 if Power < 0 else self._score_lower_better(Power, performance['Power']['target'])
        vos_score = self._score_lower_better(vos, performance['vos']['target'])

        ''' AC ''' 
        ac_results = sim_results.ac(file_name=os.path.basename(self.path['ac_results_path']))
        cmrrdc = 0 if ac_results is None else ac_results[1][1]
        PSRP = 0 if ac_results is None else ac_results[2][1]
        PSRN = 0 if ac_results is None else ac_results[3][1]
        dcgain = 0 if ac_results is None else ac_results[4][1]
        
        
        cmrrdc_score = -1.0 if cmrrdc > 0 else self._score_lower_better(cmrrdc, performance['cmrrdc']['target'])
        PSRP_score = -1.0 if PSRP > 0 else self._score_lower_better(PSRP, performance['PSRP']['target'])
        PSRN_score = -1.0 if PSRN > 0 else self._score_lower_better(PSRN, performance['PSRN']['target'])
        PSRR, PSRR_score = (float('nan'), -1.0) if PSRP > 0 or PSRN > 0 else (max(PSRP, PSRN), self._score_lower_better(max(PSRP, PSRN), performance['PSRR']['target']))

        # dcgain: higher is better (dB)
        if dcgain > 0:
            dcgain_score = self._score_higher_better(dcgain, performance['dcgain']['target'])
            GBW_PM_results = sim_results.GBW_PM(file_name=os.path.basename(self.path['GBW_PM_path']))
            # Handle case where GBW_PM_results is empty
            if GBW_PM_results is None:
                GBW = 0
                GBW_score = -1.0
                phase_margin = 0
                phase_margin_score = -10
                FOMS = 0
                FOMS_score = -1.0
            else:
                GBW = GBW_PM_results[1][1]
                GBW_score = self._score_higher_better(GBW, performance['GBW']['target'])
                FOMS = (GBW * 1e-6 * CL) / Power
                FOMS_score = self._score_higher_better(FOMS, performance['FOMS']['target']) 
                phase_margin = GBW_PM_results[2][1]
                # Calculate phase margin score
                phase_margin_score = 0.0 if 45 <= phase_margin <= 90 else \
                                     (90 - phase_margin) / 30 if 90 < phase_margin < 120 else \
                                     (phase_margin - 45) / 45 if 0 < phase_margin < 45 else \
                                     -10
                # reward2
                # if 45 <= phase_margin <= 90:
                #     phase_margin_score = 0
                # elif 90 < phase_margin < 120:
                #     phase_margin_score = self._score_lower_better(phase_margin, 90)
                # elif 0 < phase_margin < 45:
                #     phase_margin_score = self._score_higher_better(phase_margin, 45)
                # else:
                #     phase_margin_score = -10
        else:
            dcgain_score = -1.0
            GBW = 0
            GBW_score = -1.0
            FOMS = 0
            FOMS_score = -1.0
            phase_margin = 0
            phase_margin_score = -10
      
        """ Tran """
        tran_results = sim_results.tran(file_name=os.path.basename(self.path['tran_results_path']))
        # Check if tran_results are valid
        if tran_results is None:
            # No valid tran results, assign poor values
            sr, FOML, settlingTime = 0, 0, 10
        else:
            # Valid tran results, continue normal calculation
            sr = (tran_results[1][1] + tran_results[2][1]) / 2
            FOML = (sr * CL) / Power
            
            """ setting_time """
            d0 = 0.01
            time_data, vin_data, vout_data = sim_results.extract_tran_data(file_name=os.path.basename(self.path['tran_dat_path']))
            if time_data is None:
                return None
            
            d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n = sim_results.analyze_amplifier_performance(vin_data, vout_data, time_data, d0)
            d0_settle, d1_settle, d2_settle = abs(d0_settle), abs(d1_settle), abs(d2_settle)
            SR_p, SR_n = abs(SR_p), abs(SR_n)
            settlingTime_p, settlingTime_n = abs(settling_time_p), abs(settling_time_n)
        
            d0_settle = 10 if math.isnan(d0_settle) else d0_settle
            if math.isnan(d1_settle) or math.isnan(d2_settle):
                if math.isnan(d1_settle): d0_settle += 10
                if math.isnan(d2_settle): d0_settle += 10
                d_settle = d0_settle
            else:
                d_settle = max(d0_settle, d1_settle, d2_settle)

            SR = -d_settle if math.isnan(SR_p) or math.isnan(SR_n) else min(SR_p, SR_n)
            settlingTime = d_settle if math.isnan(settlingTime_p) or math.isnan(settlingTime_n) else max(settlingTime_p, settlingTime_n)

        sr_score = self._score_higher_better(sr, performance['sr']['target'])
        FOML_score = self._score_higher_better(FOML, performance['FOML']['target'])
        settlingTime_score = self._score_lower_better(settlingTime, performance['settlingTime']['target'])
        if not np.isfinite(phase_margin):
            pm_violation = 1.0
        elif 45.0 < phase_margin < 90.0:
            pm_violation = 0.0
        elif phase_margin <= 45.0:
            pm_violation = (45.0 - phase_margin) / 45.0
        else:
            pm_violation = (phase_margin - 90.0) / 45.0

        """ Active Area """
        Active_Area = self._calculate_area(individual_params)
        # Active_Area: smaller is better
        Active_Area_score = self._score_lower_better(Active_Area, performance['Active_Area']['target'])
        
        """ FOM_amp """
        def indicator_function(condition):
            return 1 if condition else 0

        indicator_TC = indicator_function(TC < performance['TC']['baseline'])
        # indicator_vos = indicator_function(vos < performance['vos']['baseline'])

        if settlingTime > 0 and Active_Area > 0 and TC > 0 and vos > 0:
            FOM_AMP = ((PSRR/performance['PSRR']['baseline']) * (cmrrdc/performance['cmrrdc']['baseline']) * (dcgain/performance['dcgain']['baseline']) * (FOMS/performance['FOMS']['baseline']) * \
                           (FOML/performance['FOML']['baseline'])) * ((performance['settlingTime']['baseline']/settlingTime) * (performance['Active_Area']['baseline']/Active_Area)) * \
                           ((performance['TC']['baseline']/TC) * indicator_TC) \
                            # * ((performance['vos']['baseline']/vos) * indicator_vos)
        else:
            FOM_AMP = 0
        
        constraint_reward = (
            phase_margin_score
            + dcgain_score
            + PSRP_score
            + PSRN_score
            + cmrrdc_score
            + settlingTime_score
        )

        base_constraint_reward = float(constraint_reward)
        base_FOML_score = float(FOML_score)
        base_FOMS_score = float(FOMS_score)
        base_Active_Area_score = float(Active_Area_score)

        bonus_metrics = set(self.post_target_bonus_config.get('metrics', ['FOML', 'FOMS', 'Active_Area']))
        bonus_scale = float(self.post_target_bonus_config.get('scale', 1.0))
        bonus_max = float(self.post_target_bonus_config.get('max_per_metric', 1.0))
        bonus_enabled = bool(self.post_target_bonus_config.get('enabled', False))
        all_targets_satisfied = (
            base_constraint_reward >= -1e-12
            and base_FOML_score >= -1e-12
            and base_FOMS_score >= -1e-12
            and base_Active_Area_score >= -1e-12
        )

        FOML_bonus = 0.0
        FOMS_bonus = 0.0
        Active_Area_bonus = 0.0
        bonus_active = bool(bonus_enabled and all_targets_satisfied)
        if bonus_active:
            if 'FOML' in bonus_metrics:
                FOML_bonus = self._bonus_higher_better(
                    FOML,
                    performance['FOML']['target'],
                    scale=bonus_scale,
                    max_bonus=bonus_max,
                )
            if 'FOMS' in bonus_metrics:
                FOMS_bonus = self._bonus_higher_better(
                    FOMS,
                    performance['FOMS']['target'],
                    scale=bonus_scale,
                    max_bonus=bonus_max,
                )
            if 'Active_Area' in bonus_metrics:
                Active_Area_bonus = self._bonus_lower_better(
                    Active_Area,
                    performance['Active_Area']['target'],
                    scale=bonus_scale,
                    max_bonus=bonus_max,
                )

        FOML_score = base_FOML_score + FOML_bonus
        FOMS_score = base_FOMS_score + FOMS_bonus
        Active_Area_score = base_Active_Area_score + Active_Area_bonus
        reward = constraint_reward + FOML_score + FOMS_score + Active_Area_score
        reward_vector = np.array(
            [constraint_reward, FOML_score, FOMS_score, Active_Area_score],
            dtype=np.float32,
        )
        objective_rewards = {
            'constraint_reward': float(constraint_reward),
            'FOML_score': float(FOML_score),
            'FOMS_score': float(FOMS_score),
            'Active_Area_score': float(Active_Area_score),
            'PM_violation': float(pm_violation),
        }

        return {
            'phase_margin (deg)': phase_margin,
            'dcgain': dcgain,
            'PSRP': PSRP,
            'PSRN': PSRN,
            'PSRR': PSRR,
            'cmrrdc': cmrrdc,
            'vos': vos,
            'TC': TC,
            'setting_time': settlingTime,
            'FOML': FOML,
            'FOMS': FOMS,
            'Active Area': Active_Area,
            'Power': Power,
            'GBW': GBW,
            'sr': sr,
            'FOM_AMP': FOM_AMP,
            'phase_margin_score': phase_margin_score,
            'dcgain_score': dcgain_score,
            'PSRR_score': PSRR_score,
            'PSRP_score': PSRP_score,
            'PSRN_score': PSRN_score,
            'cmrrdc_score': cmrrdc_score,
            'vos_score': vos_score,
            'TC_score': TC_score,
            'settlingTime_score': settlingTime_score,
            'FOML_score': FOML_score,
            'FOMS_score': FOMS_score,
            'Active_Area_score': Active_Area_score,
            'Power_score': Power_score,
            'GBW_score': GBW_score,
            'sr_score': sr_score,
            'PM_violation': pm_violation,
            'constraint_reward': constraint_reward,
            'reward_vector': reward_vector.tolist(),
            'objective_rewards': objective_rewards,
            'reward_components': {
                'reward': float(reward),
                'constraint_reward': float(constraint_reward),
                'FOML_score': float(FOML_score),
                'FOMS_score': float(FOMS_score),
                'Active_Area_score': float(Active_Area_score),
                'PM_violation': float(pm_violation),
                'base_constraint_reward': float(base_constraint_reward),
                'base_FOML_score': float(base_FOML_score),
                'base_FOMS_score': float(base_FOMS_score),
                'base_Active_Area_score': float(base_Active_Area_score),
                'FOML_bonus': float(FOML_bonus),
                'FOMS_bonus': float(FOMS_bonus),
                'Active_Area_bonus': float(Active_Area_bonus),
                'bonus_active': bool(bonus_active),
            },
            'post_target_bonus_active': bool(bonus_active),
            'post_target_bonus': {
                'FOML_bonus': float(FOML_bonus),
                'FOMS_bonus': float(FOMS_bonus),
                'Active_Area_bonus': float(Active_Area_bonus),
                'scale': float(bonus_scale),
                'max_per_metric': float(bonus_max),
            },
            'reward': reward,
        }
        
    def _get_obs(self, sim_results, individual_params):
        """Build observation matrix"""
        op_results = sim_results.dcop(file_name=os.path.basename(self.path['op_results_path']))
        # Pick some .OP parameters from the dictionary
        try:
            with open(f'{self.path["op_mean_std_path"]}', 'r') as f:
                op_mean_std = json.load(f)
            op_mean = np.array([op_mean_std['OP_M_mean'][key] for key in ['id', 'gm', 'gds', 'vth', 'vdsat', 'vds', 'vgs']])
            op_std = np.array([op_mean_std['OP_M_std'][key] for key in ['id', 'gm', 'gds', 'vth', 'vdsat', 'vds', 'vgs']])
        except Exception as e:
            print('You need to run <_random_op_sims> to generate mean and std for transistor .OP parameters')
            return None
        
        # Calculate normalized transistor parameters
        transistor_norm_params = {}
        for key, op_data in op_results.items():
            if key.lower().startswith('m'):  # Support both upper and lower case M for transistor names
                # Extract all required parameters and normalize
                op_values = np.array([op_data['id'], op_data['gm'], op_data['gds'], 
                                     op_data['vth'], op_data['vdsat'], op_data['vds'], op_data['vgs']])
                op_values_norm = (op_values - op_mean) / op_std
                transistor_norm_params[key] = op_values_norm

        row_formats = self.circuit_config['graph']['observation_matrix']
        observation = []
        # Process all rows defined in row_formats (including transistors and other components)
        for row_key, row_format in row_formats.items():
            if row_key.startswith('M') and row_key in transistor_norm_params:
                first_non_zero_idx = next((idx for idx, val in enumerate(row_format) if val != 0), None)
                if first_non_zero_idx is not None:
                    row = [0.0] * first_non_zero_idx + transistor_norm_params[row_key].tolist()
                    observation.append(row)
            elif row_key.startswith('C'):
                row = list(row_format)
                L_C = 30
                W_C = 30
                cap_const = (L_C * W_C * 2e-15 + (L_C + W_C) * 0.38e-15)
                for idx, param_name in enumerate(row_format):
                    if isinstance(param_name, str) and row_key in op_results and param_name in op_results[row_key]:
                        value = op_results[row_key][param_name]
                        constraint_key = f"M_{row_key.upper()}"
                        if constraint_key in self.paraBound:
                            M_low, M_high = self.paraBound[constraint_key]['min'], self.paraBound[constraint_key]['max']
                            # Normalize all passive components
                            value = ActionNormalizer(action_space_low=M_low * cap_const, 
                                                    action_space_high=M_high * cap_const).reverse_action(value)
                        row[idx] = value
                observation.append(row)
            elif row_key.startswith('R'):
                row = list(row_format)
                M_R_key = f"M_{row_key.upper()}"
                
                Rsheet = 1112.4
                L_R = 3.0
                W_R = 0.35
                for idx, param_name in enumerate(row_format):
                    if isinstance(param_name, str) and M_R_key in self.paraBound:
                        M_low, M_high = self.paraBound[M_R_key]['min'], self.paraBound[M_R_key]['max']
                        R_low = Rsheet * L_R / W_R / M_high
                        R_high = Rsheet * L_R / W_R / M_low
                        M_R = individual_params[M_R_key]
                        R_value = Rsheet * L_R / W_R / M_R
                        value = ActionNormalizer(action_space_low=R_low, action_space_high=R_high).reverse_action(R_value)
                        row[idx] = value
                observation.append(row)

            else:
                row = list(row_format)
                for idx, param_name in enumerate(row_format):
                    if isinstance(param_name, str) and row_key in op_results and param_name in op_results[row_key]:
                        row[idx] = op_results[row_key][param_name]
                observation.append(row)
        
        observation = np.clip(np.array(observation), -5, 5)
        return observation

    def _run_single_random_sim(self, task_args):
        """Run a single random simulation and return OP results
        
        Args:
            task_args: Tuple containing simulation directory and device types list
        """
        sim_dir, device_types = task_args
        try:
            self._do_simulation(sim_dir)
            sim_results = OutputParser2(self.ckt_hierarchy, sim_dir)
            dcop = sim_results.dcop(file_name=os.path.basename(self.path['op_results_path']))
            
            if dcop:
                device_data = {dtype: [] for dtype in device_types}
                for key in dcop:
                    dtype = key[0].upper()
                    if dtype in device_data:
                        device_data[dtype].append(np.array([dcop[key][item] for item in dcop[key]]))
                
                return {dtype: np.array(data) for dtype, data in device_data.items()}
        except Exception as e:
            print(f"Error in random simulation {sim_dir}: {e}")
        return None

    def _init_random_sim(self, max_sims=100):
        '''
        This is NOT the same as the random step in the agent, here is basically 
        doing some completely random design variables selection for generating
        some device parameters for calculating the mean and variance for each
        .OP device parameters (getting a statistical idea of, how each ckt parameter's range is like'), 
        so that you can do the normalization for the state representations later.
        '''
        print(f"Starting {max_sims + 1} random simulations to generate OP parameter statistics...")
        # Initialize result containers
        device_types = ['M', 
                        # 'R', 'C', 'V', 'I'
                        ]
        op_results = {dtype: [] for dtype in device_types}
        batch_size = 20
        total_sims = max_sims + 1
        
        # Process simulations in batches
        for start_idx in range(0, total_sims, batch_size):
            end_idx = min(start_idx + batch_size, total_sims)
            print(f'* Random simulation batch #{start_idx} to #{end_idx - 1} *')
            
            # Generate random actions for this batch
            batch_actions = np.random.uniform(
                self.action_space_low, 
                self.action_space_high, 
                (end_idx - start_idx, len(self.action_space_low))
            )
            
            # Prepare simulation tasks
            tasks = []
            for i, action in enumerate(batch_actions):
                sim_dir = os.path.join(self.base_sim_dir, f'random_{start_idx + i}')
                os.makedirs(sim_dir, exist_ok=True)
                # Copy required files
                files_to_copy = [os.path.basename(self.path['ACDC_cir_path']), os.path.basename(self.path['Tran_cir_path'])]
                self._prepare_action_files(sim_dir, files_to_copy)
                # Write parameters
                params = self._parse_action_to_params(action)
                self._write_vars_file(sim_dir, params)
                tasks.append(sim_dir)
            
            # Execute tasks in parallel - using class method instead of local function
            num_processes = min(len(tasks), multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=num_processes) as pool:
                batch_results = pool.map(self._run_single_random_sim, [(sim_dir, device_types) for sim_dir in tasks])
            
            # Collect valid results
            for result in batch_results:
                if result:
                    for dtype in device_types:
                        op_results[dtype].append(np.array(result[dtype]))
        
        # Calculate mean and standard deviation for each device type
        op_mean_std = {}
        params_map = {
            'M': self.params_mos,
            # 'R': self.params_r,
            # 'C': self.params_c,
            # 'V': self.params_v,
            # 'I': self.params_i
        }
        
        for dtype in device_types:
            data = op_results[dtype]
            if data and len(data) > 0:
                
                combined_data = []
                for sim_data in data:
                    if isinstance(sim_data, np.ndarray) and sim_data.size > 0:
                        combined_data.append(sim_data)
                
                if combined_data:
                    
                    stacked_data = np.vstack(combined_data)
                    mean = np.mean(stacked_data, axis=0)
                    std = np.std(stacked_data, axis=0)
                    param_names = params_map[dtype]
                    
                    op_mean_std[f'OP_{dtype}_mean'] = {param_names[idx]: mean[idx] for idx in range(len(param_names))}
                    op_mean_std[f'OP_{dtype}_std'] = {param_names[idx]: std[idx] for idx in range(len(param_names))}
        
        OP_M_mean_std = {
            'OP_M_mean': op_mean_std['OP_M_mean'],         
            'OP_M_std': op_mean_std['OP_M_std']
            }

        # Save results to the path specified in configuration
        save_path = self.path['op_mean_std_path']
        with open(save_path, 'w') as file:
            json.dump(OP_M_mean_std, file)
