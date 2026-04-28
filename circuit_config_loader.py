import os
import yaml
from typing import Dict, Any, List

class CircuitConfigLoader:
    """电路配置加载器，用于读取和解析外部配置文件，包含权重提取功能"""

    def __init__(self, config_dir: str = "circuit_configs", load_all: bool = False):
        """初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
            load_all: 是否在初始化时加载所有配置文件，默认为False（按需加载）
        """
        self.config_dir = os.path.abspath(config_dir)
        self._circuit_configs: Dict[str, Dict[str, Any]] = {}
        self._available_circuits = None
        
        # 确保配置目录存在
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            self._available_circuits = []
        
        # 只在需要时加载所有配置
        if load_all:
            self._load_all_configs()
        else:
            # 只扫描可用电路名称，不加载具体配置
            self._scan_available_circuits()

    def _scan_available_circuits(self) -> None:
        """扫描配置目录下的所有YAML文件，记录可用的电路名称"""
        self._available_circuits = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                circuit_name = os.path.splitext(filename)[0]
                self._available_circuits.append(circuit_name)
    
    def _load_all_configs(self) -> None:
        """加载配置目录下的所有配置文件"""
        self._scan_available_circuits()
        
        # 遍历所有可用电路，加载配置
        for circuit_name in self._available_circuits:
            try:
                self._load_single_config(circuit_name)
            except Exception as e:
                print(f"加载配置文件 {circuit_name}.yaml 失败: {str(e)}")
    
    def _load_single_config(self, circuit_name: str) -> Dict[str, Any]:
        """加载单个电路配置文件
        
        Args:
            circuit_name: 电路名称
            
        Returns:
            加载的配置字典
            
        Raises:
            ValueError: 当电路配置不存在或格式错误时
        """
        # 检查配置是否已加载
        if circuit_name in self._circuit_configs:
            return self._circuit_configs[circuit_name]
        
        # 确定配置文件路径（尝试.yaml和.yml扩展名）
        yaml_path = os.path.join(self.config_dir, f"{circuit_name}.yaml")
        yml_path = os.path.join(self.config_dir, f"{circuit_name}.yml")
        
        if os.path.exists(yaml_path):
            config_path = yaml_path
        elif os.path.exists(yml_path):
            config_path = yml_path
        else:
            # 如果缓存中有可用电路列表，检查电路名称是否有效
            if self._available_circuits is not None and circuit_name not in self._available_circuits:
                raise ValueError(f"未知电路: {circuit_name}，可用电路: {self.get_available_circuits()}")
            raise ValueError(f"配置文件不存在: {circuit_name}.yaml 或 {circuit_name}.yml")
        
        # 加载和处理配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证配置完整性
        self._validate_config(config, circuit_name)
        
        config['paths'] = self._prepare_paths(config)

        # 归一化性能约束中的权重
        self._normalize_performance_weights(config)
        
        # 保存配置（添加文件路径信息和电路名称）
        config['config_path'] = config_path
        config['name'] = circuit_name
        config['category'] = circuit_name.split('_')[0] if '_' in circuit_name else 'default'
        
        # 存储到缓存中
        self._circuit_configs[circuit_name] = config
        
        return config

    def _validate_config(self, config: Dict[str, Any], circuit_name: str) -> None:
        """验证配置是否完整"""
        required_fields = [
            'base_dir', 'device',
            'performance', 'file_paths'
        ]
        # 检查必填字段
        for field in required_fields:
            if field not in config:
                raise ValueError(f"电路 {circuit_name} 配置缺少必要字段: {field}")

    def get_available_circuits(self) -> List[str]:
        """获取所有可用电路名称"""
        if self._available_circuits is None:
            self._scan_available_circuits()
        return self._available_circuits.copy()

    def _convert_str_numbers(self, obj):
        """递归转换字符串数字为数值"""
        if isinstance(obj, dict):
            return {k: self._convert_str_numbers(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_str_numbers(item) for item in obj]
        elif isinstance(obj, str):
            # 尝试转 float，再转 int（如果整数）
            try:
                num = float(obj)
                return int(num) if num.is_integer() else num
            except ValueError:
                return obj
        else:
            return obj

    def get_circuit_config(self, circuit_name: str) -> Dict[str, Any]:
        """获取指定电路的配置，包含提取的权重信息、设计变量初始值和动作空间范围
        
        如果配置尚未加载，则按需加载
        """
        # 按需加载配置
        config = self._load_single_config(circuit_name)
        
        # 返回配置的副本并转换字符串数字
        config_copy = config.copy()
        return self._convert_str_numbers(config_copy)
        
    def get_simulation_dir(self, circuit_name: str) -> str:
        """获取仿真文件目录"""
        config = self.get_circuit_config(circuit_name)
        return config.get('base_dir', '')

    def _prepare_paths(self, config):
        """准备仿真所需的路径信息
        
        Args:
            config: 电路配置字典
            
        Returns:
            构建好的路径字典
        """
        # 获取项目根目录（基于脚本所在位置）
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 获取基础目录和文件路径配置
        base_dir = config.get('base_dir', '')
        file_paths = config['file_paths']
        # 构建完整路径
        paths = {}
        for key, rel_path in file_paths.items():
            if base_dir:
                if base_dir.startswith('../'):
                    # 移除../前缀，获取相对路径部分
                    base_dir_without_dotdot = base_dir[3:]
                else:
                    base_dir_without_dotdot = base_dir
                combined_path = os.path.join(project_root, base_dir_without_dotdot, rel_path)
            else:
                combined_path = os.path.join(project_root, rel_path)
            # 规范化最终路径
            paths[key] = os.path.normpath(combined_path)
        
        return paths
        
    def _normalize_performance_weights(self, config: Dict[str, Any]) -> None:
        """归一化性能约束中的权重，确保所有权重和为1"""
        performance_constraints = config.get('performance', {})
        
        # 计算所有权重的总和
        total_weight = 0
        weighted_constraints = []
        
        for key, constraint in performance_constraints.items():
            if 'weight' in constraint:
                total_weight += constraint['weight']
                weighted_constraints.append(key)
        
        # 如果总权重不为零且存在带权重的约束，则归一化
        if total_weight > 0 and weighted_constraints:
            for key in weighted_constraints:
                performance_constraints[key]['weight'] = performance_constraints[key]['weight'] / total_weight

    def reload_configs(self) -> None:
        """重新加载所有配置文件"""
        self._circuit_configs.clear()
        self._load_all_configs()
