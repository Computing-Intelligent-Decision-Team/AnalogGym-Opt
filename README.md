# GRPO Analog Circuit Optimizer (最小可执行版本)

这是一个针对模拟电路参数优化的智能体训练仓库，核心算法基于 **GRPO (Group Relative Policy Optimization)** 组相对策略优化。

## 📌 项目说明
- **最小可执行版本**：本版本已剔除了所有冗余、调试或对比性脚本（如 DDPG 对比脚本等），仅保留最核心、规范的执行文件。
- **中文注释**：关键文件和主入口脚本的注释已转换为中文，便于理解。
- **测试案例**：默认提供 `dfcfc2`（配置位于 `circuit_configs/amp_dfcfc2.yaml`）作为快速执行验证案例。

## 📂 数据集说明
> [!IMPORTANT]
> **完整数据集将在本论文（Paper）被接收 (Acceptance) 后正式上传。**
> 当前版本仅包含执行优化算法的核心仿真及逻辑框架，确保最小可执行。

## 🛠️ 环境依赖
在开始之前，请确保已安装以下依赖（推荐使用 Python 3.8+）：
- PyTorch
- NumPy
- Matplotlib
- Tabulate
- Ngspice (需要配置在系统环境变量中，用于执行底层仿真)

## 🚀 快速启动
运行主程序以对 `dfcfc2` 测试案例进行优化训练：

```bash
python main_AMP_grpo.py
```

优化结果、最优设计方案以及训练曲线数据将保存在 `training_saves/` 目录下。

## 📄 核心文件结构
- `main_AMP_grpo.py`: 顶层训练入口。
- `grpo.py`: GRPO 算法主要实现。
- `AmpEnv.py`: 模拟电路底层仿真与奖励计算环境（集成 Ngspice 交互）。
- `models.py`: Actor 网络架构定义。
- `circuit_configs/`: 存放各种电路的基础配置文件（包含 `amp_dfcfc2.yaml`）。
