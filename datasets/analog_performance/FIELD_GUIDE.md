# 字段说明

所有 `.mat` 的 VariableUnits/VariableDescriptions 为空。本次保留字段名和全部有限数值；“inferred”是解释线索，不是已验证单位。未确认单位的字段不做归一化或换算。

| 原字段 | 含义 | 建议解释单位 | 状态 | 注意事项 |
|---|---|---|---|---|
| gen | 优化代数 | 未确认/不适用 | confirmed | 源表字段；不是独立随机样本编号。 |
| index | 该代内个体编号 | 未确认/不适用 | confirmed | 与 source_id/gen 联合定位；原始编号从 0 开始。 |
| pm | 相位裕度 | degree | inferred | 表格无 VariableUnits；按字段语义解释，未做单位换算。 |
| gbw | 增益带宽积 | Hz | inferred | 按字段语义和 computeFOM.m 用法解释，未做单位换算。 |
| gain | 增益 | dB | inferred | 按字段语义及数值量级解释，未确认具体测量频率和工况聚合方式。 |
| noise | 噪声指标 | 未确认/不适用 | unresolved | 不能确定是积分噪声还是谱密度；不要擅自标为 V 或 V/sqrt(Hz)。 |
| CMRR | 共模抑制比 | dB | inferred | 具体频率和工况聚合方式尚未核实。 |
| PSRR | 电源抑制比 | dB | inferred | 不能从此表区分 PSRR+、PSRR-、测量频率或工况。 |
| SR | 压摆率 | V/s | inferred | 同包 extractLisAndCheckfile.m 解析 V/s，但尚未证明所有表都用该提取函数。负值保留并标记。 |
| ivdd_27 | 标记为 27 的电源电流 | A | inferred | 字段名暗示 27 摄氏度；不可推断其余性能列也都是 27 摄氏度或 TT。 |
| vos | 输入失调指标 | V | inferred | 按字段语义解释；保留源值，不推断符号处理或统计口径。 |
| tc | 温度系数指标 | 未确认/不适用 | unresolved | 可能涉及温漂；基准量和公式缺失，不能自动标注 ppm/degreeC。 |
| d_settle | 原始 d_settle 指标 | 未确认/不适用 | unresolved | 含义、尺度和单位未确认，不解释为秒，不与 settlingTime 自动相减。 |
| settlingTime | 建立时间指标 | 未确认/不适用 | unresolved | 部分行与 d_settle 完全相同且原脚本会过滤；统一时间单位未确认。 |
| FOMS | 小信号性能优值 | 未确认/不适用 | unresolved | 保留原值，公式及缩放需按生成程序核实。 |
| FOML | 大信号性能优值 | 未确认/不适用 | unresolved | 保留原值，公式及缩放需按生成程序核实；存在负数。 |
| chip_area | 源脚本平方根面积指标 | 未确认/不适用 | unresolved | extract_pop_data.m 执行 sqrt(abs(chip_area))；不能直接当作真实芯片面积，亦不擅自平方还原。 |
| fitness | 源优化适应度 | 未确认/不适用 | unresolved | 可能包含惩罚或目标函数；负值本身不判错，不跨任务直接排序。 |
| FOM_AW | 源自定义性能优值 | 未确认/不适用 | formula_in_source | computeFOM.m 当前公式 gbw*1500e-12/ivdd_27，固定 1500pF，不能解释为采用目录 CL 的统一 FoM。 |

## 上下文与来源

- `topology` 保留原拓扑目录名。`tech_node_label`、`vdd_label`、`vcm_label`、`cl_label`、`run_label` 从目录读取。按目录约定 tech 常指 nm、VDD/VCM 常指 V、CL 常指 pF，但不是逐条仿真条件的验证结果。
- 实际抽查到 `Cascode_Miller_Pin_2/180/1.8/0.4/10/2` 的瞬态网表使用 VCM=0.5、VDD=1.98；目录分组不能等同于实际角落条件。
- `source_id` 对应 `source_manifest.json` 的相对路径和 SHA-256。`source_row`/`row_1based` 是 MATLAB 原表行号，从 1 开始。
- `generation` 对应原字段 gen，`individual_index` 对应 index；它们保留原表 0 起始编号。
- JSONL 的 `performance` 保留其余 17 个源字段，CSV/SQLite 保留完整 19 列。
- `quality_flag_count` 为标记数，CSV/SQLite 的 `quality_flags` 用竖线分隔，JSONL 为字符串数组。空数组/空字符串只表示没有触发本次规则，不代表仿真成功、闭环稳定或所有规格达标。
- `performance_unflagged` 是 SQLite 的可选视图，只选标记数为零的记录；主表 performance 包含全部记录。

## 核查规则

- `sr_negative`：SR < 0；保留原值，视作待核查。负数不自动认定为物理压摆率，亦不改成绝对值。
- `sr_minus_one`：SR 恰为 -1；疑似失败哨兵值，尚未找到原始仿真端定义。
- `foml_negative`：FOML < 0；与负 SR 同现时不适合直接做越大越好的排序。
- `settling_fields_equal`：d_settle == settlingTime；原 filterByFOM.m 排除此类行，本次仅标记，不删除。
- `nonfinite`：原值为 NaN/+Inf/-Inf；JSON 和 SQLite 中用 null，原类型记录在异常日志。
- `key_conflict`：同一来源文件内 gen/index 相同而数值不同，保留全部并标记。
- `nonpositive_basic_metric`：gbw、noise、ivdd_27 或 chip_area 非正，待核查。
- `phase_outside_0_180`：pm 不在 [0,180]；仅作约定范围核查，不作为全规格合格判据。
