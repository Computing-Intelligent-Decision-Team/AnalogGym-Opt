> GitHub 发布版的数据文件采用 `.gz` 压缩。先运行 `python scripts/unpack.py --format sqlite` 以使用查询脚本；需要 CSV 或全部 JSONL 时分别选择 `csv` / `jsonl` / `all`。无需 MATLAB 即可读取本包。

# 模拟电路性能数据库（可检索版）

已从 73 个 MATLAB 性能表转换出 **358,234 条记录、18 个拓扑**。原始数据库未修改。

## 给大模型怎么用

先提供 **LLM_START_HERE.md** 和 **FIELD_GUIDE.md**。需要全量统计或筛选时，让能执行代码的大模型查询 SQLite；普通聊天可提供选中拓扑的摘要和筛选后的小份 JSONL。全库约 36 万行，不适合一次性粘贴到聊天窗口。

## 文件

- `data/performance.sqlite`：完整性能表、来源表、索引和未触发规则视图，适合精确数值检索。
- `data/performance.csv`：完整平面表，UTF-8，保留源字段名与数值，方便 Python/Excel。
- `data/by_topology/*.jsonl`：18 份按拓扑拆分的记录，每行一个 JSON 对象，适合按需读取或构建检索服务。
- `topology_index.json`、`summaries/*.md`：目录索引、记录数、条件分组、数值分布。
- `source_manifest.json`：原始路径、表名、行数、SHA-256、配套 TD/TBM 文件位置。
- `field_dictionary.json`、`FIELD_GUIDE.md`：字段解释、单位状态和核查规则。
- `audit/QUALITY_REPORT.md`、`audit/quality_report.json`：清理计数与限制；`audit/verification.json` 为验证结果。
- `examples/sample_records.jsonl`：每个源表首/中/末行示例，共 219 行，供查看结构，不代表总体分布。

## 查询示例

在本文件夹打开终端，使用 Python 3（查询只依赖标准库）：

```powershell
python scripts/query.py --sql "SELECT record_id, topology, gain, gbw, pm, ivdd_27 FROM performance_unflagged WHERE tech_node_label=180 AND gain>=80 AND pm>=60 ORDER BY ivdd_27 ASC" --limit 10
python scripts/query.py --sql "SELECT topology, COUNT(*) AS n FROM performance GROUP BY topology ORDER BY n DESC" --limit 30
python scripts/query.py --sql "SELECT record_id, SR, FOML, quality_flags FROM performance WHERE SR<0" --limit 5
python scripts/query.py --sql "SELECT * FROM performance_unflagged WHERE topology='Fan_SMC_Pin_3' AND tech_node_label=22" --limit 20 --output selected_records.jsonl
```

使用 Python 3 运行上述命令。查询脚本默认最多返回 20 行，上限 1000。第一条命令只是检索演示，不是推荐设计；示例中的阈值没有被用于全库删选。SQL 中 `index` 如需引用应写作 `"index"`。

## 清理结果

- 全部 358,234 行保留；同一源表内完全重复行 0 条；非有限数值 0 个。
- 125,974 行触发至少一项核查规则，232,260 行未触发。标记不会删除记录。
- 保留负数和极端值，不擅自插补、缩尾、归一化或单位换算。
- `chip_area` 的源处理包含开平方；建立时间、噪声、温度系数的单位仍有不确定性，详见字段说明。

## 重新生成

`scripts/export_mat_tables.m` 用本机 MATLAB 只读加载源表，导出普通数值数组到 `_work`；`scripts/build_dataset.py` 使用 numpy/scipy 生成本包。应在新目录运行，避免覆盖已交付数据库。示例入口见 `scripts/rebuild.ps1`。交付压缩包不含 `_work` 中间文件。

## 可选：合并重复性能画像

`performance_unique_profiles` 视图在同一来源文件内，将 17 个性能字段完全相同的观测合并展示，选原表行号最小者代表，并保留 `observation_count`。共有 **317,619 组**性能画像，覆盖原始 **358,234 条**观测；重复观测数 **40,615**。不同代数和个体的原始记录仍全部在主表。

这只是相同性能值的分组，不证明晶体管参数或电路设计相同。适合减少检索结果中的重复行，不应把组数称为独立设计数。

```sql
SELECT record_id, topology, gain, gbw, pm, ivdd_27, observation_count
FROM performance_unique_profiles
WHERE quality_flag_count=0 AND tech_node_label=180 AND gain>=80 AND pm>=60
ORDER BY ivdd_27 ASC
LIMIT 10;
```
