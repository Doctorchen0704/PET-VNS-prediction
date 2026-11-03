# STEP05：按已核实的研究编号合并临床、结局和影像特征

环境：Python，版本见根目录环境文件。姓名、登记号保存在单独的受控Excel中；
数值分析表只通过已核实的`study_id`连接，不按姓名模糊匹配、文件夹顺序或Excel行号匹配。

## 输入

- `cohort.csv`：`study_id`一列，**每行均是已确定纳入的病例，顺序即冻结分析顺序**。
- `clinical.csv`：`study_id,age_at_implant_years,sex,epilepsy_duration_reported_years,baseline_monthly_frequency,mri_lesional_status,responder_12m`。
- `pet_features.csv`：研究编号及六个规范PET字段；`t1_features.csv`：编号及六个规范T1字段，字段名见`paper_settings()`。
- 已保存的`split_plan.json`：原研究的20次重复、外层和内层分折。主研究复现使用保存分折，不能只设置相同seed后宣称分折相同。

性别用`female/male`，病灶状态用`nonlesional/lesional`；结局必须为明确的`0/1`。
月发作频率是病史记录的每月平均频率，不是28天或30天严格事件计数。脚本仅做`log1p`，不按天数换算。
空值保留为空值，尤其不能由二分类结局反推术后发作次数或精确减幅。

如STEP04每例出一份JSON，可先分别汇总，输入文件由研究者显式列出：

```sh
python STEP05_clinical_data_assembly/aggregate_features.py --kind pet --inputs /work/caseA/pet_features.json /work/caseB/pet_features.json --output /work/pet_table
python STEP05_clinical_data_assembly/aggregate_features.py --kind t1 --inputs /work/caseA/t1_features.json /work/caseB/t1_features.json --output /work/t1_table
```

## 执行

```sh
python STEP05_clinical_data_assembly/assemble_model_input.py --cohort /private/cohort.csv --clinical /private/clinical.csv --pet /work/pet_table/pet_features.csv --t1 /work/t1_table/t1_features.csv --split-plan /private/split_plan.json --output /work/step05
```

输出`model_input.json`、`predictors.csv`、`outcomes.csv`及简短汇总。
默认逐行按原临床准备实现使用Python `math.log1p`。若复现冻结输入的精确浮点表示，
临床CSV可额外包含来源明确的`log1p_baseline_monthly_frequency`，并显式加`--preserve-frozen-log`。
该选项先检查冻结派生列与原频率变换的差不超过`1e-12`及缺失位置一致，才保留既有派生浮点值；
输出记录此模式。不得修改真实原始频率来追求末位浮点一致；普通重新整理临床数据不必使用该选项。
`--generate-splits`仅供新的数据集，不与`--split-plan`同时使用；它不代表历史分折已重现。
脚本不会自动排除任何必需病例；缺失编号、重复编号、未知分类或不合格的结局编码会报错。

## 人工检查后再继续

检查人数、应答人数、名单顺序、结局版本及六PET/六T1列含义。纳排和结局必须先在受控原始记录中确定，
不是根据此步后续模型表现调整。确认后手动进入STEP06或STEP07。
