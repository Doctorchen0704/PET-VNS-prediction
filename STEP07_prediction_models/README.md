# STEP07：固定模型的拟合与性能摘要

本页保留原C/CP/CPT三模型流程。新增的PET设备调整C/CT/CP/CPT比较使用
[独立子目录及说明](scanner_adjusted_four_models/README.md)；两条流程不能混用结果。

只读STEP05的`model_input.json`，保留受试者顺序、C/CP/CPT字段及保存的全部内外层分折。
不在这一步寻找更高AUC的参数或重新选择病例。

## A. 真正重新拟合

先查看计划，不写输出：

```sh
python STEP07_prediction_models/fit_models.py --input /work/step05/model_input.json --output /work/step07
```

确认后显式执行当前步骤：

```sh
python STEP07_prediction_models/fit_models.py --input /work/step05/model_input.json --output /work/step07 --execute --timeout-seconds 7200
```

使用原固定规格：20次重复、外层类别支持规则、3折内层；完整6×5惩罚网格；以内层Brier选参，
平局优先更小C再更小L1比例；训练折内插补/标准化；原确定性弹性网求解器和独立数值检查。
C为临床5项，CP增加六区域PET，CPT再增加六个对应T1指标。

输出`predictions.json`、`performance.json`、`conditional_intervals.json`和受控的拟合记录。
失败不改参数重试，不覆盖已有输出目录，不自动启动STEP08。

## B. 仅重新汇总已经算好的预测

```sh
python STEP07_prediction_models/evaluate_predictions.py --input /work/step05/model_input.json --predictions /private/predictions.json --output /work/step07_summary_only
```

预测JSON需`status:PASS`、`participant_ids`及`predictions`中C/CP/CPT三份20×人数概率矩阵。
此命令**不拟合模型**，也不能证明这些外部提供的预测原本没有泄漏；训练来源须另行检查。

汇总AUC来自每人20次留出概率取平均，不是某单一模型的新患者表现。2000次个体重抽样区间以固定预测为条件，
不等于500次完整开发bootstrap。真实结局、个体概率和拟合日志均不能放入公开代码仓库。
