# STEP08：分别手动运行敏感性与验证分析

该步骤不负责PVC影像生成；影像及ROI特征先按STEP04的独立PVC子步骤准备。
每个命令只做指定分析，没有自动遍历其他分支、恢复任务或改参数重试。
不加`--execute`仅预检和显示计划，不运行模型。

```sh
python STEP08_sensitivity_analyses/run_sensitivity.py ablation --input /work/step05/model_input.json --output /work/ablation --execute
python STEP08_sensitivity_analyses/run_sensitivity.py scanner --input /work/step05/model_input.json --metadata /private/metadata.csv --output /work/scanner --execute
python STEP08_sensitivity_analyses/run_sensitivity.py development-bootstrap --input /work/step05/model_input.json --output /work/development --execute
```

- 消融：完整CP以及去丘脑、去纹状体/苍白球、去两者三模型，**分别重新调参**，共用已保存分折；不是把既有系数置零。
  差值定义为CP减去消融模型，AUC正、Brier负表示保留该区域有利。
- 设备分层：`metadata.csv`需`study_id,scanner`，两种实际设备名称字符串；按名称词典序分别留出设备。
  比较C与CP。这是同中心设备迁移检查，不是外部临床验证。
- 完整开发bootstrap：固定500次、seed20260921；每次重新调参/拟合，重复抽中的同一人的复制记录只能位于同一个内层折。
  它校正完整数据表观开发表现，不是校正已报告OOF AUC。失败抽样不增加替代抽样。

`--timeout-seconds`只控制时限，不改拟合定义；长时运行可在确认资源后自行加大。
输出`results_PRIVATE.json`和运行状态；逐例预测及bootstrap稳定性区间需要受控保存。

## 处理敏感性的14个模型

`processing_prediction_features.json`格式为`study_ids`及`variants`，七个矩阵均为N×6，
列顺序`settings.pet`。必须包含`same_seg_noPVC,pvc54,pvc4,pvc6,wholeBrain,cerebellarCortex,petprep`。
此处不再放`original`矩阵：原始C/CP/CPT由STEP07本来的引擎提供。

```sh
python STEP08_sensitivity_analyses/run_sensitivity.py processing-prediction --input /work/step05/model_input.json --processing-features /private/processing_prediction_features.json --output /work/processing_predictions --execute
python STEP08_sensitivity_analyses/evaluate_processing_predictions.py --input /work/step05/model_input.json --primary-predictions /work/step07/predictions.json --processing-results /work/processing_predictions/results_PRIVATE.json --output /work/processing_comparisons
```

14个模型使用原处理敏感性引擎，不替换为主模型包装器。第二条命令计算全部28个预定配对比较，
保留原seed20260910与2000次固定个体平均OOF概率重抽样；两项差值都为a减b，方向与部分主分析展示不同。
它不重新训练分类器。必须完整提供14份结果、三组主预测及一致的分折指纹，不能只评价表现好的处理条件。

人工检查模型覆盖、对照方向、全部比较及拟合来源，确认后再进入STEP10。
