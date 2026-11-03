# STEP06：区域关联及完整多重比较校正

输入是STEP05的`model_input.json`与按研究编号连接的`metadata.csv`。
`metadata.csv`至少含`study_id,scanner_indicator`；0/1设备编码的参照必须在研究记录中注明。
扩展临床分析还需要：

- `epilepsy_type`：`generalized/focal/combined_generalized_and_focal`，以generalized为参照；
- `etiology_group`：`unknown/structural/other_known`，以unknown为参照。`other_known`沿用已固定的感染、免疫和代谢病因分组，不能临时按结果重分。

每次手动运行一个完整family，不按显著区域缩小校正范围：

```sh
python STEP06_regional_associations/run_associations.py --input /work/step05/model_input.json --metadata /private/metadata.csv --family primary6 --output /work/primary6
python STEP06_regional_associations/run_associations.py --input /work/step05/model_input.json --metadata /private/metadata.csv --family morphometric12 --output /work/morphometric12
python STEP06_regional_associations/run_associations.py --input /work/step05/model_input.json --metadata /private/metadata.csv --family clinical18 --output /work/clinical18
```

- `primary6`：六区域，临床5项＋设备调整；
- `morphometric12`：六区域的临床5项调整和临床5项＋设备＋对应T1调整，共12项；
- `clinical18`：基底模型分别加癫痫类型、病因及两者，共18项。

年龄、病程、log1p频率在完整关联样本中按样本SD标准化；性别和病灶为固定0/1编码。
这是关联回归的设计，不替代预测中的训练折预处理。使用HC3协方差、残差自由度t推断及完整family的BH校正。
无自动插补或静默删人；不完整设计应回到上一步核对，不可为了得到P值跳过病例。

## 处理敏感性

`processing_features.json`结构为`{"study_ids":[...],"variants":{"original":[[六值],...],...}}`。
行顺序必须与`model_input`一致；每个矩阵列顺序为`settings.pet`。
必须含`original,wholeBrain,cerebellarCortex,same_seg_noPVC,pvc54,pvc4,pvc6,petprep`全部八种。

```sh
python STEP06_regional_associations/run_associations.py --input /work/step05/model_input.json --metadata /private/metadata.csv --family processing96 --processing-features /private/processing_features.json --output /work/processing96
python STEP06_regional_associations/run_associations.py --input /work/step05/model_input.json --metadata /private/metadata.csv --family followup114 --processing-features /private/processing_features.json --output /work/followup114
```

96项为八处理×六区域×两调整；`followup114`保留18项和96项各自的校正，同时给出原实现的辅助114项校正。
5.4 mm是论文主要PVC条件，但只画5.4 mm不意味着从原96项校正族删除4/6 mm等条件。
原PetPrep部分图像中间文件有历史缺口：现存特征表可复算统计，不能据此称该影像分支已经重新提取。

输出`associations.json/csv`；其中逐例影响诊断包含敏感信息，不作为自动删人依据。人工核对完整family计数、
效应方向、参照编码与原结果后，再手动进入作图。
