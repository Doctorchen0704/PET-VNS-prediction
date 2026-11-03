# STEP09：代谢空间模式与公共图谱

各脚本一次只做一个明确分析，需人工选择并运行；不会下载数据、自动接续或选择更好结果。
使用固定纳入/QC名单，不能按空间相关程度改变名单。所有输入均由研究者本地提供。

## 09a PC1及患者重抽样

```sh
python STEP09_spatial_pattern_and_atlas/09a_fit_PC1.py --input /private/roi82.json --branch original --output /work/PC1
python STEP09_spatial_pattern_and_atlas/09a_fit_PC1.py --input /private/roi82.json --branch original --bootstrap --output /work/PC1_bootstrap
```

JSON字段为`values`（N×82正值）、`features`（82脑区固定顺序）、可选`ids`。
原队列重抽样另需`scanner`（N个分层标记）。不读取疗效。
原队列按最大绝对载荷正向定号；外部队列需`--branch external`，按双侧苍白球固定定号，不能为相关更高翻转符号。
默认无bootstrap，仅`--bootstrap`开展全部2,000次；不会因为某次不理想而替换重抽样。
输出`PC1_arrays.npz`及`PC1_results.json`。先检查82区/顺序、符号、特征值间隙和载荷稳定性。

## 09b 原队列分子对应和空间梯度控制

```sh
python STEP09_spatial_pattern_and_atlas/09b_molecular_spatial_tests.py --input /private/molecular_inputs.npz --order /private/atlas_order.json --mode spin --atlas-use-authorized --output /work/spin
python STEP09_spatial_pattern_and_atlas/09b_molecular_spatial_tests.py --input /private/molecular_inputs.npz --order /private/atlas_order.json --mode conditional --atlas-use-authorized --output /work/conditional
python STEP09_spatial_pattern_and_atlas/09b_molecular_spatial_tests.py --input /private/molecular_inputs_with_bootstrap.npz --order /private/atlas_order.json --mode bootstrap --atlas-use-authorized --output /work/molecular_participant_uncertainty
```

NPZ必须包含`observed68`、`receptors68x18`、`sphere_coords68x3`、`ap68`、`g168`。
order JSON必须包含`cortical_labels`（右34→左34同名配对）和`receptor_names`（与`spatial.TARGETS`相同18项）。
可供给冻结`spins`（9999×68）、`basis`、`signs`；未提供时按原固定种子产生。精确复现优先用冻结空间抽样数组。
条件检验用秩、半球、前后梯度及功能G1；残差不重新排秩，全残差空间MSR，BH18。
`--mode bootstrap`另需`bootstrap68`（固定PC1方向对齐的2000×68载荷），只计算保存重抽样下的18项原始/条件Spearman区间，不重跑参与者抽样、不新增空间P值。
sphere坐标不能代替pial前后坐标；早期零填充G1不能作为输入。

## 09c 外部独立PC1

```sh
python STEP09_spatial_pattern_and_atlas/09c_external_replication.py --input /private/external_inputs.npz --metadata /private/external_order.json --branch healthy --atlas-use-authorized --output /work/healthy
python STEP09_spatial_pattern_and_atlas/09c_external_replication.py --input /private/TLE_inputs.npz --metadata /private/TLE_order.json --branch adult_tle --atlas-use-authorized --output /work/TLE
```

共同NPZ：`raw_roi_means` N×82、`cortical_indices`68、`sphere_coords68x3`、`ap68`、`g168`、`original_loading82`。
健康分支另需`receptors68x18`；metadata为`feature_names`82和`receptor_names`18。
TLE分支需`receptors68x2`（5-HT2A、SERT顺序）；主分析Holm3、条件分支Holm2。
`--effect-only`仅用于此前固定的TLE敏感性分支，不产生空间P值。
完整核验含2000次参与者bootstrap及9999次空间抽样，可能耗时。外部PC1复现不等于VNS疗效预测的外部验证。

## 09d 替代SERT与DAT

```sh
python STEP09_spatial_pattern_and_atlas/09d_SERT_specificity.py --mode DAT --input /private/SERT_inputs.npz --n-participants 57 --atlas-use-authorized --output /work/DAT
```

NPZ：`observed68`、`bootstrap68`、`maps68x3`（主要SERT、替代SERT、DAT）、`z68x4`。
`--mode atlas`另需9999×68的`raw_null68`及`conditional_null68`，并要求2000份bootstrap。
不同队列的四项替代图谱检验必须一起运行`--mode holm4`；输入JSON含四项`labels`和`pvalues`。
DAT控制是rank5效应与参与者区间，不沿用rank4的P值。

## 09e PC1与临床/六区域的对应

```sh
python STEP09_spatial_pattern_and_atlas/09e_clinical_bridge.py --input /private/PC1_clinical_designs.npz --output /work/clinical_bridge
python STEP09_spatial_pattern_and_atlas/09e_clinical_bridge.py --input /private/PC1_clinical_designs.npz --bootstrap-input /private/PC1_clinical_bootstrap.npz --output /work/clinical_bridge_bootstrap
```

主输入NPZ为固定临床设计矩阵：`PC1_z`、`y`、`pet6`、`age`、`female`、`scanner`、
`type_primary`、`type_MRI_ASM`、`response_primary`、`response_plus_type`、`response_original5`。
PC1按原样本标准差标准化；响应矩阵第5列为PC1；类型矩阵第5/6列为focal/combined虚拟变量，参照为generalized。
原设计先固定，脚本不自动挑变量。类型为HC3联合检验，响应为无惩罚logistic的LR与profile区间；两个主临床检验Holm2。
原六区域Spearman及条件Spearman是描述性关联，不增加六项P值。

可选bootstrap NPZ包括`clr82`、`row_indices`（2000×N）、`pc1_loadings`（2000×82）、原`score_sd`。
这里复用先前PCA重抽样载荷，不重新选PC；保留不可估计次数，不用新抽样替代。
六区域的秩相关沿用最终数值修正：每位原患者只投影一次再按抽样索引重复，确保同一患者的重复副本严格同值；不同患者不进行人为舍入。
这一步不计算AUC，不把探索性关联变为新的独立验证。

## 人工检查与使用边界

受体和fsaverage资源的许可、版本、区域映射需先确认；命令中的确认参数不代替许可。
输入NPZ是分析材料，不可随公开代码上传。代码不创建健康对照或推断临床病因。
