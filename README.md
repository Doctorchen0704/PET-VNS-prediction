# PET–VNS：逐步运行的分析代码

[English guide](README_EN.md)

手动脚本版1.0.0，共用已验证的0.3.0数值函数；本轮没有更换数值算法。

这是一套供研究人员手动运行的科研脚本。每次只启动一个STEP或其中一个子步骤，
查看输出并完成相应检查后，再手动运行下一步。没有自动总控、后台续跑或自动纳排。

从本目录运行命令；软件安装见`environment/README.md`。每个STEP的README写明
输入、输出、命令和检查点。路径由命令行提供，源码不绑定某台电脑或某个账户。

| 顺序 | 目录 | 内容 | 主要环境 |
|---|---|---|---|
| 1 | STEP01_dicom2bids | DICOM转换、人工选片、BIDS目录整理 | Python＋dcm2niix |
| 2 | STEP02_T1_segmentation | 单例FreeSurfer分区、人工QC | Linux/WSL＋FreeSurfer |
| 3 | STEP03_PET_T1_registration | 单例刚体配准、2mm分析网格、人工QC | Linux/WSL＋FreeSurfer |
| 4 | STEP04_feature_extraction | PET六区域/82脑区、T1特征；独立PVC子步骤 | Python；PVC另需FreeSurfer |
| 5 | STEP05_clinical_data_assembly | 按研究编号合并临床、结局和影像表 | Python |
| 6 | STEP06_regional_associations | 固定调整模型和完整FDR校正族 | Python |
| 7 | STEP07_prediction_models | C/CP/CPT重复嵌套交叉验证；保存预测的摘要另列 | Python |
| 8 | STEP08_sensitivity_analyses | 消融、设备、bootstrap及处理敏感性，逐项运行 | Python |
| 9 | STEP09_spatial_pattern_and_atlas | PC1、分子图谱、外部空间复现及条件分析 | Python＋有权限的图谱/外部输入 |
| 10 | STEP10_tables_and_figures | 结果表、主图和补充图；可选标准脑资产生成 | Python；三维资产另需PyVista/VTK |

## 如何使用

1. 把原始数据放在代码目录以外；确认受试者对应和选定序列。
2. 按STEP01说明执行转换和整理。身份和选片由人确认，程序不猜测。
3. STEP02、03完成后实际查看分区、配准和覆盖图，在QC表记录结论。
4. 各步骤使用新的输出目录，不覆盖输入或失败目录。已有合格结果可以直接复用，
   从需要的STEP开始，不必重跑FreeSurfer。
5. 依次完成特征、数据合并、统计和作图。纳排、结局及参数以固定研究记录为准，
   不根据新运行的AUC或P值修改。

`src/pet_vns_prediction`仅是各STEP共用的普通数值函数，避免在不同脚本中重复
写同一公式；不负责调度步骤。`step_support.py`仅提供读写文件、路径和手工QC表检查。
`configs`保存区域定义，不含患者资料；人工输入模板位于各STEP目录。

## 本研究的私有输入

作者的受控交接材料另有`study_inputs_PRIVATE`，与代码目录并列，包含实际研究
名单、临床/结局表、固定分折、图谱数组和结果参照。它不属于公开代码，不能上传
公共仓库。原始DICOM另行提供，不重复压入代码包。

原始影像、分割和配准的处理在本轮整理中未重新运行。代码验证、数值复现和
人工图像QC是不同检查；具体已验证范围在交接说明中列出。部分历史处理敏感性
仅保留特征表，详见STEP06/08；外部公开队列在STEP09从固定ROI输入开始，
不把它误称为VNS应答模型的外部验证。

代码重组不改变实际分析或论文结果。
