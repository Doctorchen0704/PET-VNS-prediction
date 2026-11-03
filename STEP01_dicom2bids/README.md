# STEP01：DICOM → 选片 → BIDS整理

环境：Windows或Linux Python；dcm2niix **v1.0.20250505**。本步骤不是调用名为
`dcm2bids`的第三方包，而是保留研究实际使用的dcm2niix转换、人工选片和显式提升。
以下命令从代码包根目录运行，路径均由使用者指定。示例标识不对应真实患者。

## 输入和人工工作

原始DICOM保持只读。依据经核实的受试者对应表建立`conversion_manifest.json`，
格式见本目录示例。先完成身份、模态及来源核实，才将对应的reviewed字段改为true。
文件夹顺序、名称相似或“转换成功”都不是身份依据。

## 1．转换

```text
python STEP01_dicom2bids/01_convert_dicom.py --manifest /private/conversion_manifest.json --raw-root /data/DICOM --staging /work/staging --converter /tools/dcm2niix
```

先看计划；确认后手动再运行相同命令并加`--execute`。输出必须是全新目录。
原始参数是`-b y -ba y -z y -w 0 -f series-%s_desc-%p`。保留全部生成序列，
脚本不擅自挑选“最好看”的结果。转换日志可能含隐私，不得公开。

## 2．人工选片后生成清单

检查T1、原生PET全视野、重建协议及受试者对应关系，填写`selection.tsv`，
格式见`selection.example.tsv`。每条精确写出已选NIfTI和JSON文件，不用模糊通配符。
将review_pass填写PASS或PASS_WITH_ATTENTION；有问题时保留HOLD，不勉强提升。

```text
python STEP01_dicom2bids/02_prepare_selection.py --selection-tsv /private/selection.tsv --staging /work/staging --output /work/selected_series.json
```

该脚本只给已人工选择的序列计算哈希，不进行影像选择。PET元数据补写须另有
依据和`metadata_patch_reviewed=YES`；不得从“约60分钟”推造个体注射时间或SUV。
如需严格复现原研究，用受控资料中的冻结选片及NIfTI/侧车参照对比新转换结果，
差异先核实，不自动替换参照。

## 3．提升为BIDS目录

```text
python STEP01_dicom2bids/03_promote_to_bids.py --manifest /work/selected_series.json --staging /work/staging --bids /work/bids
```

先查看清单，再手动加`--execute`。输出为`sub-*/anat/*_T1w.nii.gz`及
`sub-*/pet/*_pet.nii.gz`和对应JSON。之后可单独运行已安装的BIDS validator：

```text
bids-validator /work/bids
```

保留报告，并区分“整理为BIDS样式”和“验证器通过”。原研究的内部组织检查
不是官方验证器通过的证明。缺失PET元数据如不能核实，应如实保留缺项和验证
限制；不得为了通过验证器编造值。转换及匿名选项也不意味着影像已可公开分享。

## 本步结束检查

确认来源和选片正确、NIfTI/JSON成对、没有错患者/错序列、无意外头部截断或
不可接受伪影。将真实结论记入人工QC表的`source_review`，然后由人决定进入STEP02。
