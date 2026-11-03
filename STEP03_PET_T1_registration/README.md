# STEP03：PET–T1配准与2 mm分析网格

环境：Linux/WSL FreeSurfer7.4.1。路径和受试者必须与核实后的来源/分区对应。
人工QC表至少有source_review、segmentation_review、registration_review、
coverage_review，原有警示单独记录。命令默认只显示，加`--execute`才执行当前子步。

## 1．估计刚体变换

```text
python STEP03_PET_T1_registration/01_register_pet.py --subject sub-example01 --pet /work/bids/sub-example01/pet/sub-example01_pet.nii.gz --t1 /work/freesurfer/sub-example01/mri/T1.mgz --brainmask /work/freesurfer/sub-example01/mri/brainmask.mgz --qc-table /private/manual_qc.tsv --output /work/registration/sub-example01
```

采用mri_coreg，6自由度、4线程、seed20260731、sat99.9，保存LTA、参数、成本值
及`registered_review_pet.nii.gz`。原研究固定配准参考是FreeSurfer的`T1.mgz`，
不是其他名称相近的T1处理版本；若复用
原结果，直接把该例已接受的LTA传给下一子步，不再估计。

先用Freeview等叠加T1和配准PET，逐层检查皮层轮廓、脑室、丘脑/纹状体及小脑；
确认左右、头位、轴/冠/矢状局部对齐及视野。数值成本低不等于局部对位正确。
将实际结论填入registration_review；HOLD/FAIL不进入下一步。

## 2．一次性采样到分析网格

```text
python STEP03_PET_T1_registration/02_analysis_grid.py --subject sub-example01 --pet /work/bids/sub-example01/pet/sub-example01_pet.nii.gz --t1 /work/freesurfer/sub-example01/mri/T1.mgz --brainmask /work/freesurfer/sub-example01/mri/brainmask.mgz --segmentation /work/freesurfer/sub-example01/mri/aparc+aseg.mgz --lta /work/registration/sub-example01/pet_to_T1.lta --qc-table /private/manual_qc.tsv --output /work/grid/sub-example01
```

使用原生PET与已接受LTA，而不是上一子步的展示用配准PET，避免重复插值。
T1参考2 mm/cubic、PET trilinear、标签与脑掩膜nearest-neighbor。主要分析不额外
平滑、不做PVC、不算SUV。输出：

- `reference_T1w.nii.gz`
- `directRigid_pet.nii.gz`
- `aparcAseg_dseg.nii.gz`
- `brain_mask.nii.gz`

完成后再次叠加检查重采样PET、标签及掩膜；特别看小脑、下颞、额极和深部灰质
有无截断、脑外泄漏、局部错位。将覆盖结论写入coverage_review，保留原先警示，
再由人进入STEP04。此脚本不自动写人工PASS，也不自动开始特征提取。
