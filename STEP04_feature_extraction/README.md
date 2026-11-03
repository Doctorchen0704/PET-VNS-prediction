# STEP04：PET、T1特征；PVC为独立敏感性支线

主要PET/T1提取用Python、NumPy和nibabel；PVC影像命令另需Linux/WSL FreeSurfer7.4.1。
运行前人工QC表已有真实分区、配准和覆盖结论。每个输出目录必须新建，不覆盖原表。

## 1．PET六区域及82区

```text
python STEP04_feature_extraction/01_extract_pet.py --subject sub-example01 --pet /work/grid/sub-example01/directRigid_pet.nii.gz --labels /work/grid/sub-example01/aparcAseg_dseg.nii.gz --brainmask /work/grid/sub-example01/brain_mask.nii.gz --qc-table /private/manual_qc.tsv --output /work/features/sub-example01/PET
```

默认参考为supraGM。参考均值包括有限的0值；目标ROI均值只使用有限且>0体素。
先保存float32归一化图，再按固定配置计算：左右有效体素合并为双侧节点；节点
在区域复合指标内等权。82区为34对皮层和7对常规皮层下灰质，按pair字母排序、
每对先L后R，不包括混合性的腹侧间脑标签。标签体素数、有效比例、缺失及注意
标记均保留，不在提取时插补、缩尾或因结果删人。

输出`normalized_pet.nii.gz`和`pet_features.json`。JSON包含：

- subject（影像键，如sub-example01）与study_id（分析键，默认仅去sub-前缀）；
- primary_features：六个原始vns_*值；features/canonical_features：pet_supraGM__vns_*；
- roi82_features、roi82_values及逐区QC；
- 归一化参考和强度/覆盖数值检查。

不同命名体系可显式传`--study-id`，但对应关系必须来自经核实的映射，不由代码
猜测。参考敏感性通过`--reference wholeBrain`或`cerebellarCortex`分别输出新目录，
不得混入主要supraGM矩阵。

## 2．T1六结构特征

```text
python STEP04_feature_extraction/02_extract_t1.py --subject sub-example01 --aseg-stats /work/freesurfer/sub-example01/stats/aseg.stats --left-aparc-stats /work/freesurfer/sub-example01/stats/lh.aparc.stats --right-aparc-stats /work/freesurfer/sub-example01/stats/rh.aparc.stats --qc-table /private/manual_qc.tsv --output /work/features/sub-example01/T1
```

从FreeSurfer统计表读取，不用PET网格上的标签体积代替。体积指标为规定结构
之和/eTIV，皮层厚度按双侧规定parcel的面积加权，单位mm。保留不含伏隔核的
原T1纹状体/苍白球定义，不擅自与PET指标改成相同结构。
输出`t1_features.json`，features为原t1_*名，canonical_features为t1__t1_*名。
STEP05汇总时用study_id连接，不能凭文件排序与临床表对应。

## 3．可选PVC：每一子步都由人单独运行

这条线不是主分析的替代，不因为某个AUC较高改变主要指标。先准备经核对的
FreeSurfer**处理副本**，不要在保护的原分区目录运行gtmseg。

1. `03_pvc_segmentation.py`：`--subjects-dir-copy`、`--subject`、`--license`、
   `--confirm-processing-copy`、`--qc-table`、`--output`；默认打印gtmseg命令。
   人工加`--execute`后只运行gtmseg（usf2、no-seg-stats）。已有gtmseg则复用。
   人工检查高分辨率分区，记录pvc_segmentation_review。
2. `04_pvc_transforms.py`：传已对齐的**原始强度**PET、conformed T1和gtmseg.lta，
   单独生成pet_to_anat.lta及seg_to_pet.lta。这里的header-RAS identity只适用于
   已在T1物理空间的PET，不能用于原生未配准PET。
3. `05_pvc_fit.py`：显式传PET、T1、gtmseg、color table及三个LTA；先核对LTA
   类型和体素/物理几何方程，再显示拟合命令。默认FWHM5.4 mm、4线程，
   no-rescale、auto-mask1/.01、save-yhat。人工加`--execute`仅运行当前fit。
   `--matched-no-pvc`是相同新分区的无PVC对照，不等同主要原始nonPVC。
   历史4/6 mm可显式指定；不重新选择“最佳PSF”。检查拟合和覆盖后记录pvc_fit_review。
4. `06_extract_pvc.py`：传gtm.stats.dat、未舍入gtm.nii.gz、gtmseg及XtX/Xty矩阵。
   使用84个灰质标签的原分割体积加权参考，双侧节点也按体积加权，再构造
   六指标。保留负系数和数值警示，不裁剪成0。输出单独的pvc_features.json。

各脚本`--help`列出全部参数。PVC输出不能误写进主要PET six-feature文件。

以下为5.4 mm支线的完整参数示例。前三条先打印命令，人工确认后分别加
`--execute`，不是连续全自动脚本。第一条完成并检查后才运行第二条，依次进行。
PVC几何的conformed参考固定为处理副本`mri/nu.mgz`（配准步骤则用`T1.mgz`），
不混用名称相近的图像。

```text
python STEP04_feature_extraction/03_pvc_segmentation.py --subject sub-example01 --subjects-dir-copy /work/fs_pvc_copy --license /tools/license.txt --confirm-processing-copy --qc-table /private/manual_qc.tsv --output /work/pvc_segmentation/sub-example01

python STEP04_feature_extraction/04_pvc_transforms.py --subject sub-example01 --aligned-pet /work/grid/sub-example01/directRigid_pet.nii.gz --conformed-t1 /work/fs_pvc_copy/sub-example01/mri/nu.mgz --gtmseg-lta /work/fs_pvc_copy/sub-example01/mri/gtmseg.lta --qc-table /private/manual_qc.tsv --output /work/pvc_geometry/sub-example01

python STEP04_feature_extraction/05_pvc_fit.py --subject sub-example01 --aligned-pet /work/grid/sub-example01/directRigid_pet.nii.gz --conformed-t1 /work/fs_pvc_copy/sub-example01/mri/nu.mgz --gtmseg /work/fs_pvc_copy/sub-example01/mri/gtmseg.mgz --color-table /work/fs_pvc_copy/sub-example01/mri/gtmseg.ctab --pet-to-anat-lta /work/pvc_geometry/sub-example01/pet_to_anat.lta --seg-to-pet-lta /work/pvc_geometry/sub-example01/seg_to_pet.lta --gtmseg-lta /work/fs_pvc_copy/sub-example01/mri/gtmseg.lta --qc-table /private/manual_qc.tsv --output /work/pvc54/sub-example01

python STEP04_feature_extraction/06_extract_pvc.py --subject sub-example01 --stats /work/pvc54/sub-example01/fit/gtm.stats.dat --coefficients /work/pvc54/sub-example01/fit/gtm.nii.gz --segmentation /work/fs_pvc_copy/sub-example01/mri/gtmseg.mgz --xtx /work/pvc54/sub-example01/fit/aux/XtX.mat --xty /work/pvc54/sub-example01/fit/aux/Xty.mat --qc-table /private/manual_qc.tsv --output /work/pvc_features/sub-example01
```

旧PetPrep支线的6幅中间图像缺口不会在这里被“自动修复”；其保留ROI表可用于
既定统计核验，但不代表该历史支线图像重提取已验收。
