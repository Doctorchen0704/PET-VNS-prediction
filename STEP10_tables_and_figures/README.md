# STEP10：逐张出表和出图

每条命令只处理所指定的一张图或一张表，不修改分析结果，不调用在线服务。
检查输入版本后手动运行；生成后还需人工看图。所有输出目录必须新建。

## 10a 由分析结果组装绘图数据

```sh
python STEP10_tables_and_figures/10a_prepare_plot_payloads.py --figure Figure3 --input /private/model_input.json --predictions /work/step07/predictions.json --metrics /work/step07/performance.json --intervals /work/step07/conditional_intervals.json --ablation /work/ablation/results_PRIVATE.json --output /work/Figure3_payload
python STEP10_tables_and_figures/10a_prepare_plot_payloads.py --figure Figure4 --input /private/aggregate_intervals.json --arrays /work/DAT/specificity_arrays.npz --labels /private/cortical_labels.json --color-limits /private/color_limits.json --output /work/Figure4_payload
```

Figure3：由20次留出预测的每人均值计算ROC与5等宽校准组；核对AUC、Brier与性能文件一致。
STEP07性能点估计和条件区间分别保存在`performance.json`与`conditional_intervals.json`，必须显式接入；后者不被冒充为完整开发bootstrap。
消融可直接读取STEP08的`results_PRIVATE.json`（含`ablation_metrics.models/comparisons`），脚本按固定去丘脑、去纹状体/苍白球、两者均去的顺序组装并核对完整CP点估计一致。
也可读取已经核对的三行消融绘图JSON，字段`delta_auc_CP_minus_reduced`、`conditional95`。
Figure4：从已保存rank4/rank5残差中组装136个皮层点，检查与保存的效应完全相符；区间由既有参与者重抽样结果提供，不在画图时重算。

Figure2/S1/S2可直接使用相应分析阶段保存的绘图JSON。受控输入随交接材料提供。
这只是使用已计算结果作图，不能将旧版本绘图JSON混入新结果。

## 10b 每次只画指定一张

```sh
python STEP10_tables_and_figures/10b_render_figure.py --figure Figure3 --input /work/Figure3_payload/Figure3_plot_data.json --dpi 150 --output /work/Figure3_preview
python STEP10_tables_and_figures/10b_render_figure.py --figure Figure2 --input /private/Figure2_plot_data.json --assets /private/Figure2_assets.json --assets-use-authorized --dpi 600 --output /work/Figure2_final
python STEP10_tables_and_figures/10b_render_figure.py --figure Figure4 --input /work/Figure4_payload/Figure4_plot_data.json --assets /work/Figure4_surfaces/figure4_assets.json --assets-use-authorized --dpi 150 --output /work/Figure4_preview
```

支持`Figure1`、`Figure2`、`Figure3`、`Figure4`、`FigureS1`、`FigureS2`。
默认150dpi供检查，明确传`--dpi 600`才导出600dpi。输出PNG、PDF、RGB TIFF。
不要把低分辨率脑图扩大充作高分辨率原始资产。Figure4输出还含布局记录。

### Figure1

正式Figure1复现优先使用同目录`10f_export_methods_svg.js`与已批准的可编辑SVG（由受控材料提供）；它保持当前方法图布局。

```sh
node STEP10_tables_and_figures/10f_export_methods_svg.js /private/Figure1_approved.svg /work/Figure1 --dpi 150
```

使用普通Node及sharp；也可直接在Inkscape打开SVG导出。它不需要任何助手软件。
以下10b Figure1分支是根据计数重新绘制的独立备选，不自动替换已批准图。

JSON包含`counts`（vns_implantation、imaging_candidates、primary、responders、nonresponders）、`period`、三对`exclusions`、`external_counts`（TLE、Healthy）。
资产JSON含`template_stacks`，其中pet/t1/overlay各三张标准模板切片，条目字段见下。
本次独立入口采用原A–C组合图在所选DPI原生渲染后加D方法分支，D矩阵/箭头是方法示意而非测量值。
其科学内容对应整合稿，但**没有声称与旧SVG导出的Figure1逐像素一致**；PDF中的A–C是所选原生DPI栅格，D为矢量。
旧SVG的完全同版矢量导出不是本入口的已验证功能。

### Figure2

绘图JSON需要6行`primary`、6行`extended`、N×6的`raw_relative_values`、`y`、`primary_family_size:6`、`extended_family_size:18`、`shared_distribution_xlim`。
主模型行：`difference_pooled_sd`、`ci_low_pooled_sd`、`ci_high_pooled_sd`、`q_family`。
扩展行：`standardized_difference`、`standardized_CI95`、`q_family`；q必须来自完整18项家族。
资产JSON格式为`{"anatomy_assets":{"left_lateral":{...},"left_medial":{...},"axial":{...},"coronal":{...},"deep":{...}}}`。
每条资产需`image`（相对JSON或绝对路径）、`source`、`license`，可选`annotations`（标签至裁剪后0–1坐标）、`orientation`（两个方位字母）。
资产是经确认的标准脑原始渲染，不是某一患者截图。可以复用受控包中的对应文件，也可按10e重新生成；10b只负责组合。
绘图入口已采用最终的`Adjusted regional differences`标题。

### Figure3/S1/S2

Figure3不在作图时优化阈值，不绘制模型训练集的替代ROC。图内保留AUC标题及`Regional feature ablation`。
S1需两方向、两模型的`cross_scanner`四行，以及保存的`ranked_saved_CP_percentiles`，字段定义见数值模块`figures.figure_s1_validation`。
S2需`order`、`labels`、`regions`、`association_rows`、`models`、`q_family_size:96`及未显示但仍保留在家族的分支说明。
显示子集不改变多重校正范围。图中设备名称以核实后的术语提供，不从文件名猜设备。

## 10c 从公共标准脑几何重新生成Figure4脑图

```sh
python STEP10_tables_and_figures/10c_render_PC1_SERT_surfaces.py --input /private/cortical_maps.npz --fsaverage /licensed/fsaverage --color-limits /private/color_limits.json --assets-use-authorized --output /work/Figure4_surfaces
```

需额外安装nibabel、PyVista/VTK，并有可用的离屏OpenGL环境；Windows和Linux均可，但GPU/驱动会影响像素抗锯齿。
NPZ含`labels`（68区，L_/R_前缀）、`observed68`、`maps68x3`。JSON含`PC1_color_limits`、`SERT_color_limits`。
从fsaverage pial/aparc逐面映射，灰色留给非分析区域，分别导出左右外侧/内侧八张PNG。
默认1560×1080原生图，颜色、camera和73 parallel_scale沿用实际实现；另保存区域映射以供人工查错。
许可证或半球映射检查不由程序自动代替。

## 10d 描述性表

如从STEP05重新组装原Table1字段，先运行独立10h，再手动运行10d：

```sh
python STEP10_tables_and_figures/10h_prepare_Table1.py --model-input /work/step05/model_input.json --metadata /private/metadata.csv --output /work/Table1_payload
```

metadata另需`etiology_detailed`（原详细病因）和`scanner_source_key`（原分组键）。
此入口沿用原表从冻结log1p列反变换月均频率的计算，字段明确标作reconstructed；
不替换临床原始频率，不推造术后发作次数。设备原键与核实后的显示名称分开保存。

```sh
python STEP10_tables_and_figures/10d_descriptive_table.py --input /private/Table1_input.json --output /work/Table1
```

输入JSON含`rows`、`y`、`continuous`、`categorical`。字段必须已按患者键核实对齐。
发作频率保留历史报告的月均频率含义；不要将其当作严格28/30天事件计数。
输出结构化JSON及CSV，可导入Word；只做原有描述，不自动增加组间检验。

## 10e 从标准脑重新生成Figure2五个解剖资产

```sh
python STEP10_tables_and_figures/10e_render_regional_assets.py --fsaverage /licensed/fsaverage --feature-config configs/features.json --effects /work/primary_six_effects.json --display-config /private/Figure2_display.json --assets-use-authorized --output /work/Figure2_assets
```

需nibabel、PyVista/VTK和标准fsaverage的pial/aparc注释、brain及aparc+aseg体积。
六个已调整组差异均匀映射到原19双侧脑区，采用固定PuOr_r色阶[-1.3,1.3]；不按显著性删去区域。
皮层来自逐面多数标签；深部14个网格来自原分割等值面、固定显示平滑参数，并核对表面/体积坐标与平滑位移/体积变化。
固定轴位z=0、冠状位y=-10是fsaverage坐标，不写成MNI坐标，不按疗效选择切面。

显示JSON需`left_lateral`、`left_medial`、`deep`三组，每组包括原`camera`、`parallel_scale`、可选`window_size`和`crop_padding`。
`annotations`如需使用，必须是相应相机和最终裁剪后经人工核实的0–1端点；缺失时省略引线，不能猜测隐藏核团位置。
顶层`license`说明素材使用许可。受控材料可提供实际采用的显示配置。
输出五PNG、`Figure2_assets.json`及区域/几何清单，随后手动运行10b。

这一步补上脑图生成代码，但本轮未重跑VTK/GPU绘制，也未证明不同平台逐像素相同。
显示配置改变、驱动差异或没有沿用已核实端点时，必须先人工复核，不能直接替换已批准投稿图。

## 10g 从新分析输出组装Figure2/S1/S2

```sh
python STEP10_tables_and_figures/10g_prepare_additional_payloads.py --figure Figure2 --model-input /work/step05/model_input.json --primary /work/primary6/associations.json --extended /work/clinical18/associations.json --distribution-limits 0.65 1.4 --output /work/Figure2_payload
python STEP10_tables_and_figures/10g_prepare_additional_payloads.py --figure FigureS1 --scanner /work/scanner/results_PRIVATE.json --development /work/development/results_PRIVATE.json --output /work/FigureS1_payload
python STEP10_tables_and_figures/10g_prepare_additional_payloads.py --figure FigureS2 --model-input /work/step05/model_input.json --processing-associations /work/processing96/associations.json --primary-predictions /work/step07/predictions.json --processing-predictions /work/processing_predictions/results_PRIVATE.json --output /work/FigureS2_payload
```

Figure2：必须供给完整6/18项家族；固定选取`clinical5_scanner`和`base_plus_epilepsy_type_and_etiology3`，按原六区域列序排列，不能只给显著区域。
示例分布范围仅说明参数用法，正式复现应传入原冻结范围，不根据组差异重新调范围；若截掉任何焦点区域的真实散点，脚本报错。
S1：读两方向真实设备及条件区间、开发bootstrap的逐患者三分位范围，按中位概率稳定排序；不把其解释为外部验证或预测置信区间。
S2：必须给全96项关联和全14个处理预测。沿固定家族提取两区域/六展示行，不重算或缩小FDR家族。
S2的AUC/Brier条件区间来自固定的患者平均OOF预测、seed20260910和2000共享重抽样（无模型拟合；单类抽样不替换），保留全部8版本指标后才选择展示六行。
程序核对同一病例顺序、主/处理预测共同分折摘要和既定variant列名；不猜测缺失分支。

## 每张图的人工核对

1. 输入的病例数、结局版本、ROI顺序和q家族是否与正文同一版本。
2. 散点是否是每位患者/每个皮层区的真实值，而非为了美观抽样或模拟。
3. 脑图的左右、颜色、色阶及深部结构标注是否正确。
4. 标签有无截断/重叠，600dpi栅格资产是否有足够原生分辨率。
5. 图注与正文对AUC、条件区间、外部队列作用的解释是否相符。
