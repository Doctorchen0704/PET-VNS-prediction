# STEP02：T1 FreeSurfer分区

环境：Linux/WSL、FreeSurfer **7.4.1**及合法许可证。先在终端按本机安装位置
加载`SetUpFreeSurfer.sh`。每次手动运行一例，不调度下一例，不自动修复失败。

```text
python STEP02_T1_segmentation/run_freesurfer.py --subject sub-example01 --t1 /work/bids/sub-example01/anat/sub-example01_T1w.nii.gz --subjects-dir /work/freesurfer --license /tools/license.txt --qc-table /private/manual_qc.tsv --log-output /work/logs/fs_example01
```

默认打印命令；确认后手动加`--execute`。实际参数为`recon-all -all -parallel
-openmp 8`；`--cw256`仅对原先技术核定需要该选项的病例显式指定，不自动开启。
原研究逐例选项由受控名单提供，不能根据结局选择。已有完整分区时直接复用，
脚本拒绝覆盖已有同名FreeSurfer目录；中断续跑应先由人审查原状态，不盲目重启。

输出位于`SUBJECTS_DIR/sub-example01`，主要包括：

- `mri/T1.mgz`、`mri/orig.mgz`、`mri/brainmask.mgz`、`mri/aparc+aseg.mgz`；
- `stats/aseg.stats`、`stats/lh.aparc.stats`、`stats/rh.aparc.stats`；
- 原生FreeSurfer日志。

## 必须由人检查

核对T1对应关系；检查颅骨剥离、脑室/深部核团边界、白质及皮层表面、分区覆盖，
结合体积或不对称警示查看原图。可在已配置好的FreeSurfer环境启动`freeview`，
加载T1、分区及表面查看。程序结束或文件存在不等于分区通过。

将实际结论写入`manual_qc.tsv`的`segmentation_review`。PASS_WITH_ATTENTION
保留警示描述；FAIL/HOLD保留结果并暂停后续分析，不自动改变参数。
