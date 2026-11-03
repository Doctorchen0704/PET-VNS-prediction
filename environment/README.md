# 环境

## Windows 或 Linux：数值、特征及二维绘图

原统计环境为Python 3.12.4。`requirements.txt`记录已实际测试过的依赖版本：
NumPy 2.4.3、pandas 3.0.3、SciPy 1.17.1、scikit-learn 1.8.0、threadpoolctl 3.6.0、
nibabel 5.4.0、Matplotlib 3.11.0、Pillow 12.1.1。

从代码包根目录执行：

```sh
python -m venv .venv
# Windows PowerShell：.venv/Scripts/Activate.ps1
# Linux：source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

统计按单线程BLAS运行。不要开启Python的`-O`优化模式，否则会关闭必须的数值断言。

## DICOM转换

单独安装dcm2niix；原研究使用实际版本`v1.0.20250505`。
STEP01通过显式路径调用它，不安装或下载软件。`STEP01_dicom2bids`是步骤名称，
不表示改用了名为dcm2bids的另一套转换程序。

## Linux／WSL：影像处理

FreeSurfer 7.4.1及合法许可证，使用同一Linux文件系统视图的T1、PET和输出路径。
先按FreeSurfer本身说明设置`FREESURFER_HOME`、`FS_LICENSE`及环境脚本；不要在
Linux命令中直接填Windows盘符。原recon-all为8线程、配准为4线程。
数值计算可在Windows做，不必把整条流程绑在同一个操作系统。

## 标准脑三维资产（可选）

STEP10的标准脑渲染另需PyVista、VTK及可用的离屏渲染环境；参考脑和分区由
研究者依其许可提供。这里未将当前机器未实测的三维渲染依赖声称为“已验证环境”。
已有批准的标准脑PNG可直接用于组合绘图；重新生成时应记录软件版本并目视检查。

整个代码包不需要任何聊天软件、账户连接或助手运行环境。

## 原Figure1矢量导出（可选）

10f是普通Node.js脚本；从代码包根目录运行`npm install`，安装根目录`package.json`指定的Sharp。
本轮导出测试使用Node.js v24.22.1、Sharp 0.35.5。也可直接用Inkscape等矢量软件
打开受控材料中的批准SVG导出；不需要重画该方法示意图。
