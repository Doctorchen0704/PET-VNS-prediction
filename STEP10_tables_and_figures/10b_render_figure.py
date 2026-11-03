"""Render exactly one selected figure from explicit results and permitted assets."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pet_vns_prediction import figures
from pet_vns_prediction.figure4 import render_figure4


def assets_from_json(path):
    data=read_json(path);root=Path(path).resolve().parent
    def one(row):
        source=Path(row['image']);source=source if source.is_absolute() else root/source
        if not source.is_file():raise FileNotFoundError(source)
        return figures.AnatomyAsset(source,row['source'],row['license'],
            {k:tuple(v) for k,v in row.get('annotations',{}).items()},tuple(row.get('orientation',['',''])))
    if 'template_stacks' in data:
        return {key:[one(row) for row in value] for key,value in data['template_stacks'].items()}
    return {key:one(value) for key,value in data['anatomy_assets'].items()}


def final_label_changes(fig,number):
    replacements={'Associations before and after further clinical adjustment':'Adjusted regional differences',
                  'Feature-removal ablation':'Regional feature ablation'}
    for text in fig.texts:
        if text.get_text() in replacements:text.set_text(replacements[text.get_text()])
    if number=='Figure3':fig.text(.46,.540,'AUC',fontsize=9.,ha='right')
    return fig


def figure1_integrated(data,assets,dpi):
    # Render the unchanged A-C panels at the requested native resolution, then
    # place their canvas above an explicit method-only D band. No preview upsample.
    import numpy as np
    from matplotlib.patches import Rectangle
    base=figures.figure1_workflow(data['counts'],assets,period=data['period'],exclusions=data['exclusions'],dpi=dpi)
    base.canvas.draw();pixels=np.asarray(base.canvas.buffer_rgba())[...,:3].copy();plt.close(base)
    fig=plt.figure(figsize=(12.8,11.55),dpi=dpi,facecolor='white')
    ax=fig.add_axes([0,1.75/11.55,1,9.8/11.55]);ax.imshow(pixels);ax.axis('off')
    band=fig.add_axes([0,0,1,1.75/11.55]);band.set(xlim=(0,12.8),ylim=(-1.75,0));band.axis('off')
    purple,orange,muted='#79529B','#A6652F','#6E6777'
    band.plot([.26,12.54],[-.03,-.03],color='#DED9E3',lw=.75)
    band.text(.26,-.32,'D',fontsize=15,fontweight='bold');band.text(.75,-.32,'Exploratory spatial analyses',fontsize=13,fontweight='bold')
    band.text(1.78,-.65,'82-region FDG profiles',fontsize=11,fontweight='bold',color=purple,ha='center')
    for r in range(4):
        for c in range(9):band.add_patch(Rectangle((.93+c*.19,-1.32+r*.13),.15,.095,fc='#F0EBF5',ec=purple,lw=.5))
    band.text(4.55,-.65,'PC1',fontsize=11,fontweight='bold',color=purple,ha='center')
    for r in range(5):band.add_patch(Rectangle((4.43,-1.3375+r*.11),.24,.075,fc='#F0EBF5',ec=purple,lw=.6))
    arrow={'arrowstyle':'->','lw':1.,'color':muted}
    band.annotate('',(4.03,-1.08),(2.93,-1.08),arrowprops=arrow)
    band.plot([4.91,5.65,5.65],[-1.08,-1.08,-.87],color=muted,lw=.95)
    band.plot([5.65,5.65],[-.87,-1.29],color=muted,lw=.95)
    for yy,color in [(-.87,purple),(-1.29,orange)]:band.annotate('',(6.10,yy),(5.65,yy),arrowprops={'arrowstyle':'->','color':color,'lw':.95})
    band.text(6.30,-.87,'Normative molecular correspondence',fontsize=11,fontweight='bold',color=purple)
    band.text(6.30,-1.29,'Independent PC1 comparison',fontsize=11,fontweight='bold',color=orange)
    counts=data['external_counts']
    band.text(6.30,-1.54,f"TLE (n = {counts['TLE']}) / Healthy (n = {counts['Healthy']})",fontsize=10.2,color=muted)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure',choices=['Figure1','Figure2','Figure3','Figure4','FigureS1','FigureS2'],required=True)
    p.add_argument('--input',type=Path,required=True,help='Saved numerical plotting payload JSON')
    p.add_argument('--assets',type=Path,help='Explicit asset-declaration JSON; required for brain/template figures')
    p.add_argument('--assets-use-authorized',action='store_true')
    p.add_argument('--dpi',type=int,default=150,help='150 for review; 600 for final TIFF, supplied anatomy must have sufficient native resolution')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=read_json(a.input)
    if a.figure in ('Figure1','Figure2','Figure4') and not (a.assets and a.assets_use_authorized):
        p.error('Supply licensed standard-reference assets and explicitly confirm permitted use')
    if a.figure=='Figure4':
        declaration=read_json(a.assets);root=a.assets.resolve().parent
        paths={k:(Path(v) if Path(v).is_absolute() else root/v) for k,v in declaration['png_paths'].items()}
        render_figure4(data,paths,a.output,asset_color_limits=declaration['asset_color_limits'],
            assets_authorized=True,dpi=a.dpi,preview_dpi=min(150,a.dpi));return
    if a.figure=='Figure1':fig=figure1_integrated(data,assets_from_json(a.assets),a.dpi)
    elif a.figure=='Figure2':fig=figures.figure2_regional(data,assets_from_json(a.assets),dpi=a.dpi)
    else:
        renderer={'Figure3':figures.figure3_prediction,'FigureS1':figures.figure_s1_validation,'FigureS2':figures.figure_s2_processing}[a.figure]
        fig=renderer(data,dpi=a.dpi)
    final_label_changes(fig,a.figure);out=new_output(a.output)
    try:
        for suffix in ('png','pdf','tiff'):
            figures.export_figure(fig,out/(a.figure+'.'+suffix),dpi=a.dpi,
                minimum_asset_dpi=a.dpi if a.figure=='Figure2' else None)
        write_json(out/'figure_notes.json',{'figure':a.figure,'dpi':a.dpi,'statistics_refitted':False,
            'visual_check':'REQUIRED','Figure1_note':'A-C rasterized at native requested DPI; D is method-only vector. Not claimed pixel-identical to historical SVG export.' if a.figure=='Figure1' else None})
    finally:plt.close(fig)


if __name__=='__main__':main()
