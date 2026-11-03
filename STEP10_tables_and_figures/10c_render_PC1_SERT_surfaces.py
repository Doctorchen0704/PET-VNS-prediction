"""Render eight standard-fsaverage PC1/SERT assets from DK68 parcel values."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output
import numpy as np


def color_faces(annotations,faces,names,labels,values,hemi,cmap,norm):
    faceann=annotations[faces];mode=faceann[:,0].copy();repeated=faceann[:,1]==faceann[:,2]
    mode[repeated]=faceann[repeated,1]
    colors=np.tile(np.array([190,193,194],dtype=np.uint8),(len(faces),1));ledger=[]
    prefix='L_' if hemi=='lh' else 'R_'
    for label,value in zip(labels,values):
        if not label.startswith(prefix):continue
        idx=names.index(label[2:]);take=mode==idx
        if not take.any():raise ValueError('Parcel has no labelled faces: '+label)
        colors[take]=np.round(np.array(cmap(norm(float(value))))[:3]*255).astype(np.uint8)
        ledger.append({'label':label,'value':float(value),'annotation_index':idx,'faces':int(take.sum())})
    if len(ledger)!=34:raise ValueError('Exactly 34 parcels per hemisphere required')
    return colors,ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='NPZ labels(68), observed68, maps68x3')
    p.add_argument('--fsaverage',type=Path,required=True,help='Permitted reference with surf/{lh,rh}.pial and label/{lh,rh}.aparc.annot')
    p.add_argument('--color-limits',type=Path,required=True)
    p.add_argument('--assets-use-authorized',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=1560);p.add_argument('--height',type=int,default=1080)
    a=p.parse_args()
    import nibabel as nib
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    with np.load(a.input,allow_pickle=False) as f:labels=f['labels'].astype(str);values={'PC1':f['observed68'],'SERT':f['maps68x3'][:,0]}
    if len(labels)!=68 or len(set(labels))!=68:raise ValueError('Complete fixed 68-parcel mapping required')
    limits=read_json(a.color_limits)
    cmaps={'PC1':LinearSegmentedColormap.from_list('pc1',['#C17842','#F8F6F1','#715296']),
           'SERT':LinearSegmentedColormap.from_list('sert',['#EFEDE9','#D8C5E1','#A48ABC','#6B528E','#392A58'])}
    out=new_output(a.output);mapping={};paths={}
    for metric in ('PC1','SERT'):
        norm=Normalize(*limits[metric+'_color_limits']);mapping[metric]=[]
        if min(values[metric])<norm.vmin or max(values[metric])>norm.vmax:raise ValueError('Values exceed frozen display limits')
        for hemi in ('lh','rh'):
            verts,faces=nib.freesurfer.read_geometry(a.fsaverage/f'surf/{hemi}.pial')
            ann,_,names=nib.freesurfer.read_annot(a.fsaverage/f'label/{hemi}.aparc.annot',orig_ids=False)
            names=[n.decode() for n in names]
            if len(ann)!=len(verts):raise ValueError('Surface/annotation vertex mismatch')
            colors,rows=color_faces(ann,faces,names,labels,values[metric],hemi,cmaps[metric],norm)
            mapping[metric]+=rows
            mesh=pv.PolyData(verts,np.c_[np.full(len(faces),3),faces].ravel());mesh.cell_data['region_rgb']=colors
            for view in ('lateral','medial'):
                plot=pv.Plotter(off_screen=True,window_size=(a.width,a.height),lighting='three lights')
                plot.set_background('white');plot.enable_anti_aliasing('ssaa')
                plot.add_mesh(mesh,scalars='region_rgb',rgb=True,preference='cell',smooth_shading=True,
                    ambient=.58,diffuse=.42,specular=0,show_scalar_bar=False)
                focus=np.asarray(mesh.center);direction=-1 if (hemi=='lh')==(view=='lateral') else 1
                plot.camera_position=[focus+np.array([direction*450.,0.,0.]),focus,[0,0,1]]
                plot.enable_parallel_projection();plot.camera.parallel_scale=73
                key=f'{metric}_{hemi}_{view}';dest=out/(key+'.png')
                plot.screenshot(str(dest),transparent_background=False);plot.close();paths[key]=dest.name
    write_json(out/'surface_mapping.json',mapping)
    write_json(out/'figure4_assets.json',{'png_paths':paths,'asset_color_limits':{'PC1':limits['PC1_color_limits'],'SERT':limits['SERT_color_limits']},
        'standard_brain':'Caller-supplied fsaverage DK aparc','visual_QC':'REQUIRED'})


if __name__=='__main__':main()
