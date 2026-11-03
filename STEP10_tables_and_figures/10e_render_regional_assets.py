"""Generate Figure2 standard-brain cortical, deep and reference-slice assets."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output
import numpy as np


def group_definition(config):
    groups=config['primary_features'];groups=groups.get('features',groups)
    pairs=config['bilateral_pairs'];nodes={node:group for group,names in groups.items() for node in names}
    if len(groups)!=6 or len(nodes)!=19:raise ValueError('Original six groups and nineteen bilateral nodes required')
    return groups,pairs,nodes


def regional_effects(payload,config):
    if isinstance(payload,dict):
        if 'primary' in payload:rows=payload['primary']
        elif 'rows' in payload:
            if payload.get('family_tests')!=6:raise ValueError('Only the complete primary six-test family is accepted for primary brain colors')
            rows=payload['rows']
            if any(r.get('adjustment')!='clinical5_scanner' for r in rows):raise ValueError('Primary model adjustment differs')
        else:raise ValueError('Supply six primary rows, not an unlabelled object')
    else:rows=payload
    if len(rows)!=6:raise ValueError('Exactly six fixed primary estimates required')
    effects={}
    for row in rows:
        name=row.get('pathway',row.get('region'))
        if not isinstance(name,str):raise ValueError('Explicit regional identifier required')
        name=name.removeprefix('pet_supraGM__')
        if name in effects:raise ValueError('Duplicated primary regional estimate')
        value=row.get('difference_pooled_sd',row.get('standardized_difference'))
        if value is None or not np.isfinite(value):raise ValueError('Finite standardized adjusted difference required')
        effects[name]=float(value)
    groups,_,_=group_definition(config)
    if set(effects)!=set(groups):raise ValueError('All six fixed group identifiers must match feature definitions')
    return effects


def geometry(fs,config,effects,cmap,norm):
    import nibabel as nib
    import pyvista as pv
    from matplotlib.colors import to_rgb
    groups,pairs,nodes=group_definition(config)
    segim=nib.load(fs/'mri/aparc+aseg.mgz');seg=np.asarray(segim.dataobj,dtype=np.int16)
    t=segim.header.get_vox2ras_tkr();hemis={};deep={};records=[]
    for hemi,side in [('lh','left'),('rh','right')]:
        verts,faces,meta=nib.freesurfer.read_geometry(fs/f'surf/{hemi}.pial',read_metadata=True)
        if tuple(meta['volume'])!=tuple(seg.shape) or not np.allclose(meta['voxelsize'],segim.header.get_zooms()[:3]):
            raise ValueError('Surface and paired segmentation volume metadata differ')
        affine=np.eye(4);affine[:3,:3]=np.array([meta['xras'],meta['yras'],meta['zras']]).T*meta['voxelsize']
        affine[:3,3]=meta['cras']-affine[:3,:3]@(meta['volume']/2)
        if not np.allclose(affine,segim.affine):raise ValueError('Surface/segmentation coordinate-frame mismatch')
        ann,_,names=nib.freesurfer.read_annot(fs/f'label/{hemi}.aparc.annot',orig_ids=False);names=[x.decode() for x in names]
        fa=ann[faces];mode=fa[:,0].copy();repeat=fa[:,1]==fa[:,2];mode[repeat]=fa[repeat,1]
        colors=np.tile(np.round(np.array(to_rgb('#bfc3c6'))*255).astype(np.uint8),(len(faces),1))
        for node,group in nodes.items():
            if pairs[node]['class']!='cortical':continue
            idx=names.index(node);selected=mode==idx
            if not selected.any():raise ValueError('Missing cortical parcel '+node)
            colors[selected]=np.round(np.array(cmap(norm(effects[group])))[:3]*255).astype(np.uint8)
            records.append({'hemi':hemi,'node':node,'group':group,'faces':int(selected.sum())})
        mesh=pv.PolyData(verts,np.c_[np.full(len(faces),3),faces].ravel());mesh.cell_data['effect_rgb']=colors;hemis[hemi]=mesh
    for node,group in nodes.items():
        if pairs[node]['class']=='cortical':continue
        for hemi,side in [('lh','left'),('rh','right')]:
            label=pairs[node][side];ijk=np.argwhere(seg==label)
            if not len(ijk):raise ValueError('Missing deep label '+str(label))
            lo=np.maximum(ijk.min(0)-2,0);hi=np.minimum(ijk.max(0)+3,np.array(seg.shape))
            mask=(seg[tuple(slice(int(l),int(h)) for l,h in zip(lo,hi))]==label).astype(np.uint8)
            grid=pv.ImageData(dimensions=mask.shape,origin=lo.astype(float),spacing=(1.,1.,1.));grid.point_data['mask']=mask.ravel(order='F')
            mesh=grid.contour([.5],scalars='mask',method='marching_cubes');mesh.transform(t,inplace=True);mesh=mesh.clean().triangulate()
            bounds=np.array(mesh.bounds);volume=abs(mesh.volume)
            mesh=mesh.smooth_taubin(n_iter=40,pass_band=.08).compute_normals(auto_orient_normals=True,consistent_normals=True)
            if max(abs(bounds-np.array(mesh.bounds)))>=1.2 or abs(abs(mesh.volume)/volume-1)>=.06:
                raise ValueError('Display smoothing changed geometry beyond source tolerance')
            deep[(hemi,node)]=mesh
            records.append({'hemi':hemi,'node':node,'group':group,'volume_label':label,'voxels':len(ijk),'faces':mesh.n_cells})
    return hemis,deep,nodes,records


def render_meshes(meshes,colors,display,destination):
    import pyvista as pv
    from PIL import Image
    if 'camera' not in display or 'parallel_scale' not in display:raise ValueError('Explicit frozen camera and parallel_scale required')
    size=display.get('window_size',[2250,1650]);plot=pv.Plotter(off_screen=True,window_size=size,lighting='three lights')
    plot.set_background('white');plot.enable_anti_aliasing('ssaa')
    for key,mesh in meshes.items():
        style=dict(smooth_shading=True,ambient=.60,diffuse=.40,specular=0,specular_power=10,show_scalar_bar=False)
        if colors is None:plot.add_mesh(mesh,scalars='effect_rgb',rgb=True,preference='cell',**style)
        else:plot.add_mesh(mesh,color=colors[key],**style)
    plot.camera_position=display['camera'];plot.enable_parallel_projection();plot.camera.parallel_scale=display['parallel_scale']
    rgba=plot.screenshot(transparent_background=False,return_img=True);plot.close();pixels=np.asarray(rgba)[...,:3]
    yy,xx=np.where(np.any(pixels<248,axis=2))
    if not len(xx):raise ValueError('No anatomy visible from supplied camera')
    padding=int(display.get('crop_padding',66));x0=max(0,int(xx.min())-padding);x1=min(size[0],int(xx.max())+padding+3)
    y0=max(0,int(yy.min())-padding);y1=min(size[1],int(yy.max())+padding+3)
    Image.fromarray(pixels).crop((x0,y0,x1,y1)).save(destination)
    # Annotation endpoints here must be already manually verified for the frozen
    # camera and the specified crop. Missing endpoints are omitted, not guessed.
    return {'crop_xyxy':[x0,y0,x1,y1],'camera':display['camera'],'parallel_scale':display['parallel_scale'],
            'annotations':display.get('annotations',{}),'orientation':display.get('orientation',['',''])}


def slices(fs,config,effects,cmap,norm,out):
    import nibabel as nib
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups,pairs,nodes=group_definition(config)
    original=[nib.load(fs/'mri/brain.mgz'),nib.load(fs/'mri/aparc+aseg.mgz')]
    if original[0].shape!=original[1].shape or not np.allclose(original[0].affine,original[1].affine):raise ValueError('Reference brain/segmentation geometry mismatch')
    if not np.allclose(original[0].affine,original[0].header.get_vox2ras_tkr()):raise ValueError('Reference volume RAS differs from surface tkRAS')
    brainim,segim=[nib.as_closest_canonical(x) for x in original]
    if not np.allclose(brainim.affine[:3,:3],np.eye(3)):raise ValueError('Source slice recipe requires canonical 1mm reference grid')
    brain=np.asarray(brainim.dataobj);seg=np.asarray(segim.dataobj,dtype=int);aff=brainim.affine;records={}
    label_group={pairs[n][side]:group for n,group in nodes.items() for side in ('left','right')}
    for name,axis,location,planes,xlim,ylim in [('axial',2,0,(0,1),(-84,84),(-111,82)),('coronal',1,-10,(0,2),(-84,84),(-61,85))]:
        index=int(round(location-aff[axis,3]));image=np.take(brain,index,axis=axis).T;labels=np.take(seg,index,axis=axis).T
        over=np.zeros((*labels.shape,4));counts={g:0 for g in groups};plane=np.zeros(labels.shape,int)
        for label,group in label_group.items():
            selected=labels==label;over[selected,:3]=np.round(np.array(cmap(norm(effects[group])))[:3]*255)/255
            over[selected,3]=.82;counts[group]+=int(selected.sum());plane[selected]=list(groups).index(group)+1
        if not all(v>0 for v in counts.values()):raise ValueError('Fixed reference slice does not intersect all six groups')
        extent=[aff[planes[0],3]-.5,aff[planes[0],3]+image.shape[1]-.5,aff[planes[1],3]-.5,aff[planes[1],3]+image.shape[0]-.5]
        fig=plt.figure(figsize=(6,5),dpi=300,facecolor='black');ax=fig.add_axes([0,0,1,1],facecolor='black')
        ax.imshow(image,cmap='gray',vmin=0,vmax=120,origin='lower',extent=extent,interpolation='bilinear')
        ax.imshow(over,origin='lower',extent=extent,interpolation='nearest')
        cx=np.arange(labels.shape[1])+aff[planes[0],3];cy=np.arange(labels.shape[0])+aff[planes[1],3]
        for i in range(1,7):ax.contour(cx,cy,(plane==i).astype(float),levels=[.5],colors=['#5b565f'],linewidths=.28,alpha=.75)
        ax.set(xlim=xlim,ylim=ylim,aspect='equal');ax.axis('off');fig.savefig(out/(name+'.png'),dpi=300,facecolor='black');plt.close(fig)
        records[name]={'coordinate':location,'group_counts':counts,'orientation':['L','R'],'annotations':{}}
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fsaverage',type=Path,required=True)
    p.add_argument('--feature-config',type=Path,required=True)
    p.add_argument('--effects',type=Path,required=True,help='JSON: six rows with pathway and difference_pooled_sd, already adjusted estimates')
    p.add_argument('--display-config',type=Path,required=True,help='JSON left_lateral,left_medial,deep camera/scale/window/crop/verified annotations')
    p.add_argument('--assets-use-authorized',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    config=read_json(a.feature_config);display=read_json(a.display_config)
    effects=regional_effects(read_json(a.effects),config)
    cmap=LinearSegmentedColormap.from_list('purple_orange',matplotlib.colormaps['PuOr_r'](np.linspace(.12,.88,256)),N=256);norm=Normalize(-1.3,1.3)
    hemis,deep,nodes,mapping=geometry(a.fsaverage,config,effects,cmap,norm)
    out=new_output(a.output);assetmeta={}
    for key in ('left_lateral','left_medial'):
        assetmeta[key]=render_meshes({'lh':hemis['lh']},None,display[key],out/(key+'.png'))
    color={key:np.array(cmap(norm(effects[nodes[key[1]]])))[:3] for key in deep}
    assetmeta['deep']=render_meshes(deep,color,display['deep'],out/'deep.png')
    assetmeta.update(slices(a.fsaverage,config,effects,cmap,norm,out))
    declared={k:{'image':k+'.png','source':'Caller-supplied standard fsaverage pial/aparc anatomy; regional composite estimates',
        'license':display['license'],'annotations':v.get('annotations',{}),'orientation':v.get('orientation',['',''])} for k,v in assetmeta.items()}
    write_json(out/'Figure2_assets.json',{'anatomy_assets':declared})
    write_json(out/'asset_geometry.json',{'effects':effects,'mapping':mapping,'views':assetmeta,'visual_QC':'REQUIRED',
        'note':'All group effects shown without significance threshold; composites are not voxelwise/nucleus-specific effects'})


if __name__=='__main__':main()
