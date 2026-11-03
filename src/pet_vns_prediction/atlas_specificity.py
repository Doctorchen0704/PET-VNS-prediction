"""Finite atlas-source sensitivity; no data/file/PCA operations."""
import numpy as np
from scipy.stats import rankdata

MAPS=['SERT_primary_Beliveau','SERT_alternative_Savli','DAT_Dukart']

def require(condition,message):
    if not bool(condition): raise ValueError(message)

def checked(a,shape=None):
    x=np.asarray(a,dtype=np.float64)
    require(np.isfinite(x).all(),'Nonfinite input')
    if shape is not None: require(x.shape==shape,'Unexpected dimensions')
    return x

def corr(x,y):
    x=np.atleast_2d(checked(x)); y=checked(y)
    if y.ndim==1:y=y[:,None]
    require(x.ndim==2 and y.ndim==2 and x.shape[1]==y.shape[0],'Correlation shape')
    xc=x-x.mean(1,keepdims=True); yc=y-y.mean(0,keepdims=True)
    nx=np.linalg.norm(xc,axis=1); ny=np.linalg.norm(yc,axis=0)
    require(np.all(nx>1e-12) and np.all(ny>1e-12),'Degenerate correlation')
    return np.clip((xc/nx[:,None])@(yc/ny),-1,1)

def residual(x,z):
    return x-z@np.linalg.lstsq(z,x,rcond=None)[0]

def distribution(a,observed=None):
    a=checked(a)
    require(a.ndim==1 and len(a)>0,'Distribution expected full finite vector')
    result=dict(zip(['min','q025','median','q975','max'],np.quantile(a,[0,.025,.5,.975,1]).tolist()))
    result.update({'n':len(a),'positive_fraction':float(np.mean(a>0)),'negative_fraction':float(np.mean(a<0)),
        'zero_fraction':float(np.mean(a==0))})
    if observed is not None: result['same_sign_fraction']=float(np.mean(np.sign(a)==np.sign(observed)))
    return result

def holm(p):
    p=checked(p,(4,)); require(np.all((p>=0)&(p<=1)),'Invalid P')
    ix=np.argsort(p,kind='stable'); q=np.minimum(1,np.maximum.accumulate(p[ix]*np.arange(4,0,-1)))
    answer=np.empty(4);answer[ix]=q
    return answer

def effects(loadings,maps,z):
    x=np.atleast_2d(checked(loadings)); maps=checked(maps,(68,3)); z=checked(z,(68,4))
    require(x.shape[1]==68 and np.linalg.matrix_rank(z)==4,'Base design rank/shape')
    xr=rankdata(x,axis=1,method='average'); yr=rankdata(maps,axis=0,method='average')
    ex=residual(xr.T,z).T; ey=residual(yr,z)
    return {'pearson':corr(x,maps),'spearman':corr(xr,yr),'partial_spearman':corr(ex,ey)},ex,ey

def analyze(observed,boot,raw_null,conditional_null,maps,z):
    observed=checked(observed,(68,));boot=checked(boot,(2000,68))
    raw_null=checked(raw_null,(9999,68));conditional_null=checked(conditional_null,(9999,68))
    maps=checked(maps,(68,3));z=checked(z,(68,4))
    obs,ox,oy=effects(observed,maps,z)
    be,bx,by=effects(boot,maps,z)
    require(np.max(np.abs(conditional_null@z))<1e-7,'Conditional surrogate nuisance orthogonality')
    raw_alt=corr(raw_null,maps[:,1])[...,0]
    cond_alt=corr(conditional_null,oy[:,1])[...,0]
    raw_effect=float(obs['pearson'][0,1]); cond_effect=float(obs['partial_spearman'][0,1])
    raw_count=int(np.sum(abs(raw_alt)>=abs(raw_effect)-1e-12)); cond_count=int(np.sum(abs(cond_alt)>=abs(cond_effect)-1e-12))
    tests=[{'statistic':'raw_Pearson','effect':raw_effect,'tail_count':raw_count,'p_spatial':(1+raw_count)/10000},
           {'statistic':'gradient_partial_Spearman','effect':cond_effect,'tail_count':cond_count,'p_spatial':(1+cond_count)/10000}]
    result={'associations':{},'paired_absolute_strength':{},'alternative_SERT_new_spatial_tests':tests,
            'bootstrap_draws':2000,'all_draws_retained':True,'PC1_refitted':False}
    arrays={'raw_alternative_null_effects':raw_alt,'conditional_alternative_null_effects':cond_alt,
            'ranked_receptor_residuals':oy}
    for statistic in obs:
        result['associations'][statistic]=[{'map':name,'effect':float(obs[statistic][0,i]),
            'participant_bootstrap95':distribution(be[statistic][:,i],obs[statistic][0,i])} for i,name in enumerate(MAPS)]
        arrays['observed_'+statistic]=obs[statistic][0]; arrays['bootstrap_'+statistic]=be[statistic]
    for statistic in ['pearson','partial_spearman']:
        delta=np.abs(be[statistic][:,:2])-np.abs(be[statistic][:,2,None])
        od=np.abs(obs[statistic][0,:2])-abs(obs[statistic][0,2])
        result['paired_absolute_strength'][statistic]=[{'contrast':MAPS[i]+'_minus_DAT_absolute_strength',
             'difference':float(od[i]),'paired_bootstrap95':distribution(delta[:,i])} for i in range(2)]
        arrays['paired_absolute_'+statistic]=delta
    ranks=rankdata(maps,axis=0,method='average')
    rd=(ranks[:,2]-ranks[:,2].mean())/ranks[:,2].std(ddof=0)
    zd=np.column_stack([z,rd]); require(np.linalg.matrix_rank(zd)==5,'DAT design rank5 failed')
    xd=residual(rankdata(observed,method='average'),zd)
    yd=residual(ranks[:,:2],zd)
    bd=residual(rankdata(boot,axis=1,method='average').T,zd).T
    unique=corr(xd,yd)[0]; bu=corr(bd,yd)
    retain=np.sum(yd*yd,axis=0)/np.sum(oy[:,:2]*oy[:,:2],axis=0)
    require(np.all(retain>1e-12),'DAT leaves degenerate SERT targets')
    result['DAT_conditioned']={'design_rank':5,'condition_number':float(np.linalg.cond(zd)),
        'PC1_residual_variance_fraction_retained':float(xd@xd/(ox[0]@ox[0])),
        'rows':[{'map':MAPS[i],'partial_Spearman':float(unique[i]),
            'participant_bootstrap95':distribution(bu[:,i],unique[i]),
            'target_residual_variance_fraction_retained':float(retain[i]),
            'target_VIF_from_DAT':float(1/retain[i]),
            'low_residual_variance_warning':bool(retain[i]<.1)} for i in range(2)],
        'spatial_P':'NOT_COMPUTED_RANK5_EFFECT_ONLY'}
    arrays.update({'DAT_design':zd,'DAT_target_residuals':yd,'DAT_PC1_residual':xd,
         'DAT_bootstrap_PC1_residuals':bd,'DAT_observed_effects':unique,'DAT_bootstrap_effects':bu})
    return result,arrays

def atlas_relationships(maps,z):
    rank=rankdata(maps,axis=0,method='average'); resid=residual(rank,z)
    return {'map_order':MAPS,'pearson':corr(maps.T,maps),'spearman':corr(rank.T,rank),
            'gradient_partial_spearman':corr(resid.T,resid),'P_values':'NOT_COMPUTED_FIXED_MAP_DESCRIPTIVE'}

def synthetic_tests():
    rng=np.random.default_rng(2026102607); checks=[]
    def check(name,condition): require(condition,name);checks.append(name)
    maps=rng.normal(size=(68,3)); x=rng.normal(size=(9,68))
    z=np.column_stack([np.ones(68),np.r_[np.ones(34),np.zeros(34)],rng.normal(size=(68,2))])
    e,ex,ey=effects(x,maps,z)
    check('Pearson agrees with scalar np.corrcoef',abs(e['pearson'][0,1]-np.corrcoef(x[0],maps[:,1])[0,1])<1e-12)
    check('Spearman scalar rank agreement',abs(e['spearman'][0,0]-np.corrcoef(rankdata(x[0]),rankdata(maps[:,0]))[0,1])<1e-12)
    q=np.linalg.qr(z)[0]; xx=rankdata(x[0]); yy=rankdata(maps[:,1]);
    check('Partial vs independentQR',abs(e['partial_spearman'][0,1]-np.corrcoef(xx-q@(q.T@xx),yy-q@(q.T@yy))[0,1])<1e-12)
    check('Residual orthogonality',np.max(abs(ex@z))<1e-10 and np.max(abs(z.T@ey))<1e-10)
    check('Holm4 knowncase',np.allclose(holm([.01,.04,.03,.8]),[.04,.09,.09,.8]))
    try: corr(np.ones(68),maps)
    except ValueError: checks.append('Degenerate constant rejected')
    else: raise ValueError('Constant not rejected')
    return {'status':'PASS','checks':checks,'real_data_used':False}
