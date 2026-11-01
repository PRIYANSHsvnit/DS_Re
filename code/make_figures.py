"""Create manuscript figures directly from the stored experimental results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'paper/results';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':350})

# Each dot is a dependent repeated split, not a separate clinical sample.
d=pd.read_csv(ROOT/'results/paired_smote/per_split.csv')
d=d[~d.time_feature].copy(); delta=d.global_contaminated_accuracy-d.train_only_accuracy
fig,ax=plt.subplots(figsize=(3.45,2.5))
ax.hist(delta,bins=np.arange(-.16,.181,.011111),color='#336f9e',edgecolor='white',linewidth=.5)
ax.axvline(0,color='#333',linestyle='--',lw=1);ax.axvline(delta.mean(),color='#c2512e',lw=2,label=f'Mean difference: {delta.mean():+.3f}')
ax.set(xlabel='Accuracy difference (contaminated − training-only)',ylabel='Number of splits');ax.xaxis.label.set_size(7);ax.tick_params(labelsize=7);ax.legend(frameon=False,fontsize=7);fig.tight_layout();fig.savefig(OUT/'fig_paired_smote.png');plt.close(fig)

# External discrimination and average 180-day event risk, fixed models.
b=pd.read_csv(ROOT/'results/support_bootstrap/summary.csv')
names=['CoxPH','RSF','GBSurv'];labels=['Cox model','Survival forest','Boosted survival'];c=b[(b.model.isin(names))&(b.metric=='harrell_c')].set_index('model').loc[names]
fig,(ax,bx)=plt.subplots(2,1,figsize=(3.45,3.8))
y=np.arange(3);v=c.estimate.to_numpy();ax.errorbar(v,y,xerr=[v-c.ci_low.to_numpy(),c.ci_high.to_numpy()-v],fmt='o',color='#24618d',capsize=3,lw=1.5)
ax.axvline(.5,color='gray',ls='--',lw=1);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set(xlim=(.49,.62),xlabel="Harrell's C");ax.set_title('(a) External discrimination',fontsize=9)
r=b[(b.model.isin(names))&(b.metric=='mean_predicted')].set_index('model').loc[names].estimate.to_numpy()*100
bx.barh(y,r,color=['#3678a5','#6298b9','#9bc1d1']);bx.axvline(27.023,color='#b54f32',ls='--',lw=1.5,label='Observed 27.0%');bx.set_yticks(y,labels);bx.invert_yaxis();bx.set(xlim=(0,60),xlabel='180-day mortality (%)');bx.legend(frameon=False,fontsize=7,loc='lower right');bx.set_title('(b) Mean predicted risk',fontsize=9)
for z in (ax,bx): z.tick_params(labelsize=7); z.xaxis.label.set_size(8)
fig.tight_layout();fig.savefig(OUT/'fig_external_support.png');plt.close(fig)

# Feature set estimates by split; each line is an independently fitted model.
d=pd.read_csv(ROOT/'results/feature_groups/feature_group_metrics.csv'); groups=['Random 80/20','Held out: Annual_Checkup','Held out: Hospital_Inpatient','Held out: Preventive_Cardiology'];features=['Demographics and lifestyle','Plus comorbidities and vitals','Plus labs, ECG and echo','All candidate predictors'];xs=np.arange(4)
fig,ax=plt.subplots(figsize=(3.45,2.8));colors=['#16618b','#b4512d','#53835a','#775393']; markers=['o','s','^','D']
for i,(g,c,m) in enumerate(zip(groups,colors,markers)):
 z=d.set_index(['split','features']); vals=[z.loc[(g,f),'auroc'] for f in features];ax.plot(xs,vals,marker=m,color=c,lw=1.4,ms=5,label=['Random split','Annual checkup','Hospital inpatient','Preventive cardiology'][i])
ax.set_xticks(xs,['Demog.\n+ lifestyle','+ conditions\n+ vitals','+ labs, ECG\n+ echo','All 63\nfeatures']);ax.tick_params(labelsize=7);ax.set_ylim(.70,1.0);ax.set_ylabel('AUROC');ax.grid(axis='y',alpha=.25);ax.legend(ncol=2,frameon=False,fontsize=6,loc='lower right');fig.tight_layout();fig.savefig(OUT/'fig_feature_groups.png');plt.close(fig)
