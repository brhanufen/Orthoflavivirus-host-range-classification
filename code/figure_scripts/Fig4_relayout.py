from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parent
data=root/'figure_data'; out=root/'output'; out.mkdir(exist_ok=True)
ab=pd.read_csv(data/'Fig4A_ablation_results.csv'); length=pd.read_csv(data/'Fig4B_length_control.csv')
cv=pd.read_csv(data/'Fig4C_cv_summary.csv'); loso=pd.read_csv(data/'Fig4D_loso_cv_results.csv')
BLUE='#1F77B4'; ORANGE='#FF7F0E'; GREEN='#2CA02C'; RED='#D62728'
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':10,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':0.7,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(8.1,10.54))
gs=GridSpec(2,3,figure=fig,height_ratios=[1.0,2.25],hspace=0.68,wspace=0.58,left=0.12,right=0.985,top=0.92,bottom=0.085)
axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2]); axD=fig.add_subplot(gs[1,:])
bw=0.36
# A: feature ablation
features=['Composition (n=2)','Dinucleotide (n=34)','RSCU (n=61)','Combined (n=97)']; short=['Comp.\n(n=2)','Dinuc.\n(n=34)','RSCU\n(n=61)','Combined\n(n=97)']; xx=np.arange(4)
for model,color,offset in [('LR',BLUE,-bw/2),('RF',ORANGE,bw/2)]:
 vals=[float(ab[(ab.Feature_Set==feat)&(ab.Model==model)]['PR-AUC (ISFV)'].iloc[0]) for feat in features]
 bars=axA.bar(xx+offset,vals,width=bw,color=color,zorder=2)
 for b,v in zip(bars,vals): axA.text(b.get_x()+b.get_width()/2,v+0.018,f'{v:.2f}',ha='center',va='bottom',color=color,fontweight='bold',fontsize=7.5)
axA.set_xticks(xx,short); axA.set_ylim(0,1.14); axA.set_ylabel('PR-AUC (ISFV)'); axA.set_title('Feature Set Ablations',fontweight='bold',pad=7); axA.text(-0.19,1.12,'A',transform=axA.transAxes,fontsize=14,fontweight='bold',va='top'); axA.spines[['top','right']].set_visible(False)
# B: length control
conds=['With Length (n=97)','Without Length (n=96)']; labels=['With length\n(n=97)','Without length\n(n=96)']; xxB=np.arange(2)
for model,color,offset in [('LR',BLUE,-bw/2),('RF',ORANGE,bw/2)]:
 vals=[float(length[(length.Condition==cond)&(length.Model==model)]['PR-AUC (ISFV)'].iloc[0]) for cond in conds]
 bars=axB.bar(xxB+offset,vals,width=bw,color=color,zorder=2)
 for b,v in zip(bars,vals): axB.text(b.get_x()+b.get_width()/2,v+0.018,f'{v:.2f}',ha='center',va='bottom',color=color,fontweight='bold',fontsize=7.5)
axB.set_xticks(xxB,labels); axB.set_ylim(0,1.14); axB.set_ylabel('PR-AUC (ISFV)'); axB.set_title('Length-Removed Control',fontweight='bold',pad=7); axB.text(-0.19,1.12,'B',transform=axB.transAxes,fontsize=14,fontweight='bold',va='top'); axB.spines[['top','right']].set_visible(False)
# C: CV means and SD
metrics=['PR-AUC (ISFV)','ROC-AUC','Balanced Acc','MCC']; short_metrics=['PR-AUC','ROC-AUC','Bal. Acc.','MCC']; xxC=np.arange(4)
for model,color,offset,mean_col,sd_col in [('LR',BLUE,-bw/2,'LR_Mean','LR_SD'),('RF',ORANGE,bw/2,'RF_Mean','RF_SD')]:
 means=[float(cv.loc[cv.Metric==m,mean_col].iloc[0]) for m in metrics]; sds=[float(cv.loc[cv.Metric==m,sd_col].iloc[0]) for m in metrics]
 axC.bar(xxC+offset,means,width=bw,yerr=sds,capsize=2.5,color=color,error_kw={'elinewidth':0.8,'capthick':0.8},zorder=2)
axC.set_xticks(xxC,short_metrics); axC.tick_params(axis='x',labelrotation=20); [t.set_ha('right') for t in axC.get_xticklabels()]; axC.set_ylim(0.75,1.145); axC.set_ylabel('Score (mean ± SD)'); axC.set_title('5-Fold Species-Stratified CV',fontweight='bold',pad=7); axC.text(-0.19,1.12,'C',transform=axC.transAxes,fontsize=14,fontweight='bold',va='top'); axC.spines[['top','right']].set_visible(False)
for i in range(4):
 x1=xxC[i]-bw/2; x2=xxC[i]+bw/2
 top=max(float(cv.loc[cv.Metric==metrics[i],'LR_Mean'].iloc[0])+float(cv.loc[cv.Metric==metrics[i],'LR_SD'].iloc[0]),float(cv.loc[cv.Metric==metrics[i],'RF_Mean'].iloc[0])+float(cv.loc[cv.Metric==metrics[i],'RF_SD'].iloc[0]))
 y=max(1.065,top+0.012); axC.plot([x1,x1,x2,x2],[y-0.006,y,y,y-0.006],color='black',lw=0.7,clip_on=False); axC.text((x1+x2)/2,y+0.004,'ns',ha='center',va='bottom',fontsize=7.5,fontstyle='italic',clip_on=False)
shared=[Patch(facecolor=BLUE,label='Logistic Regression'),Patch(facecolor=ORANGE,label='Random Forest')]
fig.legend(handles=shared,loc='upper center',bbox_to_anchor=(0.55,0.64),ncol=2,frameon=True,framealpha=0.95,edgecolor='#CCCCCC',borderpad=0.35,handlelength=1.5,columnspacing=1.4)
# D: LOSO species classifier
loso=loso.sort_values('mean_P_dual_host',ascending=False).reset_index(drop=True); name_map={'t ho virus':"T'Ho Virus"}; names=[name_map.get(s,s.title()) for s in loso.species]; colors=[RED if x=='WRONG' else GREEN for x in loso.prediction]; y=np.arange(len(loso))
axD.barh(y,loso.mean_P_dual_host,color=colors,height=0.72,zorder=2); axD.set_yticks(y,[f'{n} (n={k})' for n,k in zip(names,loso.n_genomes)]); axD.invert_yaxis(); axD.set_xlim(0,1.07); axD.set_xlabel('Mean Predicted P(dual-host)'); axD.set_title('LOSO-CV: ISFV Species Classification',fontweight='bold',pad=7); axD.text(-0.09,1.08,'D',transform=axD.transAxes,fontsize=14,fontweight='bold',va='top'); axD.axvline(0.5,color='black',ls='--',lw=1.1,zorder=3); axD.spines[['top','right']].set_visible(False)
handles=[Patch(facecolor=GREEN,label='Correctly classified (n=20)'),Patch(facecolor=RED,label='Misclassified (n=4)'),Line2D([0],[0],color='black',ls='--',lw=1.1,label='Decision boundary')]
axD.legend(handles=handles,loc='lower right',frameon=True,framealpha=0.95,edgecolor='#CCCCCC',fontsize=7.5)
fig.text(0.5,0.025,'Wilcoxon signed-rank test: *p < 0.05, ns = not significant (n = 5 folds, power-limited)',ha='center',va='bottom',fontsize=7,fontstyle='italic',color='#555555')
fig.savefig(out/'Fig4_publication.png',dpi=300,bbox_inches='tight',facecolor='white'); fig.savefig(out/'Fig4_publication.tiff',dpi=600,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(out/'Fig4_publication.pdf',bbox_inches='tight',facecolor='white')
print('Wrote updated figure files'); print('D correct/wrong:',loso.prediction.value_counts().to_dict())
