"""Reproduce figures from the reported manuscript tables, without new experiments."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
BASE=Path(__file__).resolve().parents[1];D=BASE/'figures';D.mkdir(exist_ok=True)
DATA=json.loads((BASE/'scripts/figure_data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','text.color':'#15283a','savefig.facecolor':'white'})
B='#2366a3';O='#c97021';G='#3d8666';GRAY='#667582'
def save(f,n):
 for ext in ['png','pdf']:f.savefig(D/(n+'.'+ext),dpi=220,bbox_inches='tight')
 plt.close(f)
f,a=plt.subplots(figsize=(14,7));a.set(xlim=(0,14),ylim=(0,7));a.axis('off')
def box(x,y,title,detail,color=B,fill='#eef4fa'):
 a.add_patch(FancyBboxPatch((x,y),3.6,1.12,boxstyle='round,pad=.13',fc=fill,ec=color,lw=1.5));a.text(x+1.8,y+.80,title,ha='center',va='center',weight='bold',size=12);a.text(x+1.8,y+.34,detail,ha='center',va='center',size=10.5,linespacing=1.4)
def arrow(p,q,dash=False):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=16,lw=1.7,color=GRAY if dash else B,linestyle='--' if dash else '-'))
a.text(7,6.55,'Data-driven VSM / VSD: proposed architecture',ha='center',weight='bold',size=17)
box(.5,4.65,'Historical event logs','Recorded cases, activities,\ntimestamps and resources');box(5.2,4.65,'Current-state reconstruction','Process mining and VSM\nOperational indicators');box(9.9,4.65,'Structural / model diagnosis','Activity DFG and ML\nXAI: proposed extension');box(9.9,2.62,'Scenario assumptions','Capacity, routing, workload\nand quality alternatives');box(5.2,2.62,'Monte Carlo scenarios','Predefined scenario factors\nCompare individual KPIs',O,'#fbf2e8');box(.5,2.62,'Future-state VSD','Proposed translation to VSD\nStage-resolved validation required')
for p,q in [((4.25,5.2),(5.02,5.2)),((8.94,5.2),(9.72,5.2)),((11.7,4.5),(11.7,3.9)),((9.72,3.18),(8.94,3.18)),((5.02,3.18),(4.25,3.18))]:arrow(p,q)
a.add_patch(FancyBboxPatch((.5,.6),13,1.05,boxstyle='round,pad=.13',fc='#f5f6f7',ec=GRAY,lw=1.2,linestyle='--'));a.text(7,1.3,'Deployment extension: physical implementation and subsequent operational observations',ha='center',weight='bold',size=12);a.text(7,.88,'Future work; longitudinal implementation and live physical-digital synchronization were not evaluated.',ha='center',size=10.5)
arrow((2.3,2.45),(2.3,1.83),True);arrow((.2,1.1),(.2,5.25),True);arrow((.2,5.25),(.35,5.25),True);a.text(7,2.02,'Demonstrated components: activity diagnostics and predefined scenarios',ha='center',size=10.5,color=GRAY);save(f,'framework_architecture')
c=DATA['configurations'];lt=DATA['lead_time_reduction'];th=DATA['throughput_gain'];w=DATA['WIP_reduction'];vals=DATA['G5_changes']
series=[(lt,B,'Lead time reduction'),(th,O,'Throughput-proxy increase'),(w,G,'Workload-proxy reduction')]
f,axs=plt.subplots(1,2,figsize=(14,5.2),gridspec_kw={'width_ratios':[1,1.15]});a=axs[0];bars=a.bar(range(5),vals,color=[B,B,G,G,O],width=.66);a.set_xticks(range(5),['Lead time\nreduction','Workload proxy\nreduction','Defect rate\nreduction','Rework rate\nreduction','Throughput proxy\nincrease']);a.set(ylim=(0,85),ylabel='Relative change (%)');a.set_title('A. Synthetic simulation: G5 vs G0',loc='left',pad=16)
for p,v in zip(bars,vals):a.text(p.get_x()+p.get_width()/2,v+1.7,f'{v:.2f}%',ha='center',size=10)
a.grid(axis='y',alpha=.16);a.set_axisbelow(True);a=axs[1]
for v,col,lab in series:a.plot(c,v,marker='o',lw=2.4,color=col,label=lab)
a.set_title('B. Cumulative simulation configurations',loc='left',pad=16);a.set(ylim=(-2,84),ylabel='Model change relative to G0 (%)',xlabel='Cumulative configuration');a.legend(loc='upper left',frameon=False,fontsize=10);a.grid(alpha=.16);f.text(.5,.005,'Simulation results under the specified synthetic operating conditions; values reproduce manuscript tables.',ha='center',size=10,color=GRAY);f.tight_layout(rect=[0,.04,1,1]);save(f,'paper_synthetic_results_final_dashboard')
f,a=plt.subplots(figsize=(12.5,4.8))
for v,col,lab in series:a.plot(c,v,marker='o',lw=2.4,color=col,label=lab)
a.set_xticks(range(6),['G0\nBaseline','G1\nManual VSM','G2\nPM-VSM','G3\nAssumed factors','G4\n+ Scenario simulation','G5\nAssumed factors']);a.set(ylabel='Model change relative to G0 (%)',ylim=(-3,82));a.legend(loc='upper left',frameon=False);a.set_title('Synthetic simulation: cumulative configuration comparison',loc='left',pad=15);a.grid(alpha=.16);f.tight_layout();save(f,'fig_6_ablation_improvement')
labels=['BPI2017\nDelay','BPI2017\nBottleneck','BPI2017\nRework','BPI2019\nDelay','BPI2019\nBottleneck','BPI2019\nRework','DQI\nDelay','DQI\nBottleneck','DQI\nRework'];no=[np.nan if x is None else x for x in DATA['RF_without_CNA']];yes=[np.nan if x is None else x for x in DATA['RF_with_CNA']]
f=plt.figure(figsize=(16,7));gs=f.add_gridspec(2,3,height_ratios=[3,1.1],hspace=.55,wspace=.32);a=f.add_subplot(gs[0,0]);density=DATA['network_density'];bars=a.bar(['BPI2017','BPI2019','DQI'],density,color=[B,O,G],width=.62)
for p,v in zip(bars,density):a.text(p.get_x()+p.get_width()/2,v+.018,f'{v:.4f}',ha='center')
a.set(ylim=(0,.73),ylabel='Directed network density');a.set_title('A. Activity-network structure',loc='left',pad=16);a.grid(axis='y',alpha=.16);a.set_axisbelow(True);a=f.add_subplot(gs[0,1:]);x=np.arange(9);wid=.36
for values,off,col,lab in [(no,-wid/2,B,'Without CNA'),(yes,wid/2,O,'With CNA')]:
 bars=a.bar(x+off,values,width=wid,color=col,label=lab)
 for p,v in zip(bars,values):
  if np.isfinite(v):a.text(p.get_x()+p.get_width()/2,v+.025,f'{v:.3f}',ha='center',size=8.7)
for i,v in enumerate(no):
 if not np.isfinite(v):a.text(i,.08,'NA',ha='center',color=GRAY,size=11,weight='bold')
a.set_xticks(x,labels,fontsize=9);a.set(ylim=(0,1.24),ylabel='F1-score');a.set_title('B. Random Forest: event-log / proxy classification',loc='left',pad=16);a.legend(frameon=False,loc='upper left',ncol=2,fontsize=10);a.grid(axis='y',alpha=.16);a.set_axisbelow(True);a=f.add_subplot(gs[1,:]);a.axis('off');a.text(0,1.12,'C. Activities with the greatest reported betweenness centrality',weight='bold',size=12)
for x0,name,activity,meta in [(i*.35,['BPI2017','BPI2019','DQI'][i],v['activity'],f"{v['nodes']} nodes | {v['edges']} edges") for i,v in enumerate(DATA['critical_activities'])]:a.text(x0,.75,name,weight='bold',size=12);a.text(x0,.39,activity,size=10.7);a.text(x0,.06,meta,size=10,color=GRAY)
f.text(.5,.015,'NA: single-class test target. F1 values describe the specified event-log / proxy task, not prospective shop-floor forecasts.',ha='center',size=10,color=GRAY);f.subplots_adjust(bottom=.13,top=.94);save(f,'fig_real_datasets_experimental_results_dashboard')
(BASE/'scripts/reported_plot_values.json').write_text(json.dumps({'source':'Reported manuscript tables, not raw observations','configurations':c,'lead_time_reduction':lt,'throughput_gain':th,'WIP_reduction':w,'G5_changes':vals,'RF_without_CNA':[v if np.isfinite(v) else None for v in no],'RF_with_CNA':[v if np.isfinite(v) else None for v in yes]},indent=2))
