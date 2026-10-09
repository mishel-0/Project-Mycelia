"""Scientific plots from the recorded MRI protocol (optional matplotlib)."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path);args=parser.parse_args()
    root=args.results;s=json.loads((root/'summary.json').read_text())
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    matrix=np.asarray(s['models']['physiology_only']['heldout']['confusion_matrix'])
    ax=axs[0,0];ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(4),s['models']['physiology_only']['heldout']['confusion_labels']);ax.set_yticks(range(4),s['models']['physiology_only']['heldout']['confusion_labels'])
    ax.set(title='A · External physiology readout confusion',xlabel='Predicted dataset label',ylabel='Actual folder label')
    for (y,x),value in np.ndenumerate(matrix):ax.text(x,y,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else '#162e3b')
    names=['physiology_only','image_only','image_plus_physiology','mean_intensity_only','majority_baseline','permuted_training_labels']
    labels=['Physiology only','Direct pixels','Pixels + physiology','Mean intensity','Majority baseline','Permuted train labels']
    r=[s['models'][name]['heldout'] for name in names]
    accuracy=np.array([v['accuracy'] for v in r])*100
    low=np.array([v['accuracy_95pct_Wilson_CI'][0] for v in r])*100
    high=np.array([v['accuracy_95pct_Wilson_CI'][1] for v in r])*100
    ax=axs[0,1];ax.errorbar(accuracy,np.arange(6),xerr=np.vstack([accuracy-low,high-accuracy]),fmt='o',capsize=4,color='#126c58');ax.set_yticks(np.arange(6),labels);ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Held-out accuracy (%) · 95% Wilson intervals',title='B · Same simple readout and held-out split');ax.grid(axis='x',alpha=.2)
    for y,x in enumerate(accuracy):ax.text(x+4,y,f'{x:.2f}%',va='center',fontsize=8)
    colors=['#126c58','#b6374d','#2b6ab0','#b3700f']
    for color,label in zip(colors,s['representatives']):
        with (root/'examples'/label/'timeseries.csv').open() as f:rows=list(csv.DictReader(f))
        t=np.array([float(row['time_min']) for row in rows])
        axs[1,0].plot(t,[float(row['tip_pressure_MPa']) for row in rows],label=label,color=color)
        axs[1,1].plot(t,[float(row['permanent_extension_um']) for row in rows],label=label,color=color)
    axs[1,0].set(title='C · Tip turgor, first held-out example per class',ylabel='Turgor (MPa)',xlabel='Exposure time (min)')
    axs[1,1].set(title='D · Permanent wall extension, same four images',ylabel='Extension (µm)',xlabel='Exposure time (min)')
    for ax in axs[1]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('MYCELIA · 7,200 brain images through the dimensional core\nCore untrained · classification is an external readout · no clinical validation',fontsize=14)
    fig.savefig(root/'medical-image-plots.svg');fig.savefig(root/'medical-image-plots.png',dpi=160);plt.close(fig)
    report=root/'report.html';text=report.read_text()
    if 'medical-image-plots.svg' not in text:
        text=text.replace('<h2>First clean held-out example per class</h2>','<h2>Recorded responses and readout comparisons</h2><img src="medical-image-plots.svg" alt="Confusion matrix, held-out accuracy, turgor and wall extension"><p>Intervals assume independent images. Exact duplicates were removed; patient-level independence is unknown.</p><h2>First clean held-out example per class</h2>')
        report.write_text(text)
    print(root/'medical-image-plots.svg')


if __name__=='__main__':main()
