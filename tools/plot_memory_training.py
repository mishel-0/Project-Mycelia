"""Generate standalone scientific figures and report for completed memory training."""
import argparse,csv,html,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--dataset',type=Path)
    args=parser.parse_args();out=args.output;data=json.loads((out/'summary.json').read_text())
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    labels=data['heldout']['metrics']['labels'];heldout=data['heldout']['metrics'];full=data['full_data']['training_recall']
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    candidates=data['candidates'];validation=np.array([c['validation']['balanced_accuracy'] for c in candidates])*100
    axes[0].bar(range(len(candidates)),validation,color=['#556477' if c['name']!=data['selected_candidate'] else '#187b70' for c in candidates])
    axes[0].set_xticks(range(len(candidates)),[c['name'].replace('_','\n') for c in candidates]);axes[0].set_ylabel('Balanced accuracy (%)');axes[0].set_ylim(0,100)
    axes[0].set_title('A. Training-only validation\nCapacity selection; no Testing labels')
    for i,v in enumerate(validation):axes[0].text(i,v+2,f'{v:.1f}%',ha='center')
    values=np.array([heldout['accuracy'],full['accuracy']])*100
    axes[1].bar([0,1],values,color=['#187b70','#cc8b42']);axes[1].set_xticks([0,1],['Unseen Testing\nTraining-only model','All-data recall\nImages used for fitting']);axes[1].set_ylim(0,100);axes[1].set_ylabel('Correct predictions (%)')
    axes[1].set_title('B. Generalization versus training recall\nDifferent models and evaluation scopes')
    ci=np.array(heldout['accuracy_wilson_95_interval'])*100
    axes[1].errorbar([0],[values[0]],yerr=[[values[0]-ci[0]],[ci[1]-values[0]]],fmt='none',ecolor='#203142',capsize=4)
    for i,v in enumerate(values):axes[1].text(i,v+4,f'{v:.2f}%\n(n={heldout["n"] if i==0 else full["n"]:,})',ha='center',fontsize=9)
    matrix=np.array(heldout['confusion_matrix']);axes[2].spines[['top','right','bottom','left']].set_visible(False)
    axes[2].imshow(matrix,cmap='Blues');axes[2].set_xticks(range(4),labels,rotation=35,ha='right');axes[2].set_yticks(range(4),labels)
    axes[2].set_xlabel('Predicted label');axes[2].set_ylabel('Dataset label');axes[2].set_title('C. Held-out confusion matrix\nFrozen compartment/cord memory')
    for i in range(4):
        for j in range(4):axes[2].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.5 else '#233449')
    fig.savefig(out/'training-results.svg');fig.savefig(out/'training-results.png',dpi=160);plt.close(fig)
    examples=[]
    if args.dataset:
        rows=list(csv.DictReader((out/'heldout-predictions.csv').open()))
        folder=out/'examples';folder.mkdir(exist_ok=True)
        for label in labels:
            for correct in ('1','0'):
                row=next((r for r in rows if r['actual']==label and r['correct']==correct),None)
                if row:
                    name=label+('-correct' if correct=='1' else '-incorrect')+'.jpg'
                    shutil.copy2(args.dataset/row['file'],folder/name)
                    examples.append({'image':'examples/'+name,**row})
        (out/'examples.json').write_text(json.dumps(examples,indent=2))
    def esc(v):return html.escape(str(v))
    class_rows=''.join(f'<tr><td>{esc(label)}</td><td>{heldout["classes"][label]["count"]}</td><td>{heldout["classes"][label]["recall"]:.1%}</td><td>{heldout["classes"][label]["precision"]:.1%}</td></tr>' for label in labels)
    candidate_rows=''.join(f'<tr><td>{esc(c["name"])}</td><td>{c["validation"]["accuracy"]:.2%}</td><td>{c["validation"]["balanced_accuracy"]:.2%}</td></tr>' for c in candidates)
    checks=''.join(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>' for k,v in data['checks'].items())
    cards=''.join(f'<figure><img src="{esc(e["image"])}" alt="{esc(e["actual"])} example"><figcaption>Dataset: <b>{esc(e["actual"])}</b><br>Prediction: <b>{esc(e["prediction"])}</b><br>{"Correct" if e["correct"]=="1" else "Incorrect"}<br><small>{esc(e["file"])}</small></figcaption></figure>' for e in examples)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MYCELIA full-data memory training</title><style>body{{font:16px/1.6 system-ui;color:#243547;max-width:1200px;margin:32px auto;padding:0 24px;background:#f5f8fa}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;background:white;max-width:100%;margin:16px 0}}td,th{{padding:8px 16px;border-bottom:1px solid #dde5eb;text-align:left}}.chart{{width:100%;background:white}}.examples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}figure{{margin:0;background:white;padding:12px}}figure img{{height:190px;width:100%;object-fit:contain;background:#111}}small{{overflow-wrap:anywhere}}code,pre{{background:#e4edf1;padding:4px;overflow:auto}}a{{color:#096e65}}</style>
    <h1>MYCELIA: trained compartment and cord memory</h1>
    <p><b>Unseen-image accuracy: {heldout['accuracy']:.2%}</b> ({heldout['correct']:,}/{heldout['n']:,}; 95% Wilson interval {heldout['accuracy_wilson_95_interval'][0]:.2%}–{heldout['accuracy_wilson_95_interval'][1]:.2%}). Selected configuration: <b>{esc(data['selected_candidate'])}</b>.</p>
    <p>The final all-data predictor learned <b>{data['full_data']['images']:,} original images</b> over {data['epochs']} passes ({data['full_data']['learning']['exposures']:,} lessons). Its {full['accuracy']:.2%} recall measures predictions on images used for fitting. This is a separate model from the held-out test.</p>
    <p>No CNN, neural network, backpropagation or external learned readout. The algorithm is bio-inspired online prototype memory stored in resource-funded graph state. The dimensional physiology core remains separate and untrained. Patient-independent and clinical validation are absent.</p>
    <img class="chart" src="training-results.svg" alt="Validation selection, evaluation scopes and held-out confusion matrix">
    <h2>Protocol</h2><p>Raw counts: {esc(data['raw_counts'])}. Clean counts: {esc(data['clean_counts'])}. Duplicate removals: {esc(data['audit']['removed'])}. Encoded label conflicts: {data['audit']['encoded_label_conflicts']}.</p>
    <p>Training-only selection used {data['selection_protocol']['selection_pool']:,} inner-training images and {data['selection_protocol']['validation']:,} validation images. Every candidate used {data['epochs']} shuffled passes and seed {data['seed']}. The winner was refitted on all clean Training images and tested frozen. A fresh winner was then fitted on every original row, including Testing and duplicates.</p>
    <table><tr><th>Candidate</th><th>Validation accuracy</th><th>Balanced accuracy (selection rule)</th></tr>{candidate_rows}</table>
    <h2>Held-out class performance</h2><table><tr><th>Dataset label</th><th>Images</th><th>Recall</th><th>Precision</th></tr>{class_rows}</table>
    <h2>What learned state contributes</h2><p>Erasing receptor traces: {data['heldout']['no_receptor_memory']['abstentions']:,}/{heldout['n']:,} abstentions. Removing learned cord traces: {data['heldout']['no_learned_cord_traces']['accuracy']:.2%} accuracy versus intact {heldout['accuracy']:.2%}. Majority-class baseline: {data['heldout']['majority_baseline']['accuracy']:.2%}. Receptor ablation supports dependence on the stored imprints; cord ablation quantifies its contribution and does not establish fungal cognition.</p>
    <h2>Deterministic examples</h2><p>First correct and first incorrect held-out example per class in filename order, where available. Full predictions are linked below.</p><div class="examples">{cards}</div>
    <h2>Integrity</h2><table><tr><th>Check</th><th>Result</th></tr>{checks}</table><p>Held-out maximum budget residual: {data['heldout']['learning']['max_absolute_budget_error']:.3g}; full-data maximum: {data['full_data']['learning']['max_absolute_budget_error']:.3g}. No source image is used as a persistent predictor cache.</p>
    <h2>Files and prediction</h2><p><a href="full-data-memory.json">All-data model</a> · <a href="heldout-memory.json">Training-only evaluation model</a> · <a href="summary.json">Full metrics and evidence</a> · <a href="heldout-predictions.csv">Unseen-image predictions</a> · <a href="full-data-recall-predictions.csv">All-data training recall predictions</a> · <a href="image-manifest.csv">Source/input hashes</a></p>
    <pre>python tools/predict_memory_image.py --memory runs/full-data-memory/full-data-memory.json --image /path/to/image.jpg</pre>
    <p>{esc(data['limits'])} Exact duplicates were removed for evaluation; transformed duplicates and unknown patient overlap remain possible. The algorithm cannot guarantee perfect retention with a bounded number of colony imprints. Mismatch values are not calibrated probabilities.</p></html>'''
    (out/'report.html').write_text(text)
    print(out/'report.html')


if __name__=='__main__':main()
