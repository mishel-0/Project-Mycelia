"""Reward/punishment teaching of AssociativeMycelium on the brain-tumor MRI data.

Each training image is a trial: the colony population predicts first, then a
correct guess is rewarded (fed imprint) and a wrong guess is punished (paid
anti-imprint of the wrongly chosen colony, then the true label is taught).
A plain imprint-only model trained on the same images and order is the control.
Research prototype only; not clinically validated.
"""
from __future__ import annotations
import argparse,time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from mycelia import AssociativeMycelium,MemoryConfig,__version__
from train_mycelium import LABELS,load_images,clean_splits,metrics,labels_for,write_json


def train(records,cues,indices,config,epochs,seed,*,mode,reward,penalty,output):
    model=AssociativeMycelium(LABELS,config);rng=np.random.default_rng(seed)
    history=[]
    for epoch in range(epochs):
        outcomes=Counter()
        for step,i in enumerate(rng.permutation(indices)):
            label=records[i]['label']
            if mode=='reinforce':outcomes[model.reinforce(cues[i],label,reward=reward,penalty=penalty)['outcome']]+=1
            else:model.learn(cues[i],label);outcomes['imprint']+=1
            if (step+1)%250==0:print(f'{mode}: epoch {epoch+1}/{epochs}, {step+1}/{len(indices)} {dict(outcomes)}',flush=True)
        trials=sum(outcomes.values())
        history.append({'epoch':epoch+1,'outcomes':dict(outcomes),'online_accuracy':outcomes['reward']/trials if mode=='reinforce' else None})
        write_json(output/f'{mode}-progress.json',history)
    model.freeze()
    return model,history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--reward',type=float,default=1.);parser.add_argument('--penalty',type=float,default=.5)
    parser.add_argument('--colonies-per-label',type=int,default=16)
    parser.add_argument('--train-per-class',type=int,default=0,help='0 = all clean Training images')
    args=parser.parse_args()
    if args.epochs<1 or not 0<args.reward<=1 or not 0<=args.penalty<=1:parser.error('invalid epochs, reward or penalty')
    args.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    records,cues=load_images(args.dataset);clean,audit=clean_splits(records)
    train_idx=clean['Training']
    if args.train_per_class:
        rng=np.random.default_rng(args.seed)
        train_idx=np.concatenate([rng.permutation([i for i in train_idx if records[i]['label']==l])[:args.train_per_class] for l in LABELS])
    test=clean['Testing'];config=MemoryConfig(colonies_per_label=args.colonies_per_label,consolidation=True)
    result={'version':__version__,'seed':args.seed,'epochs':args.epochs,'reward':args.reward,'penalty':args.penalty,'config':asdict(config),
            'train_images':len(train_idx),'test_images':len(test),'audit':audit,
            'limits':'Reward and punishment are external teaching cues (computational hypothesis). Image-level split, research only.'}
    for mode in ('imprint','reinforce'):
        model,history=train(records,cues,train_idx,config,args.epochs,args.seed,mode=mode,reward=args.reward,penalty=args.penalty,output=args.output)
        result[mode]={'history':history,'heldout':metrics(labels_for(records,test),model.predict_many(cues[test]))}
        model.save(args.output/f'{mode}-memory.json')
        print(f"{mode} held-out accuracy: {result[mode]['heldout']['accuracy']:.4f}",flush=True)
        write_json(args.output/'reward-punishment-results.json',result)
    result['seconds']=time.monotonic()-start
    write_json(args.output/'reward-punishment-results.json',result)


if __name__=='__main__':main()
