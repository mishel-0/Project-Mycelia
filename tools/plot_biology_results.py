"""Scientific plots for an existing biology validation panel (optional matplotlib)."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='folder produced by python -m mycelia.biology --validate')
    args = parser.parse_args()
    directory = args.results
    names = ['full','no_wall_creep','no_motor_delivery','no_exocytosis',
             'no_ATP_regeneration','no_membrane_water','closed_septa','hyperosmotic_shock']
    series = {}
    for name in names:
        with (directory/name/'timeseries.csv').open() as f:
            series[name] = [{k: float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    figure, axs = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors = {'full':'#126c58','no_wall_creep':'#747d86','no_membrane_water':'#b3700f','closed_septa':'#2b6ab0','hyperosmotic_shock':'#b6374d','no_motor_delivery':'#747d86','no_exocytosis':'#5a4199'}
    def line(ax, name, field, label=None, **kwargs):
        data=series[name]
        ax.plot([r['time_min'] for r in data],[r[field] for r in data],label=label or name.replace('_',' '),color=colors[name],**kwargs)
    for name in ['full','no_membrane_water','closed_septa','hyperosmotic_shock']:
        line(axs[0,0],name,'permanent_extension_um')
    line(axs[0,0],'no_wall_creep','permanent_extension_um','four inhibited mechanisms',linestyle=':')
    axs[0,0].set(title='A · Permanent wall extension',ylabel='Extension (µm)')
    axs[0,0].legend(fontsize=8)
    for name in ['full','hyperosmotic_shock']:
        line(axs[0,1],name,'tip_pressure_MPa')
    axs[0,1].axvline(10,color='#b6374d',linestyle=':',alpha=.6)
    axs[0,1].annotate('+0.6 MPa bath shock',xy=(10,.18),xytext=(13,.12),arrowprops={'arrowstyle':'->','color':'#b6374d'},color='#b6374d')
    axs[0,1].set(title='B · Turgor loss without regulatory recovery',ylabel='Tip turgor (MPa)')
    axs[0,1].legend(fontsize=8)
    for name in ['full','no_motor_delivery','no_exocytosis']:
        line(axs[1,0],name,'tip_apical_cargo_pmol')
    axs[1,0].set(title='C · Finite apical cargo reservoir',ylabel='Cargo (pmol monomer equivalents)')
    axs[1,0].legend(fontsize=8)
    for name in ['full','no_membrane_water','closed_septa']:
        line(axs[1,1],name,'flow_toward_tip_pL_min')
    axs[1,1].set(title='D · Flow coupled to wall expansion',ylabel='Flow toward tip (pL/min)')
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:
        ax.set_xlabel('Time (min)');ax.grid(alpha=.18)
    figure.suptitle('MYCELIA · Mechanistic physiology experiments\nIllustrative parameters; computational validation only',fontsize=14)
    figure.savefig(directory/'physiology-plots.svg')
    figure.savefig(directory/'physiology-plots.png',dpi=160)
    plt.close(figure)
    report = directory/'report.html'
    if report.exists():
        text=report.read_text()
        block='<h2>Mechanism time courses</h2><img src="physiology-plots.svg" alt="Extension, turgor, apical cargo and flow in matched intervention experiments"><p>Four zero-growth arms overlap in panel A; their separate values are listed in the table.</p>'
        if 'physiology-plots.svg' not in text:
            text=text.replace('<h2>Computational checks</h2>',block+'<h2>Computational checks</h2>')
            report.write_text(text)
    print(directory/'physiology-plots.svg')


if __name__=='__main__':
    main()
