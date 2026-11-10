"""Plot separate neural and movement discrepancies from an E050 fit."""
import argparse
import json
from pathlib import Path

import numpy as np

from r_elegans.data.paths import get_data_root
from r_elegans.data.paired_observations import SOURCE_POINTS,SOURCE_STEP_M,fluctuation_statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    get_data_root(args.output.parent)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((args.run/'summary.json').read_text())
    config=json.loads((args.run/'configuration.json').read_text())
    rows=summary['best']['rows']
    fig,axes=plt.subplots(len(rows),2,figsize=(10,3.3*len(rows)),squeeze=False)
    for i,row in enumerate(rows):
        medium,uid=row['medium'],row['uid']
        base=next(r for r in summary['baseline']['rows'] if r['uid']==uid and r['medium']==medium)['metrics']
        x=SOURCE_POINTS[1:-1]*SOURCE_STEP_M*1e6
        ax=axes[i,0]
        ax.plot(x,row['metrics']['measured_bend_std_rad'],'o-',label='recorded')
        ax.plot(x,base['predicted_bend_std_rad'],'s--',label='parent')
        ax.plot(x,row['metrics']['predicted_bend_std_rad'],'^:',label='selected')
        ax.set(xlabel='Nominal distance behind nose (micrometers)',ylabel='Unsigned chord-bend std (rad)',yscale='log',title=uid+' / '+medium)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        with np.load(args.run/'inputs'/(uid+'-window.npz')) as z,np.load(args.run/(medium+'-'+uid+'-prediction.npz')) as pred:
            observed=fluctuation_statistics(z['fluorescence'],z['valid'])
            modeled=fluctuation_statistics(config['arguments']['basal_brightness']+pred['indicator'][:,z['indices']],z['valid'])
            mask=observed['valid'] & (observed['mean']>1e-8)
            ax=axes[i,1]
            ax.scatter(observed['cv'][mask],modeled['cv'][mask],s=18)
            maximum=max(float(observed['cv'][mask].max()),float(modeled['cv'][mask].max()),.01)
            ax.plot([0,maximum],[0,maximum],'k--',alpha=.4)
            ax.set(xlabel='Recorded fluorescence CV',ylabel='Selected relative-emission CV',title='Named-neuron fluctuations; optical model assumed')
            ax.grid(alpha=.2)
    fig.suptitle('E050 training pilot: paired statistics, uncalibrated medium',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(args.output,dpi=160)
    print(args.output)


if __name__=='__main__':main()
