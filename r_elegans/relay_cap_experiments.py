"""E092 experimental design and prior checks; bounds are engineering priors."""
import numpy as np

TARGET=.02
CAPS=('original','aiyr_2p5')
SEEDS=(9200,9201)
FACTORS=(.8,.9,1.,1.025,1.05,1.10,1.25)


def edge_coordinate(spec,ids):
    post,pre=np.nonzero(spec.chemical)
    indices=np.flatnonzero((post==ids.index('AIYL'))&(pre==ids.index('AIYR')))
    if len(indices)!=1:raise ValueError('Exactly one existing AIYR-to-AIYL chemical coordinate required')
    return int(spec.channels.sum()+indices[0])


def upper_for(cap,upper,edge):
    if cap not in CAPS:raise ValueError('Unknown prior policy')
    result=np.array(upper,copy=True)
    if cap=='aiyr_2p5':result[edge]+=np.log(1.25)
    return result


def eligible(result):
    m=result.get('metrics',{})
    return bool(result.get('status')=='completed' and m.get('safe') and
                m.get('baseline_and_recovery_preserved') and m.get('useful_response_preserved'))


def panel_assessment(rows):
    complete=len(rows)==10 and all(r.get('status')=='completed' for r in rows)
    if not complete:return dict(complete=False,engineering_criteria_met=False)
    normal=rows[:8];same=len({r['checkpoint_sha256'] for r in rows})==1
    numerical=[]
    for r in rows[6:8]:
        error=max(abs(a-b) for a,b in zip(rows[0]['metrics']['response_amplitude_ratios'],r['metrics']['response_amplitude_ratios']))
        numerical.append(dict(grid=r['grid'],difference=error,passed=error<=.25*TARGET*.001))
    changes=[x for r in normal for x in r['metrics']['relay_response_mean_changes']]
    direction=bool(changes and (all(x>0 for x in changes) or all(x<0 for x in changes)))
    blocked=eligible(rows[8]) and all(abs(r-1)<=.25*TARGET*.001 for r in rows[8]['metrics']['response_amplitude_ratios'])
    passed=bool(same and all(eligible(r) and r['metrics']['partial_attenuation_passed'] for r in normal)
                and all(r['passed'] for r in numerical) and direction and blocked)
    return dict(complete=True,same_checkpoint=same,numerical_checks=numerical,
        direction_consistent=direction,sensory_block_passed=bool(blocked),engineering_criteria_met=passed,
        scope='Relay diagnostic criteria only; no automatic fitting-parent, biological or full start/stop promotion')
