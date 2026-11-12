"""Freeze the E066 parent for a fresh connections-only amplitude stage."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--expected-parent',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--software-smoke',action='store_true')
    a=p.parse_args();source=a.source_run.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for root in (source,repo):
        if out==root or root in out.parents or out in root.parents:p.error('Separate external output required')
    from scripts.compare_rhythm_worker import verify_inputs
    from scripts.assess_wet_surface import WINNER_SHA256,READOUT_SHA256
    metadata=verify_inputs(source/'inputs')
    summary=json.loads((source/'summary.json').read_text())
    if (summary.get('experiment')!='E066' or not summary['execution_valid']
            or file_sha256(source/'inputs/candidate.npz')!=a.expected_parent
            or metadata['candidate_sha256']!=a.expected_parent
            or metadata['sha256']['old.npz']!=WINNER_SHA256 or metadata['sha256']['actor.npz']!=READOUT_SHA256):
        raise ValueError('Pinned valid hybrid source required; failed regional qualification is retained')
    if not a.software_smoke and (summary['duration_s']<60 or summary['long_duration_s']<300):
        raise ValueError('Full-duration source validation required')
    rows={r['case']['name']:r['result'] for r in summary['finished']}
    base=rows['hybrid-sham']['hybrid_assessment']['motor_profiles']
    long=rows['hybrid-long']['hybrid_assessment']['motor_profiles']
    common=set(base)&set(long)
    if not common:raise ValueError('No common repeated motor neuron in ordinary/long parent')
    reference=max(sorted(common),key=lambda name:min(base[name]['amplitude_mV'],long[name]['amplitude_mV']))
    sources={name:source/'inputs'/name for name in metadata['sha256']}
    sources.update({'training_parent_configuration.json':source/'hybrid-sham/configuration.json',
                    'training_parent_initial_state.npz':source/'hybrid-sham/initial_state.npz',
                    'reference-training-sham.npz':source/'hybrid-sham/trajectory.npz',
                    'parent_validation_summary.json':source/'summary.json',
                    'parent_input_manifest.json':source/'inputs/manifest.json'})
    hashes={name:file_sha256(path) for name,path in sources.items()}
    out.mkdir(parents=True,exist_ok=False)
    for name,path in sources.items():
        shutil.copy2(path,out/name)
        if file_sha256(out/name)!=hashes[name]:raise ValueError('Copied fit input mismatch')
    if hashes!={name:file_sha256(path) for name,path in sources.items()}:raise ValueError('Source changed during snapshot')
    metadata.update(experiment='E067',protocol='connection-amplitude',candidate_sha256=a.expected_parent,
                    sha256=hashes,sources={name:str(path) for name,path in sources.items()},
                    reference_motor_neuron=reference,software_smoke=a.software_smoke,
                    source_validation=str(source),source_regional_qualification=summary['hybrid_screening']['rhythm_validation_pass'])
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    capture_run_record(out,repo,a)
    print(json.dumps(dict(parent_sha256=a.expected_parent,reference_motor_neuron=reference,
                           source_unchanged=True,fresh_stage_not_optimizer_resume=True)),flush=True)


if __name__=='__main__':main()
