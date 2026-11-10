"""E047: source-backed neuron groups and existing sensory-input pathway audit."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from r_elegans.assets import load_asset_document
from r_elegans.neuron_roles import ROLE_GROUPS, annotate_neuron_roles
from scripts.experiment_records import capture_run_record, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if out == repo or repo in out.parents:
        parser.error('Keep detailed scientific audit outputs outside Git')
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, args)
    model = load_asset_document('runtime_model_v1.json')
    rows = annotate_neuron_roles(model['neuron_ids'])
    nmj = Counter(model['nmj']['neuron_index'])
    for row in rows:
        row['anatomical_nmj_edges'] = nmj[row['index']]
    role_counts = Counter(role for row in rows for role in row['roles'])
    groups = [dict(**asdict(group), member_indices=[model['neuron_ids'].index(n) for n in group.neurons])
              for group in ROLE_GROUPS]
    report = dict(experiment='E047', status='completed', neurons=len(rows),
        curated_neurons=sum(r['annotation_status'] == 'curated_subset' for r in rows),
        unreviewed_neurons=sum(r['annotation_status'] == 'unreviewed' for r in rows),
        role_counts=dict(role_counts), groups=groups, neuron_annotations=rows,
        branches={
            'autonomous_channel_body':dict(sensory_cells_present=True, environmental_input=False,
                body_to_brain_proprioception=False, neuromodulatory_state_mechanisms=False,
                code=['r_elegans/body/channel_bridge.py', 'r_elegans/brain/conductance_network.py']),
            'normalized_full_network_actor':dict(chemical_input_targets=['AWCL','AWCR','ASEL','ASER'],
                inputs=['adapted log concentration','recent derivative','relative concentration'],
                biological_receptor_fit=False, physical_proprioception=False,
                code=['r_elegans/rl/full_network_actor.py','r_elegans/envs/full_network_petri_dish.py']),
            'legacy_14_cell_subcircuit':dict(chemical_input_targets=['AWCL','AWCR','ASEL','ASER'],
                engineered_phase_inputs=True, physical_proprioception=False,
                code=['r_elegans/brain/sensory.py','r_elegans/brain/circuit.py'])},
        limitations=['Curated functional subset, not a complete sensory/motor/interneuron atlas.',
            'Roles overlap; counts cannot be summed into a partition of 302.',
            'Biological roles and anatomical NMJs do not prove fitted-network causal function.',
            'No dynamics, parameters, trainable masks or sensory currents changed.'])
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    manifest = {str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('experiment','neurons','curated_neurons','unreviewed_neurons','role_counts')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
