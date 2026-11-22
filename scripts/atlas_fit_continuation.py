"""Pin qualified atlas-fit parents; preserve absolute coordinate provenance."""
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.experiment_records import file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_policy import Coordinates
from scripts.stable_atlas_worker import atomic_json as write


def qualified_parent(run):
    run = Path(run).resolve()
    docs = {name:json.loads((run/(name+'.json')).read_text()) for name in ('manifest','summary','audit','budget_receipt','selection')}
    m,s,a,b,selection = [docs[k] for k in ('manifest','summary','audit','budget_receipt','selection')]
    if m.get('experiment') not in ('E115','E116') or s.get('experiment') != m['experiment']:
        raise ValueError('Qualified own-state fitting parent required')
    if not all(b.get(k) is True for k in ('all_steps_finished','validation_complete','audit_complete','fit_qualified')) or b.get('timed_out'):
        raise ValueError('Parent budget/qualification incomplete')
    if not all(a.get(k) is True for k in ('audit_complete','execution_complete','integrity_passed')) or a.get('problems') or a.get('unavailable_evidence'):
        raise ValueError('Complete clean parent audit required')
    if not all(s.get(k) is True for k in ('validation_complete','autonomy_passed','selection_reproduced','fit_qualified_before_audit')):
        raise ValueError('Parent scientific gates incomplete')
    if not s.get('parameters',{}).get('passed') or s['parameters'].get('changed_coordinates',0) <= 0 or not s.get('final_metrics',{}).get('passed'):
        raise ValueError('Parent parameter/final gate failed')
    model = Path(selection['selected']).resolve();sha = file_sha256(model)
    if sha != selection['sha256'] or sha != s['selected_sha256']:raise ValueError('Parent selected checkpoint identity differs')
    return docs


def latest_parent(base):
    candidates = sorted(Path(base).glob('measured-atlas-fit-*'))
    candidates = [p for p in candidates if p.is_dir()]
    if not candidates:raise ValueError('No E115 run found; refusing fallback')
    # Never silently skip a newer incomplete/failed run in favor of an older one.
    run = candidates[-1];qualified_parent(run);return run.resolve()


def prepare_continuation(source, root):
    source = Path(source).resolve();docs = qualified_parent(source)
    old = docs['manifest'];summary = docs['summary'];evidence = {}
    def pin(path, expected=None):
        path=Path(path);sha=file_sha256(path)
        if expected is not None and sha != expected:raise ValueError('Parent evidence changed: '+str(path))
        evidence[str(path)] = sha;return path
    for name in docs:pin(source/(name+'.json'))
    for key in ('evidence_sha256',):
        for path, sha in docs['audit'][key].items():pin(path,sha)
        for path, sha in old[key].items():pin(path,sha)
    selection=docs['selection'];model=pin(selection['selected'],summary['selected_sha256'])
    identity=json.loads(pin(model.parent/'identity.json').read_text())
    pin(model,identity['sha256']);coordinates=pin(model.parent/'coordinates.npz',identity['coordinates_sha256'])
    local=saved_arrays(coordinates)['offsets']
    origin=pin(old.get('absolute_origin',old['parent']),old.get('absolute_origin_sha256',old['parent_sha256']))
    previous=np.zeros_like(local)
    if 'parent_absolute_offsets' in old:
        previous=saved_arrays(pin(old['parent_absolute_offsets'],old['parent_absolute_offsets_sha256']))['offsets']
    cumulative=previous+local
    if not np.isfinite(cumulative).all() or np.max(abs(cumulative)) > .1+1e-12:raise ValueError('Parent exceeds cumulative bounds')
    # Local model identity/projection remains exactly reconstructible against its immediate parent.
    if not Coordinates(saved_arrays(old['parent'])).audit(saved_arrays(model),local)['passed']:
        raise ValueError('Parent coordinate reconstruction differs')
    inputs=root/'inputs';inputs.mkdir()
    def copy(path,name):
        src=pin(path);dest=inputs/name;shutil.copy2(src,dest);pin(dest,evidence[str(src)]);return str(dest)
    parent=copy(model,'parent.npz');origin_copy=copy(origin,'absolute-origin.npz')
    auto=json.loads(pin(source/'autonomy.json').read_text())['selected']
    anchors=copy(pin(auto['anchors'],auto['anchors_sha256']),'anchors.npz')
    if str(saved_arrays(anchors)['model_sha256']) != summary['selected_sha256']:raise ValueError('Parent anchors belong to other model')
    stats=copy(old['statistics'],'statistics.json')
    matrices=saved_arrays(pin(source/'final-matrices.npz'))
    baselines=[]
    for level in range(3):
        path=inputs/f'baseline-{level}.npz';np.savez_compressed(path,response=matrices['level_'+str(level)])
        pin(path);baselines.append(str(path))
    references={}
    for path in sorted((source/'tasks').glob('final-*/result.json')):
        row=json.loads(pin(path).read_text());spec=row['task']
        pin(row['artifact'],row['sha256'])
        if spec['model_sha256'] != summary['selected_sha256'] or spec['anchors_sha256'] != auto['anchors_sha256']:
            raise ValueError('Parent final response identity differs')
        references[str(spec['step_ms'])]=row['rhythms']
    if set(references) != set(old['rhythm_references']):raise ValueError('Parent final grids incomplete')
    cumulative_path=inputs/'parent-absolute-offsets.npz';np.savez_compressed(cumulative_path,offsets=cumulative);pin(cumulative_path)
    result=dict(old, experiment='E116',parent=parent,parent_sha256=summary['selected_sha256'],
        anchors=anchors,anchors_sha256=auto['anchors_sha256'],statistics=stats,baselines=baselines,
        source=str(source),evidence_sha256=evidence,rhythm_references=references,
        absolute_origin=origin_copy,absolute_origin_sha256=file_sha256(origin_copy),
        parent_absolute_offsets=str(cumulative_path),parent_absolute_offsets_sha256=file_sha256(cumulative_path),
        continuation=dict(parent_run=str(source),parent_sha256=summary['selected_sha256'],
            optimizer='Fresh Adam/SPSA state; not exact optimizer/worker resume',seed=1160),seed=1160)
    write(root/'manifest.json',result);return result
