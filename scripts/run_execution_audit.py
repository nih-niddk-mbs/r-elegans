"""E121 requirements/code/data/checkpoint audit; no fitting or forward simulation."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.stable_atlas_worker import atomic_json as write

REPO=Path(__file__).resolve().parents[1]
BEST='/home/carsonc/r-elegans-results/reverse-candidate-20261123T173934Z-963315'
EXPECTED_SHA='ac621eaa129c1c533691f52cce802d6ff259ada754d159bf287a30895c48adf3'


def local_readiness(repo=REPO):
    repo=Path(repo);names=['r_elegans/brain/common_state_replay.py','r_elegans/brain/extrasynaptic.py',
        'scripts/stable_atlas_worker.py','scripts/stable_atlas_policy.py','scripts/run_saved_atlas_candidate.py',
        'scripts/run_selected_atlas_audit.py','scripts/run_atlas_baseline.py','r_elegans/data/randi_atlas.py',
        'r_elegans/atlas_evaluation.py','r_elegans/body/channel_bridge.py','r_elegans/body/embodied_sensory.py',
        'experiments/physiology_initialization_strategy_v2.json']
    hashes={n:file_sha256(repo/n) for n in names};worker=(repo/names[2]).read_text();tree=ast.parse(worker)
    fields=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIELDS' for t in node.targets))
    active=(repo/names[0]).read_text();called={n.func.id for n in ast.walk(ast.parse(active)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    # Static facts alone do not certify runtime artifacts; full saved-data recheck is separate.
    flags=dict(active_anchor_fields=list(fields),active_extrasynaptic_state='peptide' in fields,
        active_replay_calls_peptide_step='peptide_level_step' in called,
        optional_extrasynaptic_module_present=(repo/'r_elegans/brain/extrasynaptic.py').is_file(),
        active_replay_simulates_body=False,active_replay_objective='Pooled stimulus-response correlation; no full time-series/amplitude fit')
    findings=[
      dict(id='GAP-CV',status='NOT_IMPLEMENTED_IN_ACTIVE_FITTER',severity='blocking_requested_workflow',
           finding='Five independent animal-fold model fits, fold-aware trainer and all-animal calibration are plans. E121 draft identities are not fitted CV.',evidence=['experiments/physiology_initialization_strategy_v2.json','scripts/run_saved_atlas_candidate.py']),
      dict(id='GAP-EXTRA',status='OPTIONAL_LEGACY_ONLY',severity='blocking_full_model_claim',
           finding='Peptide module/assets exist but current four-field common-state runner does not propagate peptide state or apply extrasynaptic input.',evidence=names[:3]),
      dict(id='GAP-OPT',status='DIAGNOSTIC_NOT_GENERAL_FITTER',severity='blocking_further_blind_updates',
           finding='E119 tested one saved direction; a reusable low-dimensional checked-descent/grouped fitter is not implemented. E118 raw direction was measured uphill.',evidence=['scripts/run_atlas_direction_check.py','scripts/run_panel_atlas_fit.py','experiments/records/E119-server-summary.json']),
      dict(id='SCOPE-OBS',status='LIMITED_OBJECTIVE',severity='claim_limit',finding='Current correlation loss cannot identify calibrated amplitudes or complete kinetics; mean calcium summaries are fitted, not complete neural time courses.',evidence=['scripts/stable_atlas_policy.py','r_elegans/brain/common_state_replay.py']),
      dict(id='RISK-ID',status='SERVER_DATA_CHECK_REQUIRED',severity='data_integrity_risk',finding='Missing animal IDs fall back to filenames; actual metadata must be inventoried before animal-grouped CV.',evidence=['r_elegans/data/randi_atlas.py']),
      dict(id='RISK-GENO',status='SERVER_DATA_CHECK_REQUIRED',severity='model_assumption_risk',finding='Current build_statistics does not filter by genotype; NWB file count is called animals. Genotype mixture and unique worm counts need actual data audit.',evidence=['scripts/run_atlas_baseline.py']),
      dict(id='SCOPE-BODY',status='NOT_TESTED_ON_E120',severity='claim_limit',finding='Sensory/body/physical environment modules exist, but E120 is an atlas-only neural evaluation; no E120 body, swimming or sensory start/stop qualification.',evidence=['scripts/run_saved_atlas_candidate.py','r_elegans/body/channel_bridge.py','r_elegans/body/embodied_sensory.py']),
      dict(id='SCOPE-NUMERIC',status='REPORTED_PASS_RECHECK_REQUIRED',severity='evidence_limit',finding='E120 reports full three-level atlas checks and independent state/panel repeat; full atlas loss was not independently repeated. Historical E104/E105 failures preserved.',evidence=['experiments/records/E120-server-summary.json']),
      dict(id='GAP-CELLPRIORS',status='SHARED_TEMPLATE_PRIORS',severity='blocking_measured_cell_model_claim',finding='Historical polarized baseline used296 transferred AWC channel templates plus6 RMD profiles. Per-cell fitting can change scales; this is not302 independently measured neuron models. Server snapshot checks E120 base/effective conductances and template match.',evidence=['experiments/records/model_data_audit_mac_rhythm_v1_polarized_summary.json','r_elegans/brain/conductance_network.py']),
      dict(id='SCOPE-PARAM',status='STRUCTURED_PRIORS_NOT_UNIQUE_RECOVERY',severity='claim_limit',finding='9538 fitted scales/strengths; stereotypical fields, LEAK/RMD and anatomy fixed. Some are engineering priors, not independently measured values.',evidence=['scripts/stable_atlas_policy.py']),
      dict(id='GAP-CLEAN-START',status='UNVERIFIED_FOR_NEW_FOLDS',severity='blocking_independent_cv_claim',finding='E120 has atlas exposure. A common verified pre-atlas-fit ancestor is required for independent fold fits; no new CV ancestor lineage audit/fold runs evidenced.',evidence=['experiments/physiology_initialization_strategy_v2.json']),
    ]
    return dict(audit='E121-local-readiness',audit_complete=True,inspection='Static source and committed records, not server execution',
        flags=flags,findings=findings,source_sha256=hashes,ready_for_requested_training=False,
        conclusion='Concrete implementation/execution gaps remain; more success flags on existing network-only runs do not close them.')


def reconcile_readiness(readiness,components):
    """Resolve observed risks from completed components; preserve original evidence."""
    from copy import deepcopy
    result=deepcopy(readiness);data=components.get('dataset',{});checkpoint=components.get('checkpoint',{})
    def observed_count(full,compact):
        value=data.get(full)
        return len(value) if value is not None else data.get(compact)
    for finding in result['findings']:
        resolved=None
        if finding['id']=='RISK-ID' and data.get('integrity_passed') and observed_count('files_with_unverified_identity','unverified_identity_files')==0 and observed_count('repeated_subject_files','repeated_subjects')==0:
            resolved=('VERIFIED_METADATA_NO_ID_CONFLICTS','Raw audit confirms distinct metadata IDs with no missing IDs/repeated subjects in this dataset. Filename fallback remains a structural risk for future data.')
        elif finding['id']=='RISK-GENO' and data.get('integrity_passed') and data.get('genotype_file_counts')=={'WT':data.get('raw_nwb_files')} and observed_count('subjects_with_conflicting_genotypes','conflicting_genotype_subjects')==0:
            resolved=('VERIFIED_METADATA_ALL_WT','All audited file genotype labels are WT with no subject conflicts. Source code still lacks explicit genotype selection for future data.')
        elif finding['id']=='SCOPE-NUMERIC' and checkpoint.get('reported_qualification_confirmed'):
            resolved=('SAVED_DATA_RECHECK_PASSED','E120 stored numerical/parameter/control/rhythm/state/panel qualification independently recomputed, source records unchanged. No new forward/full-loss replication.')
        elif finding['id']=='GAP-CELLPRIORS' and checkpoint.get('parameter_scope',{}).get('base_matches_bundled_polarized_reference'):
            scope=checkpoint['parameter_scope']
            resolved=('CONFIRMED_BASE_TEMPLATE_PROVENANCE',f"Base template inheritance confirmed; {scope['unique_base_conductance_rows']} distinct base rows and {scope['unique_effective_conductance_rows']} effective rows. Per-cell variation does not imply independently measured physiology.")
        if resolved:
            finding.setdefault('initial_status',finding['status']);finding.setdefault('initial_finding',finding['finding'])
            finding['status'],finding['finding']=resolved;finding['component_evidence_applied']=True
    result['ready_for_requested_training']=False
    result['inspection']='Static source review reconciled with available completed server components; implementation gaps remain.'
    return result


def checkpoint_scope(source):
    import numpy as np
    from scripts.selected_atlas_audit_checks import saved_arrays
    m=json.loads((source/'manifest.json').read_text());a=saved_arrays(m['candidate'])
    ids=a['neuron_ids'].tolist();base=a['cells_conductances_nS'];effective=base*np.exp(a['params_log_channel_scale'])
    reference=REPO/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz';r=saved_arrays(reference)
    matches=bool(np.array_equal(base,r['cells_conductances_nS']))
    provenance=REPO/'experiments/records/model_data_audit_mac_rhythm_v1_polarized_summary.json'
    historical=json.loads(provenance.read_text())['s3_cell_provenance']
    return dict(neurons=len(ids),channel_templates=base.shape[1],unique_base_conductance_rows=len(np.unique(base,axis=0)),
        unique_effective_conductance_rows=len(np.unique(effective,axis=0)),
        base_matches_bundled_polarized_reference=matches,
        historical_base_provenance_if_matched=historical['cell_kinds'] if matches else 'Not transferred; base differs',
        positive_base_channel_entries=int((base>0).sum()),
        chemical_edges=int((a['fixed_chemical_mask']>0).sum()),gap_directed_entries=int((a['fixed_gap_mask']>0).sum()),
        topology_matches_bundled_reference=bool(np.array_equal(a['fixed_chemical_mask'],r['fixed_chemical_mask']) and np.array_equal(a['fixed_gap_mask'],r['fixed_gap_mask'])),all_effective_conductances_finite_nonnegative=bool(np.isfinite(effective).all() and (effective>=0).all()),
        capacitance_pF_range=[float(a['cells_capacitance_pF'].min()),float(a['cells_capacitance_pF'].max())],
        checkpoint_contains_peptide_fields=any('peptide' in k or 'extrasynaptic' in k for k in a),
        checkpoint_sha256=file_sha256(m['candidate']),reference_sha256=file_sha256(reference),provenance_sha256=file_sha256(provenance),
        scope='Static parameter/provenance snapshot; varying effective conductances are not independently validated neuron-specific physiology.')


def worker(root):
    args=json.loads((root/'job.json').read_text());readiness=local_readiness();write(root/'readiness.json',readiness)
    components={};problems=[]
    # Do independent components even if another is unavailable or fails.
    try:
        from scripts.run_saved_atlas_candidate import audit
        source=Path(args['best_run']);before={name:file_sha256(source/name) for name in ('manifest.json','summary.json','audit.json','budget_receipt.json','selection.json')}
        source_summary=json.loads((source/'summary.json').read_text());source_budget=json.loads((source/'budget_receipt.json').read_text())
        result=audit(source,output_dir=root/'checkpoint-recheck')
        if before!={name:file_sha256(source/name) for name in before}:raise ValueError('Source run records changed during recheck')
        identities=source_summary.get('candidate_sha256')==EXPECTED_SHA and source_summary.get('selected_sha256')==EXPECTED_SHA
        completed=source_budget.get('all_steps_finished') and source_budget.get('candidate_qualified') and not source_budget.get('timed_out')
        scope=checkpoint_scope(source);write(root/'checkpoint-scope.json',scope)
        components['checkpoint']=dict(parameter_scope=scope,saved_data_recheck_complete=result['execution_complete'],integrity_passed=result['integrity_passed'],
            reported_qualification_confirmed=bool(identities and completed and source_summary.get('candidate_qualified_before_audit') and result['execution_complete'] and result['integrity_passed']),
            expected_sha256=EXPECTED_SHA,source_records_unchanged=True,problems=result['problems'],unavailable_evidence=result['unavailable_evidence'],
            report=str(root/'checkpoint-recheck/audit.json'),forward_simulation_performed=False)
    except Exception as exc:
        traceback.print_exc();components['checkpoint']=dict(saved_data_recheck_complete=False,error=repr(exc));problems.append('checkpoint: '+repr(exc))
    write(root/'progress.json',dict(completed_components=list(components)))
    try:
        from scripts.atlas_dataset_inventory import inventory
        best_manifest=json.loads((Path(args['best_run'])/'manifest.json').read_text())
        result=inventory(Path(args['data_root']),root/'dataset',expected_statistics=best_manifest['statistics'])
        components['dataset']={k:result[k] for k in ('inventory_complete','integrity_passed','raw_nwb_files','raw_verified_unique_subjects',
            'legacy_usable_subjects','verified_usable_subjects','successful_trials','files_with_no_accepted_trials','files_with_unverified_identity',
            'repeated_subject_files','genotype_file_counts','subjects_with_conflicting_genotypes','legacy_split_counts','draft_fold_count','training_ready','cached_statistics_comparison','errors')}
    except Exception as exc:
        traceback.print_exc();components['dataset']=dict(inventory_complete=False,error=repr(exc));problems.append('dataset: '+repr(exc))
    readiness=reconcile_readiness(readiness,components)
    result=dict(experiment='E121',audit_complete=True,components=components,problems=problems,
        evidence_integrity_passed=bool(components.get('checkpoint',{}).get('reported_qualification_confirmed') and components.get('dataset',{}).get('integrity_passed')),
        readiness=readiness,training_launched=False,forward_simulation_performed=False,ready_for_requested_training=False,
        limits=['A complete audit can report unimplemented/blocked capabilities; it is not training qualification.',
               'Draft fold identities do not mean cross-validation model fits exist.',
               'Source/run records preserved; independent saved-data recheck is not independent forward replication.'])
    write(root/'summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--best-run',type=Path,default=Path(BEST));p.add_argument('--data-root',type=Path,default=Path('/home/carsonc/r-elegans-data'))
    p.add_argument('--worker',action='store_true');p.add_argument('--local-only',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.output_dir.resolve()
    if a.report:
        print('Audit:',root)
        for name in ('summary','budget_receipt'):
            path=root/(name+'.json');s=json.loads(path.read_text()) if path.exists() else {}
            if name=='summary':
                print('audit_complete:',s.get('audit_complete'));print('evidence_integrity_passed:',s.get('evidence_integrity_passed'))
                for label,c in s.get('components',{}).items():
                    if label=='dataset':
                        keys=('inventory_complete','integrity_passed','raw_nwb_files','raw_verified_unique_subjects','legacy_usable_subjects','verified_usable_subjects','successful_trials','files_with_no_accepted_trials','genotype_file_counts','legacy_split_counts','draft_fold_count','training_ready','cached_statistics_comparison')
                        brief={k:c.get(k) for k in keys};brief.update(unverified_identity_files=len(c.get('files_with_unverified_identity',[])),repeated_subjects=len(c.get('repeated_subject_files',{})),conflicting_genotype_subjects=len(c.get('subjects_with_conflicting_genotypes',{})),error=c.get('error'),first_errors=c.get('errors',[])[:3])
                    else:brief=c
                    print(label+':',json.dumps(brief))
                for f in reconcile_readiness(s['readiness'],s.get('components',{})).get('findings',[]) if s.get('readiness') else []:print(f['id'],f['status'])
                print('ready_for_requested_training:',s.get('ready_for_requested_training'));print('problems:',s.get('problems'))
            else:print(name+':',json.dumps(s))
        return
    if a.worker:worker(root);return
    if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External audit output required')
    start=time.monotonic()
    root.mkdir(parents=True,exist_ok=False);capture_run_record(root,REPO,vars(a),phase='E121-execution-audit')
    if a.local_only:write(root/'readiness.json',local_readiness());return
    write(root/'job.json',dict(best_run=str(a.best_run.resolve()),data_root=str(a.data_root.resolve())))
    from scripts.total_job_budget import run_group
    code,timeout=run_group([sys.executable,'-u','-m','scripts.run_execution_audit','--output-dir',str(root),'--worker'],max(.01,60*60-30-(time.monotonic()-start)))
    path=root/'summary.json';s=json.loads(path.read_text()) if path.exists() else {}
    if not path.exists():
        partial=root/'progress.json';dp=root/'dataset/progress.json'
        write(path,dict(experiment='E121',audit_complete=False,problems=['Timed out or failed; inspect preserved partial records/log'],
            partial_progress=json.loads(partial.read_text()) if partial.exists() else None,
            partial_dataset_progress=json.loads(dp.read_text()) if dp.exists() else None,
            training_launched=False,ready_for_requested_training=False))
    receipt=dict(total_minutes=60,elapsed_seconds=time.monotonic()-start,exit_code=code,timed_out=timeout,audit_finished=bool(code==0 and not timeout and s.get('audit_complete')),training_launched=False)
    write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True);raise SystemExit(0 if receipt['audit_finished'] else 2)


if __name__=='__main__':main()
