"""Saved audit reporting must expose failures without touching source artifacts."""
import json
import subprocess
import sys
from pathlib import Path

from scripts.report_selected_atlas_audit import print_details


def test_details_exposes_errors_and_reference_without_writing(tmp_path, capsys):
    reference = tmp_path / 'references' / 'parent' / 'summary.json'
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({
        'checkpoint_sha256': 'parent', 'finite': True,
        'cells': {'RMDL': {'passed': True, 'median_period_s': 0.5}},
    }))
    checks = {
        'parameters': {'passed': False, 'errors': ['Frozen field changed: example'],
                       'recorded_masks_match': False, 'bounds': {'example': {'at_upper': 2}}},
        'rhythm': {'reference_checkpoint_sha256': 'parent', 'cells': {
            'RMDL': {'coarse': {'passed': False, 'gates': {'multiple_cycles': False}},
                     'fine': {'passed': False, 'reason': 'nonfinite_or_short_trace'},
                     'relative_reference_criteria_passed': False,
                     'coarse_vs_fine': {'passed': False, 'reason': 'Persistence failed'}}}},
    }
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    print_details(tmp_path, checks)
    output = capsys.readouterr().out
    for evidence in ('Frozen field changed: example', 'recorded_masks_match',
                     'multiple_cycles', 'nonfinite_or_short_trace', 'Persistence failed',
                     'median_period_s', 'at_upper', 'unavailable'):
        assert evidence in output
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_details_missing_or_mismatched_reference_is_explicit(tmp_path, capsys):
    print_details(tmp_path, {})
    assert 'Rhythm reference unavailable:' in capsys.readouterr().out
    reference = tmp_path / 'references' / 'expected' / 'summary.json'
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({'checkpoint_sha256': 'wrong', 'cells': {'RMDL': {'secret_marker': True}}}))
    print_details(tmp_path, {'rhythm': {'reference_checkpoint_sha256': 'expected'}})
    output = capsys.readouterr().out
    assert "'checkpoint_matches': False" in output
    assert 'secret_marker' not in output
    assert 'Parent RMDL: "unavailable"' in output


def test_report_cli_details_reads_saved_cohort(tmp_path):
    (tmp_path / 'summary.json').write_text(json.dumps({
        'audit_complete': True, 'preparation_failures': [], 'finished': [{
            'label': 'E104-test', 'result': {'audit_complete': True, 'checks': {
                'parameters': {'passed': False, 'errors': ['Per-cell bound exceeded: example']}}}}]}))
    result = subprocess.run([sys.executable, '-m', 'scripts.report_selected_atlas_audit',
                             '--run', str(tmp_path), '--details'],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert 'Per-cell bound exceeded: example' in result.stdout
    assert 'No model promoted' in result.stdout
