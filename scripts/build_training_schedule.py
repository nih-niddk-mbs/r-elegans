"""Generate experiments/TRAINING_SCHEDULE.md from the experiment plan and outcome records.

Only recorded fields are used; missing fields are shown as '-' rather than inferred. Experiments without a plan record are
covered by LAB_NOTEBOOK.md and TRAINING_PROTOCOLS.md. Regenerate after adding a record: python -m scripts.build_training_schedule
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / 'experiments/records'


def number(path):
    match = re.match(r'E(\d+)', path.name)
    return int(match.group(1)) if match else 10 ** 9


def short(value, limit):
    text = ' '.join(str(value).split()) if value not in (None, '') else '-'
    return text if len(text) <= limit else text[:limit - 1] + '...'


def budget_text(plan):
    budget = plan.get('budget')
    if isinstance(budget, dict):
        keys = ('total_minutes', 'total_minutes_default', 'workers', 'population', 'generations')
        parts = [f'{k}={budget[k]}' for k in keys if k in budget]
        return ', '.join(parts) or short(budget, 90)
    return short(budget, 90)


def outcome(experiment):
    files = sorted(RECORDS.glob(f'{experiment}-server-summary.json')) + sorted(RECORDS.glob(f'{experiment}.json'))
    for path in files:
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        for key in ('outcome', 'status', 'result'):
            if data.get(key):
                return f'{short(data[key], 110)} ([{path.name}](records/{path.name}))'
        return f'[{path.name}](records/{path.name})'
    return '-'


def main():
    plans = sorted(RECORDS.glob('E*-plan.json'), key=number)
    lines = ['# Training and experiment schedule', '',
             'Generated from `experiments/records/E*-plan.json` and outcome records by `python -m scripts.build_training_schedule`.',
             'Each row is what was planned (command, budget, parent, trainable fields) and, where recorded, the outcome. A plan is not evidence that',
             'the run happened; outcomes link to the user-reported summaries. Narrative, audits and negative results are in `LAB_NOTEBOOK.md`;',
             'protocol design is in `TRAINING_PROTOCOLS.md`; curricula and policies are the JSON files in `experiments/`.', '',
             f'{len(plans)} experiments have a plan record. Earlier work (before the plan-record convention) is documented only in the notebook.', '']
    for path in plans:
        plan = json.loads(path.read_text())
        experiment = plan.get('experiment', path.name.split('-')[0])
        lines += [f'## {experiment} ({plan.get("record_date", "-")})', '',
                  f'- Question: {short(plan.get("question"), 400)}',
                  f'- Method: {short(plan.get("method") or plan.get("initialization"), 500)}',
                  f'- Parent: {short(plan.get("parent_checkpoint_sha256") or plan.get("parent") or plan.get("source_run"), 220)}',
                  f'- Trainable: {short(plan.get("trainable"), 300)}',
                  f'- Fixed: {short(plan.get("fixed"), 300)}',
                  f'- Command: `{short(plan.get("command"), 200)}`',
                  f'- Report: `{short(plan.get("report"), 160)}`',
                  f'- Budget: {budget_text(plan)}',
                  f'- Acceptance: {short(plan.get("acceptance_criteria") or plan.get("acceptance"), 500)}',
                  f'- Outcome record: {outcome(experiment)}', '']
    out = REPO / 'experiments/TRAINING_SCHEDULE.md'
    out.write_text('\n'.join(lines) + '\n')
    print('wrote', out, len(plans), 'experiments')


if __name__ == '__main__':
    main()
