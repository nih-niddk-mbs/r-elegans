"""Per-edge synaptic polarity from Fenyves et al. 2020 (PLOS Comput Biol, 10.1371/journal.pcbi.1007974).

Sources, in precedence order: the NT+receptor prediction (SI s003, sheet '5. Sign prediction') and the
transmitter-only prediction (SI s004, sheet 'Connectome'). An edge gets a sign only where a method gives '+' or '-';
'complex', 'no pred' and edges absent from both tables stay undetermined (0). Their connectome version differs
slightly from the Cook 2019 matrix in the model, so coverage is reported against the model's own edges.
"""
from pathlib import Path

import numpy as np

SIGN = {'+': 1, '-': -1}
SOURCE_NT_R, SOURCE_NT_ONLY, SOURCE_NONE = 1, 2, 0


def _sheet_rows(path, sheet, header=None):
    from openpyxl import load_workbook
    worksheet = load_workbook(path, read_only=True, data_only=True)[sheet]
    return list(worksheet.iter_rows(values_only=True))


def _normalize(value):
    text = str(value).strip().lower()
    return SIGN.get(text, 0)


def parse_polarity_tables(root):
    root = Path(root)
    nt_r = {}
    for row in _sheet_rows(root / 'pcbi.1007974.s003.xlsx', '5. Sign prediction')[2:]:
        if row[0] and row[3] and row[5] == 'chemical':
            nt_r[(str(row[0]).strip(), str(row[3]).strip())] = _normalize(row[16])
    nt_only = {}
    for row in _sheet_rows(root / 'pcbi.1007974.s004.xlsx', 'Connectome')[1:]:
        if row[0] and row[2] and row[4] == 'chemical':
            nt_only[(str(row[0]).strip(), str(row[2]).strip())] = _normalize(row[5])
    return nt_r, nt_only


def edge_polarity(root, neuron_ids, chemical_counts):
    """Arrays [post, pre]: sign in {-1, 0, +1}, source code, and a conflict flag, restricted to model edges."""
    nt_r, nt_only = parse_polarity_tables(root)
    index = {n: i for i, n in enumerate(neuron_ids)}
    sign = np.zeros((len(neuron_ids),) * 2, np.int8)
    source = np.zeros_like(sign)
    conflict = np.zeros(sign.shape, bool)
    for post, pre in zip(*np.nonzero(np.asarray(chemical_counts))):
        key = (neuron_ids[pre], neuron_ids[post])
        a, b = nt_r.get(key, 0), nt_only.get(key, 0)
        if a:
            sign[post, pre], source[post, pre] = a, SOURCE_NT_R
            conflict[post, pre] = bool(b and b != a)
        elif b:
            sign[post, pre], source[post, pre] = b, SOURCE_NT_ONLY
    unknown_names = sorted({n for pair in list(nt_r) + list(nt_only) for n in pair if n not in index})
    return sign, source, conflict, unknown_names
