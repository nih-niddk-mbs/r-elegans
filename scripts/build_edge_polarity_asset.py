"""Build r_elegans/assets/edge_polarity_v1.{npz,json} from the Fenyves et al. 2020 supplementary tables.

Needs the raw tables under $R_ELEGANS_DATA_DIR/raw/neurotransmitters/synaptic_polarity_2020 (see REPRODUCE.md).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from r_elegans.assets import load_asset_document
from r_elegans.data.connectome import load_connectome
from r_elegans.data.synapse_polarity import edge_polarity

ASSETS = Path(__file__).resolve().parents[1] / 'r_elegans/assets'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root / 'raw/neurotransmitters/synaptic_polarity_2020'
    ids = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    chemical = np.asarray(load_connectome().chemical_counts)
    sign, source, conflict, unknown = edge_polarity(root, ids, chemical)
    edges = chemical > 0
    np.savez_compressed(ASSETS / 'edge_polarity_v1.npz', sign=sign, source=source, conflict=conflict, neuron_ids=np.array(ids))
    metadata = dict(
        schema_version=1,
        source='Fenyves et al. 2020, Synaptic polarity and sign-balance prediction using gene expression data in the C. elegans chemical synapse connectome network, PLOS Comput Biol',
        doi='10.1371/journal.pcbi.1007974', license_note='Creative Commons Attribution; retain source attribution. Derived signs only; source tables are not redistributed.',
        precedence='NT+receptor prediction (SI s003, sheet 5. Sign prediction, column Predicted polarity), then transmitter-only (SI s004, sheet Connectome)',
        source_codes={'1': 'NT+receptor', '2': 'transmitter-only', '0': 'undetermined (complex, no prediction, or absent from the tables)'},
        source_sha256={n: sha256(root / n) for n in ('pcbi.1007974.s003.xlsx', 'pcbi.1007974.s004.xlsx')},
        runtime_model_sha256=sha256(ASSETS / 'runtime_model_v1.json'),
        model_chemical_edges=int(edges.sum()), edges_nt_receptor=int((edges & (source == 1)).sum()), edges_nt_only=int((edges & (source == 2)).sum()),
        edges_undetermined=int((edges & (sign == 0)).sum()), excitatory=int((edges & (sign > 0)).sum()), inhibitory=int((edges & (sign < 0)).sum()),
        method_sign_conflicts=int((edges & conflict).sum()), table_names_unknown_to_model=unknown,
        reversal_priors_mV=dict(excitatory=0.0, inhibitory=-85.0, neutral_if_requested=-40.0, status='project priors, not verified literature values'))
    (ASSETS / 'edge_polarity_v1.json').write_text(json.dumps(metadata, indent=1) + '\n')
    print(json.dumps(metadata, indent=1))


if __name__ == '__main__':
    main()
