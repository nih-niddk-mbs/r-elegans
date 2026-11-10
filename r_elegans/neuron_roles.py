"""Source-backed, overlapping functional annotations for a locomotion subset.

This is not an exhaustive cell-type atlas or a functional validation of fitted
neurons. Unannotated neurons remain in the full simulation. Roles do not change
weights, input currents, trainable masks or connectivity.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class NeuronRoleGroup:
    key: str
    neurons: tuple[str, ...]
    roles: tuple[str, ...]
    description: str
    sources: tuple[str, ...]


def _pairs(*classes):
    return tuple(name+side for name in classes for side in ('L', 'R'))


def _numbered(name, count):
    return tuple(f'{name}{i:02d}' for i in range(1, count+1))


ANATOMY = 'https://pubmed.ncbi.nlm.nih.gov/22462104/'
TOUCH = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6565008/'
FORWARD = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5948959/'
PROPRIOCEPTION = 'https://doi.org/10.1016/j.neuron.2012.08.039'

ROLE_GROUPS = (
    NeuronRoleGroup('olfactory_attractant', _pairs('AWA', 'AWC'), ('sensory', 'chemosensory'),
        'Odor detection; attraction in the cited odor assays, with context-dependent responses.',
        ('https://pubmed.ncbi.nlm.nih.gov/8348618/',)),
    NeuronRoleGroup('olfactory_awb', _pairs('AWB'), ('sensory', 'chemosensory'),
        'Odor detection and avoidance in the cited assays.',
        ('https://pmc.ncbi.nlm.nih.gov/articles/PMC316880/',)),
    NeuronRoleGroup('soluble_chemical', _pairs('ASE'), ('sensory', 'chemosensory'),
        'Salt and other water-soluble chemical sensing; not interchangeable with odor transduction.',
        ('https://pubmed.ncbi.nlm.nih.gov/1660283/',)),
    NeuronRoleGroup('touch_receptors', _pairs('ALM', 'PLM')+('AVM', 'PVM'), ('sensory', 'mechanosensory'),
        'Six touch-receptor cells; functions and effects need not be identical.',
        ('https://doi.org/10.1016/0012-1606(81)90459-0', TOUCH)),
    NeuronRoleGroup('forward_premotor', _pairs('AVB', 'PVC'), ('interneuron', 'premotor'),
        'Forward-associated descending control; AVB electrical coupling can recruit B-type oscillations.',
        (TOUCH, FORWARD)),
    NeuronRoleGroup('reverse_premotor', _pairs('AVA', 'AVD', 'AVE'), ('interneuron', 'premotor'),
        'Reverse-associated descending control; not a standalone binary reverse switch.',
        (TOUCH, 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5780044/')),
    NeuronRoleGroup('sensory_integration', _pairs('AIY', 'AIB', 'AIZ', 'RIA'), ('interneuron',),
        'Selected olfactory/sensorimotor integration interneurons; not all decision-making cells.',
        ('https://pmc.ncbi.nlm.nih.gov/articles/PMC3038580/',)),
    NeuronRoleGroup('stop_candidate', ('RIS',), ('interneuron', 'state_control'),
        'Experimentally implicated in locomotor stopping; GABA/FLP-11 mechanisms not implemented here.',
        ('https://pmc.ncbi.nlm.nih.gov/articles/PMC6736843/',)),
    NeuronRoleGroup('body_a_motor', _numbered('DA', 9)+_numbered('VA', 12), ('motor',),
        'A-type body motor neurons, associated with backward locomotion.',
        (ANATOMY, 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5780044/')),
    NeuronRoleGroup('body_b_motor', _numbered('DB', 7)+_numbered('VB', 11), ('motor',),
        'B-type body motor neurons, associated with forward locomotion.', (ANATOMY, FORWARD)),
    NeuronRoleGroup('body_d_motor', _numbered('DD', 6)+_numbered('VD', 13), ('motor',),
        'D-type inhibitory body motor neurons contributing to dorsoventral coordination.', (ANATOMY,)),
    NeuronRoleGroup('body_as_motor', _numbered('AS', 11), ('motor',),
        'AS body motor neurons contribute to asymmetric muscle activation and coordination.',
        (ANATOMY, 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6173582/')),
    NeuronRoleGroup('selected_head_motor',
        ('RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR'),
        ('motor',), 'Selected RMD/SMD head motor neurons; not an exhaustive head-motor inventory.', (ANATOMY,)),
    NeuronRoleGroup('b_motor_proprioception', _numbered('DB', 7)+_numbered('VB', 11), ('proprioceptive',),
        'B-type motor neurons also transduce body-bending feedback during forward locomotion.', (PROPRIOCEPTION,)),
    NeuronRoleGroup('dva_proprioception', ('DVA',), ('interneuron', 'proprioceptive'),
        'Stretch-sensitive DVA; TRP-4-dependent proprioceptive regulation of bending.',
        ('https://doi.org/10.1038/nature04538',)),
    NeuronRoleGroup('smdd_proprioception', ('SMDDL', 'SMDDR'), ('proprioceptive',),
        'SMDD motor/sensory role in steering; TRPC-dependent feedback in the cited experiments.',
        ('https://doi.org/10.1371/journal.pbio.2004929',)),
)


def annotate_neuron_roles(neuron_ids):
    """Annotate every input ID in order; preserve unknown roles explicitly.

    Requires the full canonical inventory, including all curated members.
    Missing names/duplicates raise instead of silently discarding annotations.
    """
    ids = tuple(neuron_ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Neuron IDs must be unique')
    known = set(ids)
    missing = {n for group in ROLE_GROUPS for n in group.neurons} - known
    if missing:
        raise ValueError('Missing annotated canonical IDs: '+', '.join(sorted(missing)))
    rows = []
    for index, neuron in enumerate(ids):
        groups = [g for g in ROLE_GROUPS if neuron in g.neurons]
        rows.append(dict(neuron=neuron, index=index,
            roles=sorted({r for group in groups for r in group.roles}),
            groups=[g.key for g in groups],
            annotation_status='curated_subset' if groups else 'unreviewed',
            sources=sorted({s for group in groups for s in group.sources})))
    return rows
