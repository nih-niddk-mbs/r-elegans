import re
import jax.numpy as jnp
import numpy as np
from r_elegans.full_body_assessment import regional_amplitudes,rod_nodes,write_motion_preview
from r_elegans.body.mechanics import relative_segment_centers


def test_regions_measure_independent_amplitudes_and_reject_static_and_ramps():
    t=np.arange(0,20,.002);q=np.sin(2*np.pi*t[:,None]+np.arange(11)*np.pi)*np.asarray([.002]*4+[.003]*4+[.004]*3)
    r=regional_amplitudes(q,q)
    np.testing.assert_allclose(r['regional_joint_median_p2p_rad'],[.004,.006,.008],atol=1e-6)
    assert r['all_three_regional_amplitude_targets_met']
    assert r['region_order']==['tail','middle','head']
    q[:,:4]=.1
    assert not regional_amplitudes(q,q)['all_three_regional_amplitude_targets_met']
    q[:,:4]=t[:,None]*.001
    assert regional_amplitudes(q,q)['regional_joint_median_p2p_rad'][0]==0


def test_preview_nodes_match_existing_mechanics_and_preserve_rod_lengths():
    q=.2*np.sin(np.arange(11));heading=.7;p=np.array([.003,-.002]);length=.001
    nodes=rod_nodes(q[None],p[None],np.array([heading]),length)[0]
    centers=.5*(nodes[:-1]+nodes[1:])
    local=np.asarray(relative_segment_centers(jnp.asarray(q),jnp.asarray(length/12)))
    rotation=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
    np.testing.assert_allclose(centers,local@rotation.T+p,atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(np.diff(nodes,axis=0),axis=1),length/12,atol=1e-12)
    np.testing.assert_allclose(centers.mean(axis=0),p,atol=1e-12)


def test_preview_defaults_to_true_geometry_caps_frames_and_labels_amplification(tmp_path):
    t=np.arange(0,30,.002);q=.003*np.sin(2*np.pi*t[:,None])*np.ones((1,11))
    path=tmp_path/'motion.html'
    info=write_motion_preview(path,q,np.zeros((len(t),2)),np.zeros(len(t)),dt=.002,body_length=.001,seconds=20,title='Local <test>')
    text=path.read_text()
    assert info['frames']<=600 and info['default_gain']==1
    assert '1× true angles' in text and '20× visual amplification' in text
    assert 'Local &lt;test&gt;' in text and 'not physical results' in text
    assert 'https://' not in text and 'src=' not in text
    assert '__DATA__' not in text and '__TITLE__' not in text
    assert len(re.findall(r'<script>',text))==1


def test_assessment_import_does_not_initialize_jax():
    import subprocess
    import sys
    subprocess.run([sys.executable,'-c',
        "import sys;import scripts.assess_full_body;assert 'jax' not in sys.modules"],check=True)


def test_full_body_pass_requires_all_regions_refinement_and_quiet_control():
    import copy
    from r_elegans.full_body_assessment import full_body_checks
    cases=[]
    for name in ('long','perturb-plus','perturb-minus','brain-refined','body-refined','held-voltage'):
        held=name=='held-voltage'
        cases.append(dict(case=name,recorded_full_body_amplitude_qualified=not held,recorded_neural_rhythms=0 if held else 3,
            regions=dict(regional_joint_median_p2p_rad=[0,0,0] if held else [.004,.006,.008],
                all_three_regional_amplitude_targets_met=not held,raw_late_joint_p2p_max_rad=0 if held else .01)))
    assert full_body_checks(cases,[.004,.006,.008])['full_body_amplitude_assay_pass']
    drift=copy.deepcopy(cases);drift[3]['regions']['regional_joint_median_p2p_rad'][1]=.012
    r=full_body_checks(drift,[.004,.006,.008])
    assert r['all_active_amplitude_checks_pass'] and not r['full_body_amplitude_assay_pass']
    cases[-1]['recorded_neural_rhythms']=1
    assert not full_body_checks(cases,[.004,.006,.008])['full_body_amplitude_assay_pass']
