"""Read-only amplitude summaries and true-geometry previews of saved body states."""
import html
import json
from pathlib import Path
import numpy as np
from .amplitude_analysis import sustained_excursions


def regional_amplitudes(joint_angles,filtered_drive):
    q=np.asarray(joint_angles);a=np.asarray(filtered_drive)
    if q.ndim!=2 or q.shape[1]!=11 or a.shape!=q.shape:raise ValueError('Expected eleven recorded joints/drives')
    qa,qr=sustained_excursions(q);aa,ar=sustained_excursions(a)
    groups=np.array_split(np.arange(11),3)
    medians=np.asarray([np.median(qa[g]) for g in groups])
    return dict(region_order=['tail','middle','head'],regional_joint_median_p2p_rad=medians.tolist(),
        regional_filtered_drive_median_p2p=[float(np.median(aa[g])) for g in groups],
        joint_sustained_p2p_rad=qa.tolist(),filtered_drive_sustained_p2p=aa.tolist(),
        raw_late_joint_p2p_max_rad=float(np.ptp(q[len(q)//2:],axis=0).max()),
        repeated_joint_count=int(qr.sum()),repeated_drive_count=int(ar.sum()),
        all_three_regional_amplitude_targets_met=bool(np.all(medians>=.003)))


def full_body_checks(cases,reference_amplitudes):
    """Reapply existing E037 regional thresholds/refinement/control to saved data."""
    by_name={r['case']:r for r in cases}
    expected={'long','perturb-plus','perturb-minus','brain-refined','body-refined','held-voltage'}
    if set(by_name)!=expected or len(cases)!=6:raise ValueError('Six unique validation cases required')
    reference=np.asarray(reference_amplitudes,dtype=float)
    if reference.shape!=(3,) or not np.isfinite(reference).all() or np.any(reference<0):raise ValueError('Three finite reference amplitudes required')
    refinement={}
    for name in ('brain-refined','body-refined'):
        values=np.asarray(by_name[name]['regions']['regional_joint_median_p2p_rad'])
        relative=np.abs(values-reference)/np.maximum(reference,.003)
        refinement[name]=dict(relative_error_with_target_floor=relative.tolist(),within_25_percent=bool(np.all(relative<=.25)))
    active=[r for r in cases if r['case']!='held-voltage']
    above=all(r['recorded_full_body_amplitude_qualified'] is True and r['regions']['all_three_regional_amplitude_targets_met'] for r in active)
    held=by_name['held-voltage']
    control=held.get('recorded_neural_rhythms')==0 and held['regions']['raw_late_joint_p2p_max_rad']<1e-5
    return dict(existing_E037_criteria_reapplied=True,all_active_amplitude_checks_pass=bool(above),
        reference_bends_tail_middle_head=reference.tolist(),regional_refinement=refinement,held_voltage_control_pass=bool(control),
        full_body_amplitude_assay_pass=bool(above and control and all(r['within_25_percent'] for r in refinement.values())))


def rod_nodes(q,position,heading,body_length):
    """Twelve equal rods,center of segment centers,then world rotation/position.

    Matches mechanics.relative_segment_centers/world_segment_centers. First
    endpoint is tail,last is head. Stroke width is not physical worm thickness.
    """
    q=np.asarray(q,dtype=float);p=np.asarray(position,dtype=float);h=np.asarray(heading,dtype=float)
    if q.ndim!=2 or q.shape[1]!=11 or p.shape!=(len(q),2) or h.shape!=(len(q),):raise ValueError('Invalid body state shapes')
    if not np.isfinite(body_length) or body_length<=0 or not all(np.isfinite(x).all() for x in (q,p,h)):raise ValueError('Finite states and positive body length required')
    angles=np.column_stack((np.zeros(len(q)),np.cumsum(q,axis=1)))
    tangents=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    vertices=np.concatenate((np.zeros((len(q),1,2)),np.cumsum(body_length/12*tangents,axis=1)),axis=1)
    center=((vertices[:,:-1]+vertices[:,1:])*.5).mean(axis=1)
    relative=vertices-center[:,None,:]
    c,s=np.cos(h),np.sin(h)
    rotation=np.stack((np.stack((c,-s),axis=-1),np.stack((s,c),axis=-1)),axis=1)
    return relative@np.swapaxes(rotation,1,2)+p[:,None,:]


def write_motion_preview(path,q,position,heading,*,dt,body_length,seconds,title):
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(seconds) or seconds<=0:raise ValueError('Positive preview timing required')
    q=np.asarray(q);p=np.asarray(position);h=np.asarray(heading)
    start=max(len(q)//2,len(q)-int(np.ceil(seconds/dt)))
    stride=max(1,int(np.ceil(1/(60*dt))),int(np.ceil((len(q)-start)/600)))
    ids=np.arange(start,len(q),stride)
    if len(ids)<2:raise ValueError('Preview needs two frames')
    qq=q[ids];pp=p[ids]-p[ids[0]];hh=h[ids]
    world=rod_nodes(qq,pp,hh,body_length)/body_length
    mean=q[len(q)//2:].mean(axis=0)
    shapes={str(g):np.round(rod_nodes(mean+g*(qq-mean),np.zeros_like(pp),np.zeros_like(hh),body_length)/body_length,8).tolist() for g in (1,10,20)}
    data=dict(times_s=((ids+1)*dt).tolist(),world=np.round(world,8).tolist(),shapes=shapes,
        source_sample_dt_s=dt,preview_stride=stride,frames=len(ids),preview_step_s=stride*dt,
        region_order=['tail','middle','head'])
    payload=json.dumps(data,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:16px system-ui;margin:24px;max-width:1150px;color:#17233b;background:#f6f8fb}canvas{width:100%;background:white;border:1px solid #ccd4df;border-radius:10px}button,select,input{font:inherit;margin:8px}#seek{width:65%}.note{color:#57657b;font-size:14px}#gainnote{color:#9b3d00}</style>
<h1>__TITLE__</h1><p>Saved trajectory only. No simulation or fitted model changes.</p>
<canvas id="view" width="1100" height="480"></canvas><br>
<button id="play">Pause</button><input id="seek" type="range" min="0" value="0" step="1"><span id="time"></span><br>
<label>Playback <select id="rate"><option value="0.25">0.25×</option><option value="1" selected>1× simulated time</option><option value="2">2×</option></select></label>
<label>Centered-view oscillatory deviation <select id="gain"><option value="1" selected>1× true angles</option><option value="10">10× visual amplification</option><option value="20">20× visual amplification</option></select></label>
<p id="gainnote">Both views use true angles.</p><p class="note" id="sampling"></p>
<p class="note">Left: true world geometry,origin at preview-start body center. Right: translation and heading removed. Optional gain multiplies deviations from the late-window mean posture only;amplified shapes are illustrations,not physical results. Axis units are body lengths with equal x/y scale. Rod colors:tail red,middle blue,head green;stroke width is schematic. This preview alone does not establish propulsion or coordination.</p>
<script>const data=__DATA__;const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const seek=document.getElementById('seek'),gain=document.getElementById('gain'),rate=document.getElementById('rate');seek.max=data.frames-1;
document.getElementById('sampling').textContent=`Final ${(data.times_s.at(-1)-data.times_s[0]).toFixed(2)} s shown;${data.frames} preview frames,one every ${data.preview_step_s.toFixed(4)} simulated seconds. Full metrics use all source samples (dt ${data.source_sample_dt_s} s).`;
let playing=true,frame=0,last=performance.now(),elapsed=0;
function bounds(frames){let x=[],y=[];for(const f of frames)for(const p of f){x.push(p[0]);y.push(p[1]);}return [(Math.min(...x)+Math.max(...x))/2,(Math.min(...y)+Math.max(...y))/2,Math.max(1.2,Math.max(...x)-Math.min(...x),Math.max(...y)-Math.min(...y))*1.25];}
const worldBounds=bounds(data.world);const shapeBounds={};for(const g of ['1','10','20'])shapeBounds[g]=bounds(data.shapes[g]);
function panel(nodes,x,title,b){let cx=x+265,cy=255,scale=Math.min(490,350)/b[2];let map=p=>[cx+(p[0]-b[0])*scale,cy-(p[1]-b[1])*scale];ctx.fillStyle='#17233b';ctx.font='17px system-ui';ctx.fillText(title,x+20,32);
ctx.strokeStyle='#e6ebf1';ctx.lineWidth=1;for(let k=-2;k<=2;k++){ctx.beginPath();ctx.moveTo(x+20,cy+k*scale*.25);ctx.lineTo(x+510,cy+k*scale*.25);ctx.stroke();}
for(let i=0;i<12;i++){let a=map(nodes[i]),b=map(nodes[i+1]);ctx.strokeStyle=i<4?'#c94e52':i<8?'#2868b2':'#268562';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
ctx.fillStyle='#17233b';ctx.font='15px system-ui';for(const [n,label]of [[0,'T'],[12,'H']]){const p=map(nodes[n]);ctx.fillText(label,p[0]+7,p[1]-8);}
ctx.strokeStyle='#17233b';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x+25,445);ctx.lineTo(x+25+scale*.25,445);ctx.stroke();ctx.fillText('0.25 body lengths',x+25,468);}
function draw(){ctx.clearRect(0,0,1100,480);panel(data.world[frame],0,'World view · true angles',worldBounds);panel(data.shapes[gain.value][frame],550,`Centered shape · ${gain.value}× oscillatory deviation`,shapeBounds[gain.value]);document.getElementById('time').textContent=data.times_s[frame].toFixed(3)+' s';seek.value=frame;document.getElementById('gainnote').textContent=gain.value==='1'?'Both views use true angles.':`Centered shape is visually amplified ${gain.value}×;world view remains true scale.`;}
seek.oninput=()=>{frame=Number(seek.value);playing=false;document.getElementById('play').textContent='Play';elapsed=frame*data.preview_step_s;draw();};gain.onchange=draw;
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';last=performance.now();};
function tick(now){if(playing){elapsed+=(now-last)/1000*Number(rate.value);frame=Math.floor(elapsed/data.preview_step_s)%data.frames;draw();}last=now;requestAnimationFrame(tick);}draw();requestAnimationFrame(tick);
</script></html>'''
    Path(path).write_text(template.replace('__TITLE__',html.escape(title)).replace('__DATA__',payload))
    return dict(frames=len(ids),nominal_start_s=float(data['times_s'][0]),nominal_end_s=float(data['times_s'][-1]),
        stride=stride,default_gain=1,world_geometry='true twelve-rod geometry,equal axes',centered_gain='1/10/20x oscillatory deviation,explicit visual-only label')
