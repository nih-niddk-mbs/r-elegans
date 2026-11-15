"""Delete excess intermediate trajectory arrays, keeping checkpoints and references.

No /proc or /data access is needed. Only recognized raw-trace filenames from
completed, quiet fits are eligible. Default is a dry run; --apply deletes them.
"""
import argparse,hashlib,json,os,re,stat,time,shutil
from pathlib import Path

TRACE_NAMES={'trajectory.npz','active-trajectory.npz','sham-trajectory.npz','dose-low-trajectory.npz'}
CMA_TRACE_NAMES={'no_cue.npz','stop_only.npz','start_only.npz','stop_start.npz'}
RESERVED={'run_records','source','parent_worker','parent','inputs','control_inputs','swim_inputs','validation_inputs'}


def load_json(path):
    with path.open() as stream:return json.load(stream)


def walk(base):
    for folder,dirs,files in os.walk(base,followlinks=False):
        dirs[:]=[n for n in dirs if not n.startswith('.') and n not in RESERVED and 'input' not in n.lower() and not (Path(folder)/n).is_symlink()]
        yield Path(folder),files


def reference_inventory(base):
    """Protect explicit file references and selected-evaluation directories."""
    protected=set();directories=set();uncertain=set()
    def collect(value,context):
        if isinstance(value,str):
            if value.endswith('.npz') and ('/' in value):
                p=Path(value);protected.add((p if p.is_absolute() else context/p).resolve())
        elif isinstance(value,list):
            for x in value:collect(x,context)
        elif isinstance(value,dict):
            for key,x in value.items():
                if isinstance(key,str) and key.startswith('/') and key.endswith('.npz'):protected.add(Path(key).resolve())
                if key=='evaluation' and isinstance(x,str):directories.add(Path(x).resolve())
                collect(x,context)
    # Source/input manifests are relevant references even though they are not
    # deletion candidates; include them, omit only archived source-code copies.
    for folder,dirs,files in os.walk(base,followlinks=False):
        dirs[:]=[n for n in dirs if not n.startswith('.') and n not in ('run_records','source') and not (Path(folder)/n).is_symlink()]
        for name in files:
            if not name.endswith('.json'):continue
            path=Path(folder)/name
            try:collect(load_json(path),path.parent)
            except (OSError,ValueError):
                rel=path.relative_to(base)
                if rel.parts:uncertain.add(base/rel.parts[0])
    return protected,directories,uncertain


def completed_worker(worker,cutoff):
    summary=worker/'summary.json'
    if not summary.is_file() or summary.is_symlink() or summary.stat().st_mtime>cutoff:return None
    try:
        s=load_json(summary)
        if s.get('status')!='completed' and s.get('completed') is not True:return None
        return s
    except (OSError,ValueError):return None


def final_indices(summary,best):
    keep={0}
    for record in (summary,best):
        for key in ('selected_iteration','initial_iteration','iteration','last_iteration'):
            value=record.get(key)
            if isinstance(value,int):keep.add(value)
    return keep


def candidates(base,cutoff,protected,directories,uncertain):
    for folder,files in walk(base):
        if any(root==folder or root in folder.parents for root in uncertain):continue
        if folder.name=='selected':
            worker=folder.parent;s=completed_worker(worker,cutoff)
            if s is None:continue
            try:best=load_json(worker/'best.json') if (worker/'best.json').exists() else {}
            except (OSError,ValueError):continue
            keep=final_indices(s,best)
            iterations=[];legacy=[]
            for entry in folder.iterdir():
                match=re.fullmatch(r'iteration-(\d+)',entry.name)
                if match and entry.is_dir() and not entry.is_symlink():iterations.append((int(match[1]),entry))
                match=re.fullmatch(r'trace-(\d+)\.npz',entry.name)
                if match:legacy.append((int(match[1]),entry))
            indices=[i for i,_ in iterations+legacy]
            if indices:keep.update([min(indices),max(indices)])
            for index,path in iterations+legacy:
                if index in keep or s.get('selected')==path.name:continue
                for trace in ([path] if path.is_file() else [path/n for n in TRACE_NAMES]):
                    if not trace.exists() or trace.is_symlink():continue
                    resolved=trace.resolve()
                    if resolved in protected or any(d==resolved or d in resolved.parents for d in directories):continue
                    if trace.stat().st_mtime<=cutoff:yield trace,worker
        elif re.fullmatch(r'g-\d+-c-\d+',folder.name) and folder.parent.name=='evaluations':
            root=folder.parent.parent
            summary=root/'summary.json'
            if not summary.exists() or summary.stat().st_mtime>cutoff:continue
            try:
                result=load_json(summary)
                if not result.get('all_stages_finished'):continue
                state=load_json(folder/'summary.json')
                if state.get('status') not in ('completed','failed'):continue
            except (OSError,ValueError):continue
            generation=folder.name.split('-')[1]
            if not (root/'optimizer'/f'g-{generation}-told/metadata.json').exists():continue
            if folder.resolve() in directories:continue
            for name in CMA_TRACE_NAMES:
                trace=folder/name
                if trace.is_file() and not trace.is_symlink() and trace.resolve() not in protected and trace.stat().st_mtime<=cutoff:
                    yield trace,root


def prune(base,apply=False,minimum_age_hours=1.,event=lambda *a,**k:None):
    base=base.resolve();cutoff=time.time()-minimum_age_hours*3600
    protected,directories,uncertain=reference_inventory(base)
    event('inventory',protected_files=len(protected),selected_directories=len(directories),uncertain_runs=len(uncertain))
    count=0;freed=0;model_hashes={}
    for path,worker in candidates(base,cutoff,protected,directories,uncertain):
        before=path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:continue
        # Re-read worker selection metadata immediately before unlinking. If it
        # was touched since planning, leave this worker for a later cleanup.
        summary=worker/'summary.json'
        if not summary.exists() or summary.stat().st_mtime>cutoff:continue
        if worker not in model_hashes:
            model=worker/'network_best.npz'
            model_hashes[worker]=hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None
        details=dict(path=str(path),bytes=before.st_size,inode=before.st_ino,mtime_ns=before.st_mtime_ns,
                     checkpoint_sha256=model_hashes[worker],content_hash_recorded=False)
        event('delete_planned' if apply else 'would_delete',**details)
        now=path.stat()
        if (now.st_ino,now.st_size,now.st_mtime_ns)!=(before.st_ino,before.st_size,before.st_mtime_ns):continue
        if apply:
            path.unlink();event('deleted',**details)
        count+=1;freed+=before.st_size
    event('finished',apply=apply,files=count,bytes=freed)
    return count,freed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=Path('/home/carsonc/r-elegans-results'))
    p.add_argument('--apply',action='store_true');p.add_argument('--minimum-age-hours',type=float,default=1.)
    a=p.parse_args();base=a.base.expanduser().resolve()
    if not base.is_dir() or base in (Path('/'),Path.home()):p.error('Results directory required')
    if a.minimum_age_hours<1:p.error('At least1hour quiet age required')
    import fcntl
    lock=Path('/tmp')/f'r-elegans-prune-{os.getuid()}.lock'
    fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    journal=Path('/tmp')/f'r-elegans-prune-{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}.jsonl'
    def event(kind,**fields):
        row=dict(event=kind,**fields)
        with journal.open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(row),flush=True)
    event('start',base=str(base),apply=a.apply,journal=str(journal),program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    prune(base,a.apply,a.minimum_age_hours,event)
    if a.apply:
        records=base/'.cleanup-records';records.mkdir(exist_ok=True)
        retained=records/journal.name
        shutil.copy2(journal,retained)
        print('Retained deletion journal:',retained,flush=True)
    print('Deletion journal:',journal,flush=True)


if __name__=='__main__':main()
