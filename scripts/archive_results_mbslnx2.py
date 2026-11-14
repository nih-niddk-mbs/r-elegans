"""Free /home by deleting only verified duplicates of inactive results on /data.

No trajectory/checkpoint is permanently discarded. Old absolute paths become
symlinks. Requires Linux atomic rename exchange and rsync checksum verification.
"""
import argparse,ctypes,hashlib,json,os,shutil,stat,subprocess,time,sys,platform
from pathlib import Path


def tree_state(root):
    """Fingerprint entries and catch writers/links; contents are verified by rsync."""
    rows=[];size=0;newest=0.
    for folder,dirs,files in os.walk(root,followlinks=False):
        for path in [Path(folder),*[Path(folder)/n for n in dirs+files]]:
            s=path.lstat()
            if stat.S_ISLNK(s.st_mode):raise ValueError('Contains symlinks; preserve for separate review')
            if not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):raise ValueError('Contains special files')
            rows.append((str(path.relative_to(root)),s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode))
            newest=max(newest,s.st_mtime)
            if stat.S_ISREG(s.st_mode):size+=s.st_size
    return hashlib.sha256(json.dumps(sorted(set(rows))).encode()).hexdigest(),size,newest


def referenced_runs(base,proc_root=Path('/proc')):
    """Conservative own-user process scan; do not retain/log unrelated arguments."""
    active=set();uid=os.getuid();prefix=str(base)+'/'
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=uid:continue
            text=(proc/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
            paths=[text]
            for name in ('cwd',):
                try:paths.append(os.readlink(proc/name))
                except FileNotFoundError:pass
            try:
                for fd in (proc/'fd').iterdir():
                    try:paths.append(os.readlink(fd))
                    except FileNotFoundError:pass
            except FileNotFoundError:continue
            for value in paths:
                for part in value.split(prefix)[1:]:
                    name=part.split('/',1)[0].split()[0].strip("\"'") if part.strip() else ''
                    if name.endswith('.log'):name=name[:-4]
                    if name:active.add(name)
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError as e:raise RuntimeError('Cannot inspect all own-user processes; refusing cleanup') from e
    return active


def atomic_exchange(a,b):
    """Swap directory and replacement symlink atomically; fail closed elsewhere."""
    libc=ctypes.CDLL(None,use_errno=True)
    try:call=libc.renameat2
    except AttributeError:raise RuntimeError('Linux renameat2 required; no originals deleted')
    call.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    call.restype=ctypes.c_int
    if call(-100,os.fsencode(a),-100,os.fsencode(b),2):
        number=ctypes.get_errno();raise OSError(number,os.strerror(number))


def checksum_differences(source,destination):
    result=subprocess.run(['rsync','-aHnc','--delete','--itemize-changes','--',str(source)+'/',str(destination)+'/'],capture_output=True,text=True,check=True)
    return result.stdout.strip()


def open_references(path,proc_root=Path('/proc')):
    """After exchange, old open descriptors would still refer to duplicate tree."""
    prefix=str(path);uid=os.getuid()
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=uid:continue
            try:
                cwd=os.readlink(proc/'cwd')
                if cwd==prefix or cwd.startswith(prefix+'/'):return True
            except FileNotFoundError:pass
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except FileNotFoundError:continue
                if target==prefix or target.startswith(prefix+'/'):return True
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:raise RuntimeError('Unable to verify old file descriptors; duplicate retained')
    return False


def migrate(source,destination,event):
    if source.is_symlink():return 0
    before,size,_=tree_state(source)
    partial=destination.parent/('.'+destination.name+'.copying')
    if destination.exists() or partial.is_symlink():raise ValueError('Archive target already exists; inspect previous journal')
    partial.mkdir(exist_ok=True)
    event('copying',source=str(source),destination=str(destination),apparent_bytes=size)
    subprocess.run(['rsync','-aH','--sparse','--progress','--',str(source)+'/',str(partial)+'/'],check=True)
    if source.name in referenced_runs(source.parent):raise RuntimeError('Run became active; source retained')
    after,_,_=tree_state(source)
    if before!=after:raise RuntimeError('Source changed during copy; source retained')
    changes=checksum_differences(source,partial)
    if changes:raise RuntimeError('Checksum verification differs; source retained: '+changes[:300])
    if tree_state(source)[0]!=before:raise RuntimeError('Source changed during verification; source retained')
    # Verified destination is durable before publishing the old path as a symlink.
    partial.rename(destination);os.sync()
    event('verified',source=str(source),destination=str(destination),fingerprint=before)
    if source.name in referenced_runs(source.parent) or tree_state(source)[0]!=before:
        raise RuntimeError('Source became active/changed before exchange; both copies retained')
    temporary=source.parent/('.archive-link-'+source.name+'-'+str(os.getpid()))
    if temporary.exists() or temporary.is_symlink():raise ValueError('Temporary exchange path exists')
    temporary.symlink_to(destination,target_is_directory=True)
    try:atomic_exchange(source,temporary)
    except Exception:
        temporary.unlink();raise
    parent_fd=os.open(source.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    event('path_redirected',source=str(source),destination=str(destination),duplicate=str(temporary))
    # Do not delete if an unexpected writer retained an old descriptor.
    if open_references(temporary):raise RuntimeError('Old descriptors remain open; old duplicate retained at '+str(temporary))
    if tree_state(temporary)[0]!=before:
        # Root ctime may change due to rename; verify content rather than assume loss.
        if checksum_differences(temporary,destination):raise RuntimeError('Old duplicate changed; both versions retained')
    elif checksum_differences(temporary,destination):raise RuntimeError('Post-exchange verification differs; both retained')
    shutil.rmtree(temporary)
    event('home_duplicate_deleted',source=str(source),destination=str(destination),apparent_bytes_relocated=size)
    return size


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/home/carsonc/r-elegans-results'))
    p.add_argument('--destination',type=Path,default=Path('/data/carsonc/r-elegans-results'))
    p.add_argument('--apply',action='store_true');p.add_argument('--minimum-age-minutes',type=float,default=30.)
    a=p.parse_args();source=a.source.expanduser().resolve();destination=a.destination.expanduser().resolve()
    if not source.is_dir() or source==Path('/') or source==Path.home():p.error('Results directory required')
    if destination==source or source in destination.parents or destination in source.parents:p.error('Separate archive destination required')
    if a.minimum_age_minutes<30:p.error('Minimum quiet age30min required')
    if not shutil.which('rsync'):p.error('rsync required')
    lock_fd=None
    if a.apply:
        import fcntl
        lock=Path('/tmp')/('r-elegans-archive-'+str(os.getuid())+'-'+hashlib.sha256(str(source).encode()).hexdigest()[:12]+'.lock')
        lock_fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        destination.mkdir(parents=True,exist_ok=True)
        if source.stat().st_dev==destination.stat().st_dev:p.error('Destination must be on another filesystem')
    log=Path('/tmp')/f'r-elegans-archive-{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}.jsonl'
    def event(kind,**fields):
        row=dict(event=kind,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
        print(json.dumps(row),flush=True)
        with log.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    program=Path(__file__).resolve()
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=program.parents[1],capture_output=True,text=True)
    event('start',apply=a.apply,source=str(source),destination=str(destination),journal=str(log),
        argv=sys.argv,python=sys.executable,python_version=sys.version,platform=platform.platform(),
        program_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),source_revision=revision.stdout.strip() if revision.returncode==0 else None)
    if a.apply:
        snapshot=destination/('.maintenance-source-'+str(os.getpid())+'-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
        snapshot.mkdir();shutil.copy2(program,snapshot/program.name)
        event('source_snapshot',path=str(snapshot))
    active=referenced_runs(source);candidates=[]
    for run in source.iterdir():
        if run.name.startswith('.') or run.is_symlink() or not run.is_dir():continue
        if run.name in active:event('skip_active',run=run.name);continue
        try:
            fingerprint,size,newest=tree_state(run)
            if time.time()-newest<a.minimum_age_minutes*60:event('skip_recent',run=run.name);continue
            candidates.append((size,run))
        except (OSError,ValueError) as e:event('skip',run=run.name,reason=str(e))
    total=0;failed=0
    for size,run in sorted(candidates,key=lambda row:row[0],reverse=True):
        event('candidate',run=run.name,apparent_bytes=size)
        if not a.apply:continue
        if run.name in referenced_runs(source):event('skip_active',run=run.name);continue
        # Reserve at least10GiB on /data beyond this copy's apparent size.
        if shutil.disk_usage(destination).free<size+10*1024**3:
            event('insufficient_archive_space',run=run.name);failed+=1;continue
        try:total+=migrate(run,destination/run.name,event)
        except Exception as e:event('retained_after_error',run=run.name,error=str(e));failed+=1
    event('finished',apparent_bytes_relocated=total,errors=failed,home_free_bytes=shutil.disk_usage(source).free)
    if a.apply:
        # Preserve the journal on the archive disk, never fill /home with logs.
        shutil.copy2(log,destination/log.name)
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
