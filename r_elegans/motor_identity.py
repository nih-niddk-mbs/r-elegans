"""Explicit motor identities; AS sensory names are not numbered AS motor cells."""
NUMBERED_CLASSES=('DA','DB','DD','VA','VB','VD','AS')
HEAD_CLASSES=('RMD','SMD','SMB')


def numbered_motor(name):
    return any(name.startswith(p) and name[len(p):].isdigit() for p in NUMBERED_CLASSES)


def is_motor(name):
    return numbered_motor(name) or name.startswith(HEAD_CLASSES)


def output_group(name):
    if numbered_motor(name):return 'B' if name.startswith(('DB','VB')) else 'other-body'
    if name.startswith(HEAD_CLASSES):return 'head'
    return 'other'
