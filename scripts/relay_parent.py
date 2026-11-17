"""Explicit pinned-parent arguments for reproducible relay continuations."""
import re


def add_arguments(parser):
    parser.add_argument('--parent-sha256',help='Exact SHA256 of source run selected checkpoint')
    parser.add_argument('--experiment-id',help='Planned identifier for this new experiment')
    parser.add_argument('--seed',type=int,help='Fresh CMA seed recorded for this continuation')


def validate(parser,args,allowed):
    specified=any(x is not None for x in (args.parent_sha256,args.experiment_id,args.seed))
    if not specified:return False
    if not allowed or any(x is None for x in (args.parent_sha256,args.experiment_id,args.seed)):
        parser.error('Explicit parent requires relay continuation and all of --parent-sha256, --experiment-id, --seed')
    if not re.fullmatch('[0-9a-f]{64}',args.parent_sha256):parser.error('Parent SHA256 must be 64 lowercase hex characters')
    if not re.fullmatch('E[0-9]{3,}',args.experiment_id):parser.error('Experiment identifier must be E followed by at least 3 digits')
    if not 0<args.seed<2**32:parser.error('Seed must be in 1..2**32-1')
    return True
