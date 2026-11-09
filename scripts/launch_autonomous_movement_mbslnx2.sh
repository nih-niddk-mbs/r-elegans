#!/bin/sh
# Current curriculum first develops body amplitude. Old E031/E032
# endpoint-score experiments remain reproducible via their recorded source.
set -eu
exec sh "$(dirname -- "$0")/launch_body_amplitude_mbslnx2.sh" "$@"
