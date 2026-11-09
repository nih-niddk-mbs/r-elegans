#!/bin/sh
# Current curriculum first develops intrinsic body waves. Old E031/E032
# endpoint-score experiments remain reproducible via their recorded source.
set -eu
exec sh "$(dirname -- "$0")/launch_autonomous_wave_mbslnx2.sh"
