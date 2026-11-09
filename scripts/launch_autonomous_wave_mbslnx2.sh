#!/bin/sh
# Legacy launcher follows the current amplitude-only curriculum.
# Explicit phase/wave experiments remain available via the recorded source/CLI.
set -eu
exec sh "$(dirname -- "$0")/launch_body_amplitude_mbslnx2.sh" "$@"
