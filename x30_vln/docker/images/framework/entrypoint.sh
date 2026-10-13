#!/bin/bash
# Source ROS + the workspace overlay (if built) and run the command.
. /usr/local/bin/x30-env
exec "$@"
