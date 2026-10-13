#!/usr/bin/env bash
# kill_nodes.sh <pattern>... : kill processes in e2e-fw whose command line matches (does not kill itself)
for pat in "$@"; do docker exec e2e-fw bash -c "for p in \$(pgrep -f '$pat'); do [ \$p != \$\$ ] && kill \$p 2>/dev/null; done; true"; done
