# navigation -> infra
`docker/dds/cyclonedds.xml` sets `WhcHigh` 1 MB and a 2 MB receive buffer. `/map` and `/global_costmap/costmap` are 0.4-0.6 MB each and published reliably; navigation saw costmap stalls that may be linked. Please raise `WhcHigh` (8 MB tested) and `SocketReceiveBufferSize` (8 MB), and document pinning `<Interfaces>` for containers on two networks.
