"""thermal_ingest: optional second stream -> /thermal/image_raw (mono8, hot = bright) + /thermal/camera_info.

The stock X30 has no thermal camera (digest 6.3): with `thermal.enabled: false` or an empty url this node logs once and
exits cleanly, and every consumer must work with /thermal/* absent."""

from x30_hw.ingest_common import IngestNode, run


class ThermalIngest(IngestNode):
    def __init__(self):
        super().__init__("thermal_ingest", "thermal", "/thermal/image_raw", "/thermal/camera_info", mono=True)


def main(args=None):
    run(ThermalIngest, args)


if __name__ == "__main__":
    main()
