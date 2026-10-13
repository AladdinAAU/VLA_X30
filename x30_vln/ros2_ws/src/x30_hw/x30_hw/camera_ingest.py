"""camera_ingest: RGB camera stream (RTSP) -> /camera/color/image_raw (rgb8) + /camera/color/camera_info."""

from x30_hw.ingest_common import IngestNode, run


class CameraIngest(IngestNode):
    def __init__(self):
        super().__init__("camera_ingest", "camera", "/camera/color/image_raw", "/camera/color/camera_info", mono=False)


def main(args=None):
    run(CameraIngest, args)


if __name__ == "__main__":
    main()
