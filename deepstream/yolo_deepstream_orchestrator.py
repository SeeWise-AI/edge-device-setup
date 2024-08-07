import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib

class YoloPersonDetector:
    def __init__(self, video_source, model_config_path, model_weights_path, labels_path):
        self.video_source = video_source
        self.model_config_path = model_config_path
        self.model_weights_path = model_weights_path
        self.labels_path = labels_path
        
        # Initialize GStreamer
        Gst.init(None)

        # Create GStreamer pipeline
        self.pipeline = Gst.Pipeline()

        # Create elements
        self.source = Gst.ElementFactory.make("filesrc", "file-source")
        self.decoder = Gst.ElementFactory.make("decodebin", "decoder")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        self.yolo = Gst.ElementFactory.make("nvinfer", "yolo-detector")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "video-converter")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.sink = Gst.ElementFactory.make("nveglglessink", "video-renderer")

        # Check for element creation errors
        if not all([self.pipeline, self.source, self.decoder, self.streammux, self.yolo, self.nvvidconv, self.nvosd, self.sink]):
            raise Exception("Failed to create one or more GStreamer elements")

        # Set properties
        self.source.set_property('location', self.video_source)
        self.yolo.set_property('config-file-path', self.model_config_path)
        self.yolo.set_property('model-engine-file', self.model_weights_path)
        self.yolo.set_property('labelfile-path', self.labels_path)

        # Link elements
        self.pipeline.add(self.source)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.yolo)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)

        self.source.link(self.decoder)
        self.decoder.connect("pad-added", self._decodebin_pad_added)
        self.streammux.link(self.yolo)
        self.yolo.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.sink)

    def _decodebin_pad_added(self, decodebin, pad):
        caps = pad.get_current_caps()
        structure = caps.get_structure(0)
        name = structure.get_name()
        if name.startswith("video/x-raw"):
            sinkpad = self.streammux.get_request_pad("sink_0")
            pad.link(sinkpad)

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            loop = GLib.MainLoop()
            loop.run()
        except:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Person Detector using NVIDIA Deepstream")
    parser.add_argument('--video_source', required=True, help="Path to the video file")
    parser.add_argument('--model_config_path', required=True, help="Path to the YOLO config file")
    parser.add_argument('--model_weights_path', required=True, help="Path to the YOLO weights file")
    parser.add_argument('--labels_path', required=True, help="Path to the labels file")

    args = parser.parse_args()

    detector = YoloPersonDetector(args.video_source, args.model_config_path, args.model_weights_path, args.labels_path)
    detector.run()
