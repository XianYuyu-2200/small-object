from swallow_yolo.mindvision import camera_diagnostics, resolution_by_index


class FakeResolution:
    iIndex = 0
    iWidth = 640
    iHeight = 480
    iWidthFOV = 640
    iHeightFOV = 480

    def GetDescription(self):
        return "VGA"


class FakePreset7(FakeResolution):
    iIndex = 7

    def GetDescription(self):
        return "Custom"


class FakeCapability:
    iImageSizeDesc = 2
    pImageSizeDesc = [FakeResolution(), FakePreset7()]
    class sResolutionRange:
        iWidthMin = 64
        iWidthMax = 5488
        iHeightMin = 64
        iHeightMax = 3672
        uSkipModeMask = 1
        uBinAverageModeMask = 3
        uResampleMask = 0


class FakeSdk:
    def CameraGetImageResolution(self, _): return FakeResolution()
    def CameraGetTriggerMode(self, _): return 0
    def CameraGetAeState(self, _): return 1
    def CameraGetExposureTime(self, _): return 12345.0
    def CameraGetExposureTimeRange(self, _): return (10.0, 1_000_000.0, 1.0)
    def CameraGetAnalogGainX(self, _): return 4.0
    def CameraGetAnalogGainXRange(self, _): return (1.0, 22.0, 0.125)
    def CameraGetFrameSpeed(self, _): return 2


def test_camera_diagnostics_exposes_current_capture_conditions():
    values = camera_diagnostics(FakeSdk(), 1, FakeCapability())

    assert values["resolution"] == "640x480"
    assert values["trigger_mode"] == 0
    assert values["auto_exposure"] is True
    assert values["exposure_us"] == 12345.0
    assert values["analog_gain_x"] == 4.0
    assert values["analog_gain_x_range"] == [1.0, 22.0, 0.125]
    assert values["resolution_range"] == "64x64..5488x3672"
    assert values["preset_resolutions"] == [
        {"index": 0, "description": "VGA", "output": "640x480", "fov": "640x480"},
        {"index": 7, "description": "Custom", "output": "640x480", "fov": "640x480"},
    ]


def test_resolution_by_index_uses_vendor_preset_id_not_array_position():
    class Capability:
        iImageSizeDesc = 2
        pImageSizeDesc = [type("Preset", (), {"iIndex": 0})(), type("Preset", (), {"iIndex": 7})()]

    assert resolution_by_index(Capability(), 7).iIndex == 7
