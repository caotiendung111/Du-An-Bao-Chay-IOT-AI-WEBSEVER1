import sys
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

# Inject mock modules into sys.modules to prevent ModuleNotFoundError in test environment
sys.modules['telegram'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()

import pytest
import numpy as np

from fire_detection import FireDetector
import config

@pytest.fixture
def mock_detector():
    """Provides a FireDetector instance with mocked alert components."""
    with patch('fire_detection.TelegramAlert'):
        # Mock YOLO model call return value structure
        detector = FireDetector(model_path="dummy_path.pt")
        detector.model = MagicMock()
        
        # Override async methods with AsyncMock so they are awaitable
        detector.alert_system.send_alert_photo = AsyncMock()
        detector.alert_system.send_alert_message = AsyncMock()
        yield detector

def test_initial_state(mock_detector):
    """Verifies that the detector starts with correct default counters and states."""
    assert mock_detector.fire_counter == 0
    assert not mock_detector.is_alerting

def test_detect_no_fire(mock_detector):
    """Verifies detect returns False when YOLO returns no bounding boxes."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock YOLO result containing no boxes
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_result.plot.return_value = fake_frame
    mock_detector.model.return_value = [mock_result]
    
    has_fire, ann_frame = mock_detector.detect(fake_frame)
    
    assert not has_fire
    assert np.array_equal(ann_frame, fake_frame)

def test_detect_fire(mock_detector):
    """Verifies detect returns True when YOLO returns bounding boxes."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock YOLO result containing boxes
    mock_result = MagicMock()
    mock_result.boxes = [MagicMock()]
    mock_result.plot.return_value = fake_frame
    mock_detector.model.return_value = [mock_result]
    
    has_fire, ann_frame = mock_detector.detect(fake_frame)
    
    assert has_fire

def test_noise_filter_and_alert_trigger(mock_detector):
    """Ensures alarm is triggered only after exactly FRAME_WINDOW consecutive fire frames."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_detector.trigger_hardware = MagicMock()
    
    async def run_test():
        # Run loop up to config.FRAME_WINDOW - 1 (9 frames)
        for i in range(config.FRAME_WINDOW - 1):
            await mock_detector.process_fire_logic(has_fire=True, frame=fake_frame)
            assert mock_detector.fire_counter == i + 1
            assert not mock_detector.is_alerting
            mock_detector.trigger_hardware.assert_not_called()
            
        # Send the 10th consecutive frame to trigger the alarm
        await mock_detector.process_fire_logic(has_fire=True, frame=fake_frame)
        
        assert mock_detector.fire_counter == config.FRAME_WINDOW
        assert mock_detector.is_alerting
        mock_detector.trigger_hardware.assert_called_once_with(on=True)
        mock_detector.alert_system.send_alert_photo.assert_called_once()
        mock_detector.alert_system.send_alert_message.assert_called_once()

    asyncio.run(run_test())

def test_alarm_cooldown_and_reset(mock_detector):
    """Ensures alert resets to False only when fire detections return to 0."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_detector.trigger_hardware = MagicMock()
    
    # Pre-set system state to alerting with maximum frame counter
    mock_detector.is_alerting = True
    mock_detector.fire_counter = config.FRAME_WINDOW
    
    async def run_test():
        # Simulate fire disappearing for 1 frame (counter drops to 9, should still alert)
        await mock_detector.process_fire_logic(has_fire=False, frame=fake_frame)
        assert mock_detector.fire_counter == config.FRAME_WINDOW - 1
        assert mock_detector.is_alerting
        
        # Simulate fire disappearing completely (decrement counter to 0)
        for _ in range(config.FRAME_WINDOW - 1):
            await mock_detector.process_fire_logic(has_fire=False, frame=fake_frame)
            
        assert mock_detector.fire_counter == 0
        assert not mock_detector.is_alerting
        mock_detector.trigger_hardware.assert_called_once_with(on=False)

    asyncio.run(run_test())

@patch('requests.get')
def test_trigger_hardware_api_calls(mock_get, mock_detector):
    """Verifies that trigger_hardware sends correct HTTP commands to ESP32."""
    config.ESP32_IP = "192.168.1.100"
    config.RELAY_ALARM_PIN = 12
    config.RELAY_PUMP_PIN = 13
    
    mock_detector.trigger_hardware(on=True)
    
    # Should perform two separate GET requests to control alarm and pump
    assert mock_get.call_count == 2
    mock_get.assert_any_call("http://192.168.1.100/relay?pin=12&state=on", timeout=2)
    mock_get.assert_any_call("http://192.168.1.100/relay?pin=13&state=on", timeout=2)
    
    # Trigger OFF
    mock_get.reset_mock()
    mock_detector.trigger_hardware(on=False)
    
    assert mock_get.call_count == 2
    mock_get.assert_any_call("http://192.168.1.100/relay?pin=12&state=off", timeout=2)
    mock_get.assert_any_call("http://192.168.1.100/relay?pin=13&state=off", timeout=2)
