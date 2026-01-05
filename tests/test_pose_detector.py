import os
import pytest
from utils.pose_detector import PoseDetector


def test_pose_detector_requires_model():
    # PoseDetector now requires a `.task` model by default; if missing it should raise a clear error
    # Use a temporary non-existent path to trigger the error
    fake_path = os.path.join('nonexistent_models', 'missing.task')
    with pytest.raises(RuntimeError) as exc:
        PoseDetector(model_path=fake_path)
    assert 'not found' in str(exc.value).lower() or 'download' in str(exc.value).lower()
