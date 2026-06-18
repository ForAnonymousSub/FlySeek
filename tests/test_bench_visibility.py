# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for the FlySeek-Bench VisibilityEvaluator (geometry + fallbacks)."""

from __future__ import annotations

import warnings

import numpy as np

from flyseek.bench.schema import CameraConfig
from flyseek.bench.visibility import VisibilityEvaluator, evaluate_frame


class _StubOccupancy:
    """Minimal occupancy stub exposing los_blocked_ned."""

    def __init__(self, blocked: bool):
        self._blocked = blocked

    def los_blocked_ned(self, observer, target, *, drone_eye_agl_m, target_agl_m):
        return self._blocked


def _cam(width=256, height=144):
    return CameraConfig(hfov_deg=90.0, pitch_deg=55.0, width=width, height=height)


def test_frustum_projection_in_view():
    ev = VisibilityEvaluator(max_range_m=100.0)
    # Target ahead and below the downward-pitched camera -> in frustum.
    out = ev.evaluate_frame(
        uav_pose=[0.0, 0.0, -18.0, 0.0],
        target_pose=[10.0, 0.0, -0.3],
        camera_config=_cam(),
        scene_context={"occupancy": _StubOccupancy(False)},
    )
    assert out["in_camera_frustum"] is True
    assert out["line_of_sight_clear"] is True
    assert out["target_visible"] is True
    assert 0.0 < out["visibility_score"] <= 1.0
    assert "projection" in out["visibility_source"]


def test_los_blocked_marks_not_clear():
    ev = VisibilityEvaluator()
    out = ev.evaluate_frame(
        uav_pose=[0.0, 0.0, -18.0, 0.0],
        target_pose=[10.0, 0.0, -0.3],
        camera_config=_cam(),
        scene_context={"occupancy": _StubOccupancy(True)},
    )
    assert out["line_of_sight_clear"] is False
    # No recorded judgment -> geometry decides: frustum True but LoS blocked.
    assert out["target_visible"] is False


def test_recorded_judgment_is_authoritative():
    ev = VisibilityEvaluator()
    out = ev.evaluate_frame(
        uav_pose=[0.0, 0.0, -18.0, 0.0],
        target_pose=[10.0, 0.0, -0.3],
        camera_config=_cam(),
        scene_context={"occupancy": _StubOccupancy(True)},
        existing_visibility_metadata={"target_visible": True, "vis_reason": "ok"},
    )
    # Even though LoS stub says blocked, recorded judgment wins for visibility.
    assert out["target_visible"] is True
    assert out["line_of_sight_clear"] is False  # geometry still reported


def test_reason_derived_los_without_occupancy():
    ev = VisibilityEvaluator()
    out = ev.evaluate_frame(
        uav_pose=[0.0, 0.0, -18.0, 0.0],
        target_pose=[10.0, 0.0, -0.3],
        camera_config=_cam(),
        scene_context={},  # no occupancy
        existing_visibility_metadata={"target_visible": False,
                                      "vis_reason": "los_blocked"},
    )
    assert out["line_of_sight_clear"] is False
    assert "recorded_reason" in out["visibility_source"]


def test_fallback_binary_score_and_warns():
    ev = VisibilityEvaluator()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = ev.evaluate_frame(
            uav_pose={"x": 0, "y": 0, "z": -18},   # no yaw -> no frustum
            target_pose={"x": 10, "y": 0, "z": -0.3},
            camera_config={"hfov_deg": 90.0},       # no width/height
            scene_context={},
            existing_visibility_metadata={"target_visible": True},  # no reason
        )
    assert out["target_visible"] is True
    assert out["visibility_score"] == 1.0           # binary fallback
    assert out["in_camera_frustum"] is None
    assert out["line_of_sight_clear"] is None


def test_occlusion_risk_none_without_future():
    ev = VisibilityEvaluator()
    risk = ev.estimate_occlusion_risk(np.array([0.0, 0.0, -18.0]), {}, {})
    assert risk is None


def test_occlusion_risk_from_future_positions():
    ev = VisibilityEvaluator()
    ctx = {
        "occupancy": _StubOccupancy(True),
        "future_target_positions": [[5, 0, -0.3], [6, 0, -0.3], [7, 0, -0.3]],
    }
    risk = ev.estimate_occlusion_risk(np.array([0.0, 0.0, -18.0]), ctx, {})
    assert risk == 1.0  # stub blocks all


def test_occlusion_risk_passthrough_existing():
    ev = VisibilityEvaluator()
    risk = ev.estimate_occlusion_risk(
        np.array([0.0, 0.0, -18.0]), {}, {"occlusion_risk": 0.42}
    )
    assert risk == 0.42


def test_module_level_wrapper():
    out = evaluate_frame(
        [0.0, 0.0, -18.0, 0.0], [10.0, 0.0, -0.3], _cam(),
        {"occupancy": _StubOccupancy(False)},
    )
    assert set(out) >= {"in_camera_frustum", "line_of_sight_clear",
                        "target_visible", "visibility_score", "occlusion_risk"}
