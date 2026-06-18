# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for the FlySeek-Bench schema/export layer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flyseek.bench import (
    CameraConfig,
    EpisodeMetadata,
    FrameMetadata,
    InstructionRecord,
    MetricRecord,
    SchemaValidationError,
    TrajectoryRecord,
    append_frame_jsonl,
    save_instruction_json,
    save_metadata_json,
    save_metrics_json,
    save_trajectories_json,
    validate_episode,
    validate_frame,
)


def _episode(**overrides) -> EpisodeMetadata:
    base = dict(
        episode_id="ep_0001",
        scene_id="env_airsim_16",
        difficulty_level="medium",
        target_behavior_type="zigzag",
        target_class="vehicle.car",
        instruction="Track the small red car that is weaving back and forth.",
        random_seed=42,
        max_steps=800,
        camera_config=CameraConfig(width=256, height=144),
        uav_initial_pose=[0.0, 0.0, -18.0, 0.0],
        target_initial_pose={"x": 8.0, "y": 0.0, "z": -0.3, "yaw": 0.0},
        environment_summary={"env": "env_airsim_16", "n_objects": 35528},
        occluder_summary={"buildings": 12},
    )
    base.update(overrides)
    return EpisodeMetadata(**base)


def _frame(**overrides) -> FrameMetadata:
    base = dict(
        frame_id=0,
        image_path="image_0.png",
        step_index=0,
        timestamp=0.0,
        uav_pose=[0.0, 0.0, -18.0, 0.0],
        target_pose=np.array([8.0, 0.0, -0.3]),
        uav_velocity=[0.0, 0.0, 0.0],
        target_velocity=[1.0, 0.0, 0.0],
        target_visible=True,
        in_camera_frustum=True,
        line_of_sight_clear=True,
        visibility_score=0.9,
        distance_to_target=8.0,
        relative_bearing=0.0,
        occlusion_risk=0.1,
        selected_viewpoint=[-2.0, 0.0, -18.0],
        collision=False,
        target_behavior_type="zigzag",
        difficulty_level="medium",
    )
    base.update(overrides)
    return FrameMetadata(**base)


def test_episode_validates_and_round_trips(tmp_path):
    ep = _episode()
    ep.validate()
    path = save_metadata_json(ep, tmp_path / "metadata.json")
    data = json.loads(path.read_text())
    assert data["episode_id"] == "ep_0001"
    assert data["camera_config"]["width"] == 256
    assert data["schema_version"] == "1.0"


def test_episode_missing_field_rejected():
    with pytest.raises(SchemaValidationError):
        validate_episode({"episode_id": "x"})


def test_episode_random_seed_may_be_none():
    ep = _episode(random_seed=None)
    ep.validate()  # should not raise


def test_frame_validates_and_appends(tmp_path):
    f0 = _frame(frame_id=0)
    f1 = _frame(frame_id=1, image_path="image_1.png", target_visible=False)
    p = append_frame_jsonl(f0, tmp_path / "frames.jsonl")
    append_frame_jsonl(f1, p)
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[1])
    assert rec1["target_visible"] is False
    # numpy target_pose coerced to a list
    assert isinstance(json.loads(lines[0])["target_pose"], list)


def test_visibility_score_out_of_range_rejected():
    with pytest.raises(SchemaValidationError):
        validate_frame(_frame(visibility_score=1.5).to_dict())


def test_visibility_score_none_allowed():
    validate_frame(_frame(visibility_score=None, occlusion_risk=None).to_dict())


def test_target_visible_must_be_bool():
    with pytest.raises(SchemaValidationError):
        validate_frame(_frame(target_visible=1).to_dict())


def test_image_path_must_be_nonempty_string():
    with pytest.raises(SchemaValidationError):
        validate_frame(_frame(image_path="").to_dict())


def test_instruction_trajectory_metric_round_trip(tmp_path):
    instr = InstructionRecord(
        episode_id="ep_0001",
        instruction="Track the small red car.",
        instructions=["Track the small red car.", "Follow the small red car."],
        tier="medium",
    )
    save_instruction_json(instr, tmp_path / "instruction.json")

    traj = TrajectoryRecord(episode_id="ep_0001", difficulty_level="medium")
    traj.add_frame(step_index=0, uav_pose=[0, 0, -18, 0], target_pose=[8, 0, -0.3],
                   target_visible=True)
    save_trajectories_json(traj, tmp_path / "trajectories.json")
    tdata = json.loads((tmp_path / "trajectories.json").read_text())
    assert len(tdata["frames"]) == 1

    metric = MetricRecord(
        episode_id="ep_0001", track_auc=0.8, lost_rate=0.2,
        redetection_time_s=0.45, collision_rate=0.0, fov_keep_rate=0.8,
        frames=40, duration_s=2.0,
    )
    save_metrics_json(metric, tmp_path / "metrics.json")
    mdata = json.loads((tmp_path / "metrics.json").read_text())
    assert mdata["track_auc"] == 0.8


def test_instruction_requires_episode_id():
    with pytest.raises(SchemaValidationError):
        save_instruction_json({"instruction": "x"}, "/tmp/_should_not_write.json")
