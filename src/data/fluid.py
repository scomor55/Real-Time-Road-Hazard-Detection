"""Helpers for loading and inspecting the FLUID dataset.

FLUID: A Fine-Grained Lightweight Urban Signalized-Intersection Dataset of
Dense Conflict Trajectories (Chen et al., 2025, CC BY 4.0).
https://doi.org/10.6084/m9.figshare.29974954
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Road-user classes grouped into the three categories used in the project.
CLASS_GROUPS = {
    "car": "motor vehicle",
    "van": "motor vehicle",
    "truck": "motor vehicle",
    "bus": "motor vehicle",
    "trailer": "motor vehicle",
    "moped": "two/three-wheeler",
    "tricycle": "two/three-wheeler",
    "pedestrian": "pedestrian",
}

VULNERABLE_CLASSES = {"pedestrian", "moped", "tricycle"}


def find_files(data_dir: Path, pattern: str) -> list[Path]:
    """Return all files under ``data_dir`` matching ``pattern``, sorted by name."""
    return sorted(Path(data_dir).rglob(pattern))


def recording_name(path: Path) -> str:
    """'20250529_video-1_Traj.csv' -> '20250529_video-1'."""
    return path.stem.rsplit("_", 1)[0]


def load_trajectories(paths: list[Path]) -> pd.DataFrame:
    """Load one or more *_Traj.csv files into a single DataFrame.

    Adds a ``recording`` column and a globally unique ``track_uid``,
    because track ids restart from 1 in every recording.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["recording"] = recording_name(path)
        frames.append(df)
    traj = pd.concat(frames, ignore_index=True)
    traj["track_uid"] = traj["recording"] + "/" + traj["id"].astype(str)
    traj["group"] = traj["type"].map(CLASS_GROUPS).fillna("other")
    return traj


def track_summary(traj: pd.DataFrame) -> pd.DataFrame:
    """One row per track: class, duration, displacement and speed statistics."""
    g = traj.sort_values("frame").groupby("track_uid")
    first, last = g.first(), g.last()
    summary = pd.DataFrame(
        {
            "recording": first["recording"],
            "type": first["type"],
            "group": first["group"],
            "n_rows": g.size(),
            "duration_s": last["time"] - first["time"],
            "displacement_m": np.hypot(
                last["cx_m"] - first["cx_m"], last["cy_m"] - first["cy_m"]
            ),
            "interpolated_share": 1 - g["isReal"].mean(),
            "mean_confidence": g["confidence"].mean(),
            "max_speed_smooth": g["speed_smooth"].max(),
            "entry": first["entry_direction"],
            "exit": first["exit_direction"],
        }
    )
    return summary


def fit_homography(px: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Fit a 3x3 homography H mapping pixel points to ground-plane points.

    Uses the normalised Direct Linear Transform (Hartley & Zisserman).
    ``px`` and ``world`` are arrays of shape (N, 2), N >= 4.
    """

    def normaliser(pts: np.ndarray) -> np.ndarray:
        centre = pts.mean(axis=0)
        scale = np.sqrt(2) / np.mean(np.linalg.norm(pts - centre, axis=1))
        return np.array(
            [[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]]
        )

    t_px, t_w = normaliser(px), normaliser(world)
    p = (t_px @ np.column_stack([px, np.ones(len(px))]).T).T
    q = (t_w @ np.column_stack([world, np.ones(len(world))]).T).T

    a, b = p[:, 0], p[:, 1]
    c, d = q[:, 0], q[:, 1]
    zeros, ones = np.zeros_like(a), np.ones_like(a)
    rows_x = np.column_stack([a, b, ones, zeros, zeros, zeros, -c * a, -c * b, -c])
    rows_y = np.column_stack([zeros, zeros, zeros, a, b, ones, -d * a, -d * b, -d])
    _, _, vt = np.linalg.svd(np.vstack([rows_x, rows_y]), full_matrices=False)

    h = np.linalg.inv(t_w) @ vt[-1].reshape(3, 3) @ t_px
    return h / h[2, 2]


def apply_homography(h: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Map pixel points (N, 2) to ground-plane points (N, 2) with homography ``h``."""
    pts = h @ np.column_stack([px, np.ones(len(px))]).T
    return (pts[:2] / pts[2]).T


# Codes used in the *_Route.csv files (inferred from the data).
TURN_NAMES = {"s": "straight", "l": "left", "r": "right", "t": "U-turn"}
SIGNAL_NAMES = {"G": "green", "y": "yellow", "r": "red"}


def load_routes(paths: list[Path]) -> pd.DataFrame:
    """Load *_Route.csv files (one row per motor-vehicle track).

    The WKT ``geometry`` column (full path as a line string) is dropped
    because the same positions are already in the trajectory files.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path).drop(columns="geometry", errors="ignore")
        df["recording"] = recording_name(path)
        frames.append(df)
    routes = pd.concat(frames, ignore_index=True)
    routes["track_uid"] = routes["recording"] + "/" + routes["id"].astype(str)
    routes["turn_name"] = routes["turn"].map(TURN_NAMES)
    routes["entry_signal"] = routes["in_state"].map(SIGNAL_NAMES)
    routes["exit_signal"] = routes["out_state"].map(SIGNAL_NAMES)
    return routes


FPS = 10  # FLUID videos and trajectories are sampled at 10 frames per second


def load_conflicts(paths: list[Path], fps: float = FPS) -> pd.DataFrame:
    """Load *_conflict.csv files (one row per annotated vehicle-vehicle conflict).

    Columns are renamed to snake_case; ``recording`` and ``time`` (s) are added.
    FLUID defines a conflict as 0 s <= TTC <= 2 s and 0 s <= DGT <= 4 s.
    """
    conflicts = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    conflicts = conflicts.rename(
        columns={
            "Scene": "scene", "Frame": "frame",
            "Car 1 ID": "id_1", "Car 2 ID": "id_2",
            "Car 1 X": "x_1", "Car 1 Y": "y_1", "Car 2 X": "x_2", "Car 2 Y": "y_2",
            "diff": "angle_diff_deg", "TTC": "ttc", "MTTC": "mttc",
            "DGT/PET": "dgt", "conflict_type": "conflict_type",
        }
    )
    conflicts["recording"] = conflicts["scene"].str.replace("_Traj", "", regex=False)
    conflicts["time"] = conflicts["frame"] / fps
    return conflicts


def load_signals(paths: list[Path]) -> pd.DataFrame:
    """Load *_signal.csv files: one row per (movement, state) interval.

    ``turn`` codes: s straight, l left, p pedestrian crossing.
    Exact duplicate rows present in the raw files are removed.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path).drop_duplicates()
        df["recording"] = recording_name(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def green_intervals(signals: pd.DataFrame, direction: str, turn: str) -> pd.DataFrame:
    """Green intervals (recording, begin_time, end_time) of one signal-controlled movement."""
    mask = (signals["direction"] == direction) & (signals["turn"] == turn) & (signals["state"] == "G")
    return signals.loc[mask, ["recording", "begin_time", "end_time"]].sort_values(["recording", "begin_time"])


def phase_at(time: float, phases: dict[str, pd.DataFrame]) -> tuple[str, float]:
    """Return (phase name, seconds since that phase's green began) at ``time``.

    ``phases`` maps a phase name to its green intervals for one recording.
    Times outside every green interval are reported as ('yellow / all-red', nan).
    """
    for name, intervals in phases.items():
        hit = intervals[(intervals["begin_time"] <= time) & (time < intervals["end_time"])]
        if len(hit):
            return name, time - hit["begin_time"].iloc[0]
    return "yellow / all-red", np.nan
