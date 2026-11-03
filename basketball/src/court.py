"""The court from above, and each shot's spot on it, estimated from one camera.

A pinhole camera: the rim's width and the shooter's height in pixels give their
depths, their place in the frame their sideways offsets.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SHOULDER_TO_ANKLE = 0.78   # of standing height

# Half court in meters, hoop at the origin, +y toward half court. Only the lines that
# matter here: sidelines, the three, and the paint up to the free-throw line.
BASELINE_Y = -1.25
BACKBOARD_Y = -0.375
BACKBOARD_HALF_WIDTH = 0.9
HALF_COURT_Y = 14.0 + BASELINE_Y
COURT_HALF_WIDTH = 7.5
CORNER_THREE_X = 6.6
LANE_HALF_WIDTH = CORNER_THREE_X / 3   # the paint is the middle third of the three's width


def project(feet_u: float, body_px: float, hoop, *, frame_width: int, hfov_degrees: float,
            player_height_m: float, rim_diameter_m: float, yaw_degrees: float
            ) -> tuple[float, float]:
    """The shooter's spot, ``(x, y)`` meters from the hoop (x right as they face it)."""
    f = (frame_width / 2) / math.tan(math.radians(hfov_degrees) / 2)
    cx = frame_width / 2
    z_hoop = f * rim_diameter_m / hoop.width
    z_player = f * SHOULDER_TO_ANKLE * player_height_m / body_px
    x_hoop = (hoop.cx - cx) * z_hoop / f
    x_player = (feet_u - cx) * z_player / f
    x, y = x_player - x_hoop, z_hoop - z_player
    t = math.radians(yaw_degrees)
    return x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)


class CourtView:
    """A half court from above, at least *min_depth* meters deep (narrower and centered if needed)."""

    def __init__(self, width: int, height: int, *, three_radius: float, ft_distance: float,
                 min_depth: float = 0.0, line=(235, 235, 235), floor=(0, 0, 0),
                 paint=(34, 34, 34)):
        self.w, self.h = width, height
        span_x = 2 * COURT_HALF_WIDTH + 0.6
        depth_m = min(HALF_COURT_Y, max(min_depth, height / (width / span_x) + BASELINE_Y - 0.3))
        self.depth_m = depth_m
        span_y = depth_m - BASELINE_Y
        self.scale = min(width / span_x, height / span_y)
        self.ox = width / 2
        self.oy = (height - span_y * self.scale) / 2 - BASELINE_Y * self.scale
        self.three_radius = three_radius
        img = np.zeros((height, width, 3), np.uint8)
        img[:] = floor
        self.base = img
        self._draw(img, three_radius, ft_distance, depth_m, line, floor, paint)

    def px(self, x: float, y: float) -> tuple[int, int]:
        """Court meters to pixels; the baseline is at the top."""
        return int(round(self.ox + x * self.scale)), int(round(self.oy + y * self.scale))

    def _draw(self, img, r3, ft, depth, line, floor, paint) -> None:
        s = self.scale
        t = max(1, int(round(s * 0.06)))
        a = self.px(-COURT_HALF_WIDTH, BASELINE_Y)
        b = self.px(COURT_HALF_WIDTH, depth)
        cv2.rectangle(img, a, b, floor, -1)
        aa = cv2.LINE_AA
        # Paint, then the lines over it.
        lane = (self.px(-LANE_HALF_WIDTH, BASELINE_Y), self.px(LANE_HALF_WIDTH, ft))
        cv2.rectangle(img, *lane, paint, -1)
        cv2.rectangle(img, a, b, line, t, aa)
        cv2.rectangle(img, *lane, line, t, aa)
        # The three: straight corners until they meet the arc.
        corner_y = math.sqrt(max(0.0, r3 ** 2 - CORNER_THREE_X ** 2))
        for sign in (-1, 1):
            cv2.line(img, self.px(sign * CORNER_THREE_X, BASELINE_Y),
                     self.px(sign * CORNER_THREE_X, corner_y), line, t, aa)
        start = math.degrees(math.atan2(corner_y, CORNER_THREE_X))
        cv2.ellipse(img, self.px(0, 0), (int(r3 * s),) * 2, 0, start, 180 - start, line, t, aa)
        # Backboard and rim; the rim is a third of the board's width, as on an outdoor board.
        cv2.line(img, self.px(-BACKBOARD_HALF_WIDTH, BACKBOARD_Y),
                 self.px(BACKBOARD_HALF_WIDTH, BACKBOARD_Y), (255, 255, 255), max(2, t + 1), aa)
        cv2.circle(img, self.px(0, 0), max(3, int(BACKBOARD_HALF_WIDTH / 3 * s)), (44, 98, 210),
                   max(2, t), aa)

    def corner(self) -> tuple[int, int]:
        """The bottom-right corner of the drawn court, in pixels."""
        return self.px(COURT_HALF_WIDTH, self.depth_m)

    def is_three(self, x: float, y: float) -> bool:
        if abs(x) >= CORNER_THREE_X and y <= math.sqrt(max(0.0, self.three_radius ** 2 - CORNER_THREE_X ** 2)):
            return True
        return math.hypot(x, y) > self.three_radius

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Keep a dot on the drawn court, in case the guess lands off it."""
        return (float(np.clip(x, -COURT_HALF_WIDTH + 0.2, COURT_HALF_WIDTH - 0.2)),
                float(np.clip(y, BASELINE_Y + 0.2, 1e9)))
