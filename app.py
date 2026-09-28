"""
SAR-1 "PATHFINDER" — Humanoid Search & Rescue Operator Console
===============================================================
A live, self-running Streamlit simulator + operator 3D viewer for a bipedal
humanoid explorer operating inside collapsed structures.

Hardware model : head stereo/depth cameras, LiDAR, thermal camera, 4-ch mic array
SLAM           : live 3D voxel map grown by ray-cast scanning of unstable interiors
Footstep plan  : A* over rubble / gaps / slopes with stability scoring; auto re-plan
                 when a floor-shift event re-scores terrain
Survivor det.  : fusion of thermal blobs + audio (tapping/voice) + pose detection,
                 ranked likelihood markers with confidence
Comms          : degraded-link policy -> autonomous continue, drop relay beacons,
                 return-to-last-link
Other          : fall-recovery routines, battery-aware frontier exploration,
                 post-mission PLY/CSV/JSON export for structural engineers
Safety rules   : NEVER enter sensor-flagged UNSTABLE zones without operator approval;
                 EVERY autonomous decision is logged
"""

import copy
import io
import json
import math
import os
import random
import tempfile
import time
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAVE_CLICK = True
except Exception:
    HAVE_CLICK = False

st.set_page_config(page_title="SAR-1 PATHFINDER — Operator Console",
                   page_icon="🤖", layout="wide")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
N = 64                      # grid cells per side
CELL = 0.5                  # metres per cell
WALL, FLOOR, RUBBLE, GAP, SLOPE = 0, 1, 2, 3, 4
TERRAIN_NAMES = {WALL: "wall", FLOOR: "floor", RUBBLE: "rubble",
                 GAP: "gap/void", SLOPE: "slope"}
TERRAIN_RGB = {WALL: (108, 116, 130), FLOOR: (58, 110, 165),
               RUBBLE: (176, 122, 60), GAP: (18, 20, 28), SLOPE: (139, 90, 160)}
MOVE_COST = {FLOOR: 1.0, RUBBLE: 3.0, SLOPE: 2.2}
SPEED_CPS = {FLOOR: 2.0, RUBBLE: 0.85, SLOPE: 1.1}   # cells / second


# ----------------------------------------------------------------------------
# Pathfinding (footstep planner)
# ----------------------------------------------------------------------------
def astar(terra, stab, red, allow_red, start, goal):
    import heapq
    sx, sy = start
    gx, gy = goal
    if terra[gy, gx] in (WALL, GAP):
        return None, 0.0

    def passable(x, y):
        t = terra[y, x]
        if t in (WALL, GAP):
            return False
        if red[y, x] and not allow_red:
            return False
        return True

    if not passable(gx, gy):
        return None, 0.0
    if red[sy, sx]:          # robot already inside flagged zone -> allow exit
        allow_red = True

    open_h = [(0.0, 0.0, sx, sy)]
    came, gscore, weakest, closed = {}, {(sx, sy): 0.0}, {(sx, sy): 1.0}, set()
    h = lambda x, y: math.hypot(x - gx, y - gy)

    while open_h:
        f, g, x, y = heapq.heappop(open_h)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while path[-1] != (sx, sy):
                path.append(came[path[-1]])
            path.reverse()
            return path, weakest[(gx, gy)]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < N and 0 <= ny < N) or not passable(nx, ny):
                    continue
                if dx and dy and (not passable(x + dx, y) or not passable(x, y + dy)):
                    continue
                t = terra[ny, nx]
                cost = MOVE_COST.get(t, 2.0) * (1.0 + max(0.0, 0.85 - stab[ny, nx]) * 2.5)
                if dx and dy:
                    cost *= 1.4142
                ng = g + cost
                if ng < gscore.get((nx, ny), 1e18):
                    gscore[(nx, ny)] = ng
                    weakest[(nx, ny)] = min(weakest[(x, y)], float(stab[ny, nx]))
                    came[(nx, ny)] = (x, y)
                    heapq.heappush(open_h, (ng + h(nx, ny), ng, nx, ny))
    return None, 0.0


# ----------------------------------------------------------------------------
# Simulation core
# ----------------------------------------------------------------------------
class Sim:
    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.decisions = deque(maxlen=1500)
        self.build_world()
        self.place_survivors()

        # robot state
        self.pos = np.array(self.base, float)
        self.heading = 0.0
        self.battery = 100.0
        self.signal = 100.0
        self.beacons_left = 4
        self.beacons = []                     # (pos[x,y], link_quality)
        self.mode_comms = "LINK_OK"
        self.last_link_pos = self.pos.copy()
        self.lost_timer = 0.0
        self.state = "OPERATIONAL"
        self.path, self.path_weak = [], 1.0
        self.goal, self.goal_source, self.goal_red_ok = None, None, False
        self.pending_goal = None
        self.returning = False
        self.auto_explore = True
        self.red_override = False
        self.falls, self.recover_t = 0, 0.0
        self.distance, self._last_rec = 0.0, 0.0
        self.trail = [self.pos.copy()]
        self.warn35 = self.warn20 = self.warn10 = False
        self._next_fr, self._next_shift = 3.0, 18.0
        self.mission_over = False
        self.known = np.zeros((N, N), bool)

        self.log("SYSTEM", "SAR-1 boot OK — stereo/depth ✓ LiDAR ✓ thermal ✓ mic-array ✓", "autonomous")
        self.log("SYSTEM", f"Deployed at breach point ({self.base[0]},{self.base[1]}) — SLAM initialised", "autonomous")
        self.scan()

    # ---------------- world ----------------
    def build_world(self):
        rng = self.rng
        self.terra = np.full((N, N), WALL, np.int8)
        self.height = np.zeros((N, N), float)
        self.stab = np.full((N, N), 0.96, float)

        rooms = []
        for _ in range(8):
            w, h = int(rng.integers(8, 15)), int(rng.integers(8, 14))
            x0, y0 = int(rng.integers(1, N - w - 1)), int(rng.integers(1, N - h - 1))
            rooms.append((x0, y0, w, h))
            self.terra[y0:y0 + h, x0:x0 + w] = FLOOR
        for (x0, y0, w, h), (x1, y1, w1, h1) in zip(rooms, rooms[1:]):
            cx0, cy0 = x0 + w // 2, y0 + h // 2
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            self.terra[cy0:min(N, cy0 + 2), min(cx0, cx1):max(cx0, cx1) + 1] = FLOOR
            self.terra[min(cy0, cy1):max(cy0, cy1) + 1, cx1:min(N, cx1 + 2)] = FLOOR
        self.base = (rooms[0][0] + rooms[0][2] // 2, rooms[0][1] + rooms[0][3] // 2)

        yy, xx = np.ogrid[:N, :N]
        floor_now = self.terra == FLOOR
        for _ in range(22):                                    # rubble fields
            cx, cy = int(rng.integers(3, N - 3)), int(rng.integers(3, N - 3))
            r = int(rng.integers(2, 5))
            m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & floor_now
            if m.any():
                self.terra[m] = RUBBLE
                self.height[m] = 0.2 + rng.random(int(m.sum())) * 0.3
        for _ in range(5):                                     # debris slopes / ramps
            x0, y0 = int(rng.integers(2, N - 9)), int(rng.integers(2, N - 9))
            w, h = int(rng.integers(4, 8)), int(rng.integers(4, 8))
            reg = self.terra[y0:y0 + h, x0:x0 + w]
            mask = (reg == FLOOR) | (reg == RUBBLE)
            if mask.sum() < 6:
                continue
            reg[mask] = SLOPE
            ramp = np.tile(np.linspace(0.0, 1.15, w), (h, 1))
            sub = self.height[y0:y0 + h, x0:x0 + w].copy()
            sub[mask] = ramp[mask]
            self.height[y0:y0 + h, x0:x0 + w] = sub

        open_m0 = (self.terra != WALL)
        for _ in range(8):                                     # floor gaps / voids
            cx, cy = int(rng.integers(3, N - 3)), int(rng.integers(3, N - 3))
            if math.hypot(cx - self.base[0], cy - self.base[1]) < 7:
                continue
            r = int(rng.integers(1, 3))
            m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & open_m0
            self.terra[m] = GAP
            self.height[m] = -1.7

        m_rub = self.terra == RUBBLE
        self.stab[m_rub] = 0.35 + rng.random(int(m_rub.sum())) * 0.30
        m_sl = self.terra == SLOPE
        self.stab[m_sl] = 0.50 + rng.random(int(m_sl.sum())) * 0.25

        # structural-sensor flagged UNSAFE zones
        open_m = (self.terra != WALL) & (self.terra != GAP)
        self.red = np.zeros((N, N), bool)
        self.red_zones = []
        oc = np.argwhere(open_m)
        tries = 0
        while len(self.red_zones) < 3 and tries < 400:
            tries += 1
            y, x = oc[int(rng.integers(len(oc)))]
            if math.hypot(x - self.base[0], y - self.base[1]) < 12:
                continue
            if any(math.hypot(x - zx, y - zy) < 12 for zy, zx in self.red_zones):
                continue
            r = int(rng.integers(3, 6))
            self.red |= ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) & open_m
            self.red_zones.append((int(y), int(x)))
        self.stab[self.red & open_m] = np.minimum(self.stab[self.red & open_m], 0.30)

        bx, by = self.base                                     # keep breach clear
        sl = (slice(max(0, by - 1), min(N, by + 2)), slice(max(0, bx - 1), min(N, bx + 2)))
        self.terra[sl] = FLOOR
        self.red[sl] = False
        self.height[sl] = 0.0
        self.stab[sl] = 0.97

    def place_survivors(self):
        open_m = (self.terra != WALL) & (self.terra != GAP)
        by, bx = self.base[1], self.base[0]
        yy, xx = np.where(open_m)
        d = np.hypot(xx - bx, yy - by)
        cand = [(int(y), int(x)) for y, x, dd in zip(yy, xx, d) if dd > 14]
        self.rng.shuffle(cand)
        chosen = []
        for (y, x) in cand:
            if all(math.hypot(x - cx, y - cy) > 12 for (cy, cx) in chosen):
                chosen.append((y, x))
            if len(chosen) == 3:
                break
        audios = [0.95, 0.55, 0.0]
        self.rng.shuffle(audios)
        self.survivors = []
        for i, ((y, x), au) in enumerate(zip(chosen, audios)):
            self.survivors.append(dict(
                id=i + 1, pos=np.array([x, y], float),
                thermal=float(self.rng.uniform(0.65, 1.0)), audio=au,
                visual=float(self.rng.uniform(0.35, 0.95)),
                evid=np.zeros(3), conf=0.0, detected=False))

    # ---------------- logging ----------------
    def log(self, subsys, decision, level="autonomous", detail=""):
        self.decisions.append(dict(t=round(self.t, 1), subsystem=subsys,
                                   decision=decision, level=level, detail=detail))

    # ---------------- SLAM ----------------
    def scan(self):
        px, py = int(round(self.pos[0])), int(round(self.pos[1]))
        self.known[py, px] = True
        for a in range(0, 360, 2):
            rad = math.radians(a)
            dx, dy = math.cos(rad), math.sin(rad)
            x, y = px + 0.5, py + 0.5
            for _ in range(11):
                x += dx * 0.9
                y += dy * 0.9
                cx, cy = int(x), int(y)
                if not (0 <= cx < N and 0 <= cy < N):
                    break
                self.known[cy, cx] = True
                if self.terra[cy, cx] == WALL:
                    break

    # ---------------- comms ----------------
    def link_quality(self, a, b):
        d = float(np.linalg.norm(a - b)) * CELL
        steps = max(1, int(d / (CELL * 0.5)))
        walls = 0
        for i in range(1, steps):
            pt = a + (b - a) * (i / steps)
            if self.terra[int(round(pt[1])), int(round(pt[0]))] == WALL:
                walls += 1
        return 108.0 - d * 2.4 - walls * 9.0

    def compute_signal(self):
        s = self.link_quality(self.pos, np.array(self.base, float))
        for bp, bl in self.beacons:
            s = max(s, bl - float(np.linalg.norm(self.pos - bp)) * CELL * 2.4)
        return float(np.clip(s, 0.0, 100.0))

    def update_comms(self, dt):
        sig = self.compute_signal()
        self.signal = sig
        if sig > 45:
            self.last_link_pos = self.pos.copy()
        if self.mode_comms == "LINK_OK" and sig < 60:
            self.mode_comms = "DEGRADED"
            self.log("COMMS", f"Link degraded ({sig:.0f}%) — policy A-2: continue mission autonomously", "autonomous")
        elif self.mode_comms == "DEGRADED" and sig > 70:
            self.mode_comms = "LINK_OK"
            self.log("COMMS", f"Link restored ({sig:.0f}%)", "autonomous")
        if sig < 25 and self.mode_comms != "LOST":
            if self.beacons_left > 0:
                self.beacons_left -= 1
                self.beacons.append((self.pos.copy(), min(100.0, sig + 55.0)))
                self.log("COMMS", f"Signal {sig:.0f}% — dropped relay beacon #{4 - self.beacons_left} "
                                  f"at ({self.pos[0]:.1f},{self.pos[1]:.1f})", "autonomous")
            else:
                self.mode_comms = "LOST"
                self.lost_timer = 0.0
                self.log("COMMS", "Signal <25%, no relays left — comms LOST, return-to-last-link armed (8 s grace)", "autonomous")
        if self.mode_comms == "LOST":
            self.lost_timer += dt
            if sig > 35:
                self.mode_comms = "DEGRADED"
                self.log("COMMS", "Link re-acquired — resuming degraded ops", "autonomous")
            elif self.lost_timer > 8 and not self.returning:
                self.cmd_return("comms lost >8 s — returning to last-link position", "autonomous", to_base=False)

    # ---------------- survivors ----------------
    def update_survivors(self, dt):
        for s in self.survivors:
            d = float(np.linalg.norm(s["pos"] - self.pos)) * CELL
            th_on = d < 6.0
            au_on = d < 8.5 and s["audio"] > 0
            vi_on = d < 4.5 and s["visual"] > 0.25
            rates = np.array([0.22 if th_on else 0.0,
                              0.18 * s["audio"] if au_on else 0.0,
                              0.30 * s["visual"] if vi_on else 0.0])
            s["evid"] = s["evid"] + rates * dt * (1.0 - s["evid"])
            active = int(th_on) + int(au_on) + int(vi_on)
            if active:
                w = np.array([0.50, 0.38, 0.50])
                conf = 1.0 - float(np.prod(1.0 - w * s["evid"]))
                conf = min(0.99, conf + 0.05 * (active - 1))
                if not s["detected"] and conf > 0.22:
                    s["detected"] = True
                    names = [n for n, on in zip(["thermal blob", "audio tap/voice", "visual pose"],
                                                [th_on, au_on, vi_on]) if on]
                    self.log("PERCEPTION", f"Survivor S{s['id']} detected via {' + '.join(names)} — likelihood {conf:.0%}", "autonomous")
                s["conf"] = max(s["conf"], conf)

    # ---------------- terrain events ----------------
    def floor_shift(self):
        open_m = (self.terra != WALL) & (self.terra != GAP)
        cand = np.argwhere(self.known & open_m)
        if len(cand) == 0:
            return
        y, x = cand[int(self.rng.integers(len(cand)))]
        r = int(self.rng.integers(2, 4))
        yy, xx = np.ogrid[:N, :N]
        m = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) & open_m
        self.stab[m] = np.clip(self.stab[m] + self.rng.uniform(-0.18, -0.05), 0.08, 0.97)
        if self.rng.random() < 0.3:
            mm = m & (self.terra == RUBBLE)
            if mm.any():
                self.terra[mm] = GAP
                self.height[mm] = -1.7
        self.log("NAV", f"Floor shift near ({x},{y}) — terrain re-scored, footstep plan invalidated", "autonomous")
        self.path = []

    # ---------------- goals & safety ----------------
    def request_goal(self, gx, gy, source="operator"):
        if not (0 <= gx < N and 0 <= gy < N):
            return "invalid"
        if not self.known[gy, gx]:
            self.log("NAV", f"Goal rejected: cell ({gx},{gy}) not yet mapped by SLAM", "autonomous")
            return "unmapped"
        t = self.terra[gy, gx]
        if t in (WALL, GAP):
            self.log("NAV", f"Goal rejected: ({gx},{gy}) is {TERRAIN_NAMES[t]} — no foothold possible", "autonomous")
            return "blocked"
        if self.red[gy, gx] and not self.red_override:
            self.pending_goal = (gx, gy)
            self.state = "WAITING_APPROVAL"
            self.log("SAFETY", f"Goal ({gx},{gy}) inside structurally UNSAFE zone — HOLDING for operator approval", "autonomous")
            return "approval"
        self.set_goal((gx, gy), source, red_ok=bool(self.red_override and self.red[gy, gx]))
        return "ok"

    def set_goal(self, goal, source="operator", red_ok=False):
        self.pending_goal = None
        if self.state == "WAITING_APPROVAL":
            self.state = "OPERATIONAL"
        self.goal = (int(goal[0]), int(goal[1]))
        self.goal_source = source
        self.goal_red_ok = red_ok
        self.returning = False
        self.path = []
        lvl = "operator" if source == "operator" else "autonomous"
        self.log("NAV", f"New goal ({self.goal[0]},{self.goal[1]}) from {source} — footstep planner engaged", lvl)

    def approve_goal(self):
        g = self.pending_goal
        if not g:
            return
        self.log("SAFETY", f"Operator APPROVED entry into flagged unstable zone for goal ({g[0]},{g[1]})", "operator")
        self.set_goal(g, "operator", red_ok=True)

    def deny_goal(self):
        g = self.pending_goal
        self.pending_goal = None
        if self.state == "WAITING_APPROVAL":
            self.state = "OPERATIONAL"
        self.log("SAFETY", f"Operator DENIED entry into flagged unstable zone ({g[0] if g else '?'} ,{g[1] if g else '?'})", "operator")

    def cmd_return(self, reason, level="autonomous", to_base=True):
        tgt = self.base if to_base else (int(self.last_link_pos[0]), int(self.last_link_pos[1]))
        self.pending_goal = None
        if self.state == "WAITING_APPROVAL":
            self.state = "OPERATIONAL"
        self.goal = tuple(tgt)
        self.goal_source = "return-policy"
        self.goal_red_ok = False
        self.returning = True
        self.path = []
        self.log("NAV", ("Return-to-base" if to_base else "Return-to-last-link") + f": {reason}", level)

    def plan_path(self):
        if self.goal is None:
            return False
        start = (int(round(self.pos[0])), int(round(self.pos[1])))
        path, weak = astar(self.terra, self.stab, self.red, self.goal_red_ok, start, self.goal)
        if path is None:
            return False
        self.path, self.path_weak = path[1:], weak
        return True

    # ---------------- exploration ----------------
    def best_frontier(self):
        open_m = (self.terra != WALL) & (self.terra != GAP)
        ko = self.known & open_m & ~self.red
        unk = ~self.known
        adj = np.zeros((N, N), bool)
        adj[:-1, :] |= unk[1:, :]
        adj[1:, :] |= unk[:-1, :]
        adj[:, :-1] |= unk[:, 1:]
        adj[:, 1:] |= unk[:, :-1]
        cand = np.argwhere(ko & adj)
        if len(cand) == 0:
            return None
        best, best_score = None, -1.0
        px, py = self.pos
        for (y, x) in cand:
            d_go = (abs(x - px) + abs(y - py)) * CELL
            d_back = (abs(x - self.base[0]) + abs(y - self.base[1])) * CELL
            energy = (d_go + d_back) * 0.30 + 6.0
            feasible = self.battery - energy > 12.0
            gain = int(unk[max(0, y - 1):y + 2, max(0, x - 1):x + 2].sum())
            score = gain / (d_go + 0.6)
            if not feasible:
                score *= 0.03                       # battery-aware penalty
            if score > best_score:
                best_score, best = score, (int(x), int(y))
        return best

    def maybe_auto_goal(self):
        if self.t < self._next_fr:
            return
        self._next_fr = self.t + 2.0
        f = self.best_frontier()
        if f is None:
            if not self.returning:
                self.cmd_return("no unexplored frontiers remain — survey complete", "autonomous")
            return
        self.set_goal(f, "frontier-explorer (battery-aware)")

    # ---------------- mobility ----------------
    def check_fall(self, x, y, t):
        s = float(self.stab[y, x])
        if t == RUBBLE:
            p = max(0.0, 0.60 - s) * 0.55
        elif t == SLOPE:
            p = max(0.0, 0.75 - s) * 0.40
        else:
            p = 0.0
        if self.rng.random() < p:
            self.falls += 1
            self.state = "RECOVERING"
            self.recover_t = 4.5
            self.path = []
            self.battery -= 1.0
            self.log("MOBILITY", f"FALL on {TERRAIN_NAMES[t]} (stability {s:.2f}) — recovery: brace→kneel→stand→self-check", "autonomous")

    def move(self, dt):
        if self.goal is not None and not self.path:
            if self.plan_path():
                self.log("NAV", f"Footstep plan: {len(self.path)} steps, weakest stability {self.path_weak:.2f}", "autonomous")
            else:
                self.log("NAV", f"No stable footstep plan to ({self.goal[0]},{self.goal[1]}) — goal dropped", "autonomous")
                self.goal, self.returning, self.state = None, False, "IDLE"
                return
        if not self.path:
            if self.state == "MOVING":
                self.state = "IDLE"
            return
        nx, ny = self.path[0]
        t = self.terra[ny, nx]
        if t in (WALL, GAP) or (self.red[ny, nx] and not self.goal_red_ok):
            self.log("NAV", "Route invalidated (terrain change / safety rule) — re-planning footsteps", "autonomous")
            self.path = []
            return
        speed = SPEED_CPS.get(t, 1.0) * dt
        target = np.array([nx, ny], float)
        vec = target - self.pos
        d = float(np.linalg.norm(vec))
        step_len = min(d, speed)
        if d > 1e-9:
            self.pos = self.pos + vec / d * step_len
            self.heading = math.atan2(vec[1], vec[0])
            self.distance += step_len * CELL
            self.battery -= step_len * CELL * 0.28
        if d <= speed + 1e-9:
            self.path.pop(0)
            self.check_fall(nx, ny, t)
            if not self.path and self.goal is not None and (nx, ny) == tuple(self.goal):
                if self.returning and self.battery < 25 and self.goal == tuple(self.base):
                    self.end_mission("Returned to base on battery reserve")
                    return
                if self.returning:
                    self.log("NAV", "Reached last-link position — attempting comms re-acquisition", "autonomous")
                else:
                    self.log("NAV", f"Reached goal ({nx},{ny}) — scanning sector", "autonomous")
                self.goal, self.returning, self.state = None, False, "IDLE"
        if self.state not in ("RECOVERING", "WAITING_APPROVAL"):
            self.state = "MOVING"

    def battery_policy(self):
        b = self.battery
        if not self.warn35 and b < 35:
            self.warn35 = True
            self.log("POWER", "Battery 35% — frontier scoring now heavily weights energy-to-return", "autonomous")
        if not self.warn20 and b < 20:
            self.warn20 = True
            if not self.returning:
                self.cmd_return("battery 20% reserve threshold", "autonomous")
        if not self.warn10 and b < 10:
            self.warn10 = True
            self.log("POWER", "Battery 10% — emergency return, non-essential sensing throttled", "autonomous")

    def end_mission(self, reason):
        if self.mission_over:
            return
        self.mission_over = True
        self.state = "MISSION_COMPLETE"
        self.log("SYSTEM", f"Mission complete: {reason}", "autonomous")

    def step(self, dt):
        if self.mission_over:
            return
        self.t += dt
        self.scan()
        self.battery -= dt * 0.045
        self.update_survivors(dt)
        self.update_comms(dt)
        self.battery_policy()
        if self.t > self._next_shift:
            self._next_shift = self.t + float(self.rng.uniform(18, 35))
            self.floor_shift()
        if self.state == "RECOVERING":
            self.recover_t -= dt
            if self.recover_t <= 0:
                self.state = "OPERATIONAL"
                self.log("MOBILITY", "Get-up routine complete — self-check OK, resuming route", "autonomous")
        elif self.state != "WAITING_APPROVAL":
            self.move(dt)
            if self.auto_explore and self.goal is None and not self.returning:
                self.maybe_auto_goal()
        if self.distance - self._last_rec > 0.4:
            self._last_rec = self.distance
            self.trail.append(self.pos.copy())
        self.battery = max(self.battery, 0.0)
        if self.battery <= 1.0:
            self.end_mission("Battery depleted in the field")


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def explored_pct(sim):
    open_m = (sim.terra != WALL) & (sim.terra != GAP)
    return 100.0 * (sim.known & open_m).sum() / max(1, int(open_m.sum()))


def h_at(sim, x, y):
    xi = min(N - 1, max(0, int(round(x))))
    yi = min(N - 1, max(0, int(round(y))))
    return float(sim.height[yi, xi])


def render_tactical(sim, px=9):
    img = np.zeros((N, N, 3), np.uint8)
    img[:] = (12, 14, 20)
    known = sim.known
    for code, rgb in TERRAIN_RGB.items():
        m = known & (sim.terra == code)
        if m.any():
            img[m] = rgb
    open_m = (sim.terra != WALL) & (sim.terra != GAP)
    m = known & sim.red & open_m
    if m.any():
        img[m] = (img[m] * 0.45 + np.array([235, 60, 50]) * 0.55).astype(np.uint8)
    big = np.repeat(np.repeat(img, px, axis=0), px, axis=1)
    im = Image.fromarray(big)
    dr = ImageDraw.Draw(im)

    def P(p):
        return (p[0] * px + px / 2, p[1] * px + px / 2)

    if len(sim.trail) > 1:
        dr.line([P(p) for p in sim.trail], fill=(255, 200, 60), width=2)
    if sim.path:
        for q in sim.path[::2]:
            cx, cy = q[0] * px + px / 2, q[1] * px + px / 2
            dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(240, 240, 240))
    if sim.goal:
        gx, gy = P(sim.goal)
        dr.ellipse([gx - 8, gy - 8, gx + 8, gy + 8], outline=(255, 140, 0), width=2)
        dr.line([gx - 11, gy, gx + 11, gy], fill=(255, 140, 0), width=1)
        dr.line([gx, gy - 11, gx, gy + 11], fill=(255, 140, 0), width=1)
    bx, by = P(sim.base)
    dr.rectangle([bx - 6, by - 6, bx + 6, by + 6], outline=(80, 220, 120), width=2)
    dr.text((bx + 8, by - 6), "BASE", fill=(80, 220, 120))
    for bp, _ in sim.beacons:
        x, y = P(bp)
        dr.rectangle([x - 3, y - 3, x + 3, y + 3], fill=(250, 220, 80))
    for s in sim.survivors:
        if not s["detected"]:
            continue
        x, y = P(s["pos"])
        col = (60, 230, 120) if s["conf"] > 0.7 else (250, 180, 40) if s["conf"] > 0.45 else (255, 90, 70)
        dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=col, width=2)
        dr.text((x + 11, y - 7), f"S{s['id']} {s['conf']:.0%}", fill=col)
    rx, ry = P(sim.pos)
    r, h = px * 1.5, sim.heading
    pts = [(rx + r * math.cos(h), ry + r * math.sin(h)),
           (rx + r * 0.7 * math.cos(h + 2.6), ry + r * 0.7 * math.sin(h + 2.6)),
           (rx + r * 0.7 * math.cos(h - 2.6), ry + r * 0.7 * math.sin(h - 2.6))]
    dr.polygon(pts, fill=(255, 255, 255), outline=(0, 0, 0))
    
    # FIX: Return PIL Image directly instead of numpy array so im.paste() works natively
    return im


def fig3d(sim):
    fig = go.Figure()
    ys, xs = np.where(sim.known)
    if len(xs):
        t = sim.terra[ys, xs]
        cols = [f"rgb({TERRAIN_RGB[c][0]},{TERRAIN_RGB[c][1]},{TERRAIN_RGB[c][2]})" for c in t.tolist()]
        fig.add_trace(go.Scatter3d(
            x=xs * CELL, y=ys * CELL, z=sim.height[ys, xs], mode="markers",
            marker=dict(size=3.2, color=cols, opacity=0.92),
            hovertemplate="x %{x:.1f}m<br>y %{y:.1f}m<br>z %{z:.2f}m<extra>SLAM voxel</extra>",
            name="SLAM voxels"))
        m = sim.known & sim.red & ((sim.terra != WALL) & (sim.terra != GAP))
        if m.any():
            ry_, rx_ = np.where(m)
            fig.add_trace(go.Scatter3d(
                x=rx_ * CELL, y=ry_ * CELL, z=sim.height[m] + 0.25, mode="markers",
                marker=dict(size=4.5, color="rgba(255,60,50,0.35)"),
                name="UNSAFE zone (no-entry)", hovertemplate="structurally unsafe<extra></extra>"))
    for s in sim.survivors:
        if not s["detected"]:
            continue
        col = "#3ce678" if s["conf"] > 0.7 else "#fab428" if s["conf"] > 0.45 else "#ff5a46"
        sx, sy = s["pos"]
        fig.add_trace(go.Scatter3d(
            x=[sx * CELL], y=[sy * CELL], z=[h_at(sim, sx, sy) + 0.8], mode="markers",
            marker=dict(size=9, color=col, symbol="diamond"),
            text=[f"Survivor S{s['id']} · likelihood {s['conf']:.0%}"], hoverinfo="text",
            name=f"S{s['id']} ({s['conf']:.0%})"))
    if sim.beacons:
        fig.add_trace(go.Scatter3d(
            x=[b[0][0] * CELL for b in sim.beacons], y=[b[0][1] * CELL for b in sim.beacons],
            z=[h_at(sim, b[0][0], b[0][1]) + 0.3 for b in sim.beacons], mode="markers",
            marker=dict(size=5, color="#fadf50", symbol="square"), name="Relay beacons"))
    if len(sim.trail) > 1:
        tr = np.array(sim.trail)
        fig.add_trace(go.Scatter3d(
            x=tr[:, 0] * CELL, y=tr[:, 1] * CELL,
            z=[h_at(sim, p[0], p[1]) + 0.12 for p in sim.trail],
            mode="lines", line=dict(color="#ffcf3c", width=3), name="Robot trail"))
    px_, py_ = sim.pos
    fig.add_trace(go.Scatter3d(
        x=[px_ * CELL], y=[py_ * CELL], z=[h_at(sim, px_, py_) + 0.5], mode="markers",
        marker=dict(size=7, color="#ffffff"), name="SAR-1",
        hovertemplate="SAR-1 humanoid<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", height=600, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data", bgcolor="#0b0e14"),
        legend=dict(orientation="h", y=1.06), uirevision="cam")
    return fig


def hot_lut():
    x = np.arange(256) / 255
    r = np.clip(x * 3, 0, 1)
    g = np.clip(x * 3 - 1, 0, 1)
    b = np.clip(x * 3 - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def thermal_image(sim, w=240, h=168):
    rng = np.random.default_rng(int(sim.t * 12) % 100000 + 7)
    img = np.clip(rng.normal(0.10, 0.025, (h, w)), 0, 1)
    for s in sim.survivors:
        dvec = (s["pos"] - sim.pos) * CELL
        d = float(np.linalg.norm(dvec))
        if d > 6.5 or d < 0.01:
            continue
        ang = math.atan2(dvec[1], dvec[0])
        da = (ang - sim.heading + math.pi) % (2 * math.pi) - math.pi
        if abs(da) > 1.0:
            continue
        u = int(w / 2 + (da / 1.0) * (w / 2 - 18))
        v = int(h * 0.52)
        rad = max(3, int(30 / (d + 0.8)))
        yy, xx = np.ogrid[:h, :w]
        blob = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * (rad * 0.6) ** 2))
        img += blob * (0.55 + 0.45 * s["thermal"])
    idx = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return hot_lut()[idx]


def lidar_fig(sim):
    px, py = int(round(sim.pos[0])), int(round(sim.pos[1]))
    thetas, rs = [], []
    for a in range(0, 360, 3):
        rad = math.radians(a)
        dx, dy = math.cos(rad), math.sin(rad)
        x, y = px + 0.5, py + 0.5
        r = 11
        for k in range(1, 12):
            x += dx
            y += dy
            cx, cy = int(x), int(y)
            if not (0 <= cx < N and 0 <= cy < N) or sim.terra[cy, cx] == WALL:
                r = k
                break
        thetas.append(a)
        rs.append(r * CELL)
    fig = go.Figure(go.Scatterpolar(r=rs, theta=thetas, mode="markers",
                                    marker=dict(size=3, color="#39d0ff")))
    fig.update_layout(template="plotly_dark", height=300,
                      margin=dict(l=45, r=10, t=28, b=10),
                      polar=dict(bgcolor="#0b0e14",
                                 radialaxis=dict(range=[0, 6], color="#7a8699"),
                                 angularaxis=dict(color="#7a8699")),
                      showlegend=False,
                      title=dict(text="LiDAR sweep (6 m)", font=dict(size=13)))
    return fig


def audio_levels(sim):
    levels = [0.08, 0.07, 0.09, 0.06]
    tap, bearing = False, None
    for s in sim.survivors:
        if s["audio"] <= 0:
            continue
        dvec = s["pos"] - sim.pos
        d = float(np.linalg.norm(dvec)) * CELL
        if d < 8.5:
            phase = (sim.t % 1.2) / 1.2
            burst = 0.75 if phase < 0.18 else 0.15
            levels = [min(1.0, l + burst * s["audio"] * (0.7 + 0.3 * random.random())) for l in levels]
            tap = True
            bearing = math.degrees(math.atan2(dvec[1], dvec[0]))
    return levels, tap, bearing


def _font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


def render_frame(sim, W=960, H=540):
    im = Image.new("RGB", (W, H), (8, 10, 16))
    dr = ImageDraw.Draw(im)
    f_title, f_big, f_small = _font(20), _font(16), _font(13)
    px = 8
    
    # FIX: render_tactical now returns a PIL Image directly, making im.paste() safe
    tac_pil = render_tactical(sim, px=px)
    im.paste(tac_pil, (12, 16))
    
    dr.rectangle([12, 16, 12 + N * px, 16 + N * px], outline=(60, 70, 90), width=2)
    dr.text((14, 0), "SAR-1 PATHFINDER — LIVE SLAM MAP", fill=(230, 235, 245), font=f_title)
    x0 = 12 + N * px + 24
    y = 20
    dr.text((x0, y), "MISSION STATUS", fill=(120, 200, 255), font=f_big)
    y += 26
    rows = [("T+", f"{int(sim.t // 60):02d}:{int(sim.t % 60):02d}"),
            ("STATE", sim.state),
            ("COMMS", f"{sim.mode_comms} {sim.signal:.0f}%"),
            ("DIST", f"{sim.distance:.1f} m"),
            ("MAP", f"{explored_pct(sim):.0f}% known"),
            ("FALLS", str(sim.falls)),
            ("BEACONS", f"{4 - sim.beacons_left} deployed")]
    for k, v in rows:
        dr.text((x0, y), k, fill=(140, 150, 170), font=f_small)
        dr.text((x0 + 92, y), v, fill=(235, 240, 250), font=f_small)
        y += 20
    y += 8
    dr.text((x0, y), "BATTERY", fill=(140, 150, 170), font=f_small)
    y += 18
    dr.rectangle([x0, y, x0 + 300, y + 16], outline=(90, 100, 120))
    bw = int(296 * sim.battery / 100)
    col = (80, 220, 120) if sim.battery > 50 else (250, 190, 60) if sim.battery > 20 else (255, 80, 60)
    dr.rectangle([x0 + 2, y + 2, x0 + 2 + bw, y + 14], fill=col)
    dr.text((x0 + 308, y), f"{sim.battery:.0f}%", fill=(235, 240, 250), font=f_small)
    y += 34
    dr.text((x0, y), "SURVIVOR LIKELIHOOD", fill=(120, 200, 255), font=f_big)
    y += 24
    for s in sorted(sim.survivors, key=lambda s: -s["conf"]):
        if not s["detected"] and s["conf"] <= 0:
            dr.text((x0, y), f"S{s['id']}: — not detected", fill=(110, 120, 140), font=f_small)
        else:
            c = (60, 230, 120) if s["conf"] > 0.7 else (250, 180, 40) if s["conf"] > 0.45 else (255, 90, 70)
            dr.text((x0, y), f"S{s['id']}: {s['conf']:.0%}", fill=c, font=f_small)
            dr.rectangle([x0 + 92, y + 2, x0 + 92 + int(180 * s["conf"]), y + 12], fill=c)
        y += 20
    y += 10
    dr.text((x0, y), "LATEST AUTONOMOUS DECISIONS", fill=(120, 200, 255), font=f_big)
    y += 22
    for d in list(sim.decisions)[-5:]:
        dr.text((x0, y), f"[{d['subsystem']}] {d['decision']}"[:60], fill=(170, 180, 200), font=f_small)
        y += 17
    dr.rectangle([0, H - 26, W, H], fill=(30, 14, 14))
    dr.text((12, H - 21),
            "SAFETY: no entry into sensor-flagged UNSTABLE zones without operator approval · every autonomous decision is logged",
            fill=(255, 150, 130), font=f_small)
    return np.array(im)


def record_video(sim_src, seconds, fps=12, speed=2.0, prog=None):
    sim = copy.deepcopy(sim_src)
    W, H = 960, 540
    total = int(seconds * fps)
    frames_pil = []
    vw, path = None, None
    if HAVE_CV2:
        path = os.path.join(tempfile.gettempdir(), f"sar1_demo_{int(time.time())}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not vw.isOpened():
            vw = None
    for i in range(total):
        sim.step((1.0 / fps) * speed)
        frame = render_frame(sim, W, H)
        if vw is not None:
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif seconds <= 20:
            frames_pil.append(Image.fromarray(frame))
        if prog is not None and i % (fps * 2) == 0:
            prog.progress((i + 1) / total, text=f"Recording live sim… {int((i + 1) / fps)}s / {seconds}s")
    if vw is not None:
        vw.release()
        with open(path, "rb") as f:
            data = f.read()
        try:
            os.remove(path)
        except OSError:
            pass
        return data, "video/mp4", "SAR1_live_demo.mp4"
    if frames_pil:
        buf = io.BytesIO()
        frames_pil[0].save(buf, format="GIF", save_all=True, append_images=frames_pil[1:],
                           duration=int(1000 / fps), loop=0)
        return buf.getvalue(), "image/gif", "SAR1_live_demo.gif"
    raise RuntimeError("No video backend available")


# ----------------------------------------------------------------------------
# Exports for structural engineers
# ----------------------------------------------------------------------------
def build_ply(sim):
    ys, xs = np.where(sim.known)
    lines = ["ply", "format ascii 1.0",
             f"comment SAR-1 PATHFINDER collapsed-structure scan | mission {sim.t:.0f}s | cell {CELL}m",
             f"element vertex {len(xs)}",
             "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green", "property uchar blue",
             "property float stability", "property uchar unsafe_flag", "end_header"]
    for x, y in zip(xs, ys):
        c = TERRAIN_RGB[int(sim.terra[y, x])]
        lines.append(f"{x * CELL:.2f} {y * CELL:.2f} {sim.height[y, x]:.2f} "
                     f"{c[0]} {c[1]} {c[2]} {sim.stab[y, x]:.2f} {int(sim.red[y, x])}")
    return "\n".join(lines)


def build_grid_csv(sim):
    ys, xs = np.where(sim.known)
    df = pd.DataFrame({
        "x_m": xs * CELL, "y_m": ys * CELL,
        "terrain": [TERRAIN_NAMES[int(t)] for t in sim.terra[ys, xs]],
        "height_m": sim.height[ys, xs],
        "stability": np.round(sim.stab[ys, xs], 2),
        "unsafe_flag": sim.red[ys, xs].astype(int)})
    return df.to_csv(index=False).encode()


def survivors_csv(sim):
    rows = []
    for s in sim.survivors:
        rows.append(dict(survivor=f"S{s['id']}", grid_x=int(s["pos"][0]), grid_y=int(s["pos"][1]),
                         world_x_m=round(s["pos"][0] * CELL, 2), world_y_m=round(s["pos"][1] * CELL, 2),
                         likelihood=round(s["conf"], 3),
                         evidence_thermal=round(float(s["evid"][0]), 3),
                         evidence_audio=round(float(s["evid"][1]), 3),
                         evidence_visual=round(float(s["evid"][2]), 3),
                         detected=s["detected"]))
    return pd.DataFrame(rows).to_csv(index=False).encode()


def build_summary(sim):
    return dict(mission="SAR-1 PATHFINDER — collapsed-structure search & rescue",
                exported_utc=time.strftime("%Y-%m-%d %H:%M:%S"),
                mission_time_s=round(sim.t, 1), distance_travelled_m=round(sim.distance, 1),
                map_known_pct=round(explored_pct(sim), 1), falls=sim.falls,
                beacons_deployed=4 - sim.beacons_left, battery_remaining_pct=round(sim.battery, 1),
                red_zones_flagged=[{"zone": i + 1, "center_grid": [int(z[1]), int(z[0])]}
                                   for i, z in enumerate(sim.red_zones)],
                survivors=[{"id": f"S{s['id']}", "likelihood": round(s["conf"], 3),
                            "grid": [int(s['pos'][0]), int(s['pos'][1])]} for s in sim.survivors],
                autonomous_decisions_logged=len(sim.decisions),
                safety_rules=["no entry into flagged unstable zones without operator approval",
                              "every autonomous decision logged"])


def badge(state):
    colors = {"MOVING": "#3b82f6", "IDLE": "#64748b", "RECOVERING": "#f59e0b",
              "WAITING_APPROVAL": "#ef4444", "OPERATIONAL": "#22c55e",
              "MISSION_COMPLETE": "#a855f7"}
    c = colors.get(state, "#64748b")
    return (f"<span style='background:{c};color:white;padding:3px 12px;border-radius:12px;"
            f"font-size:13px;font-weight:600'>{state}</span>")


# ----------------------------------------------------------------------------
# App state & sidebar
# ----------------------------------------------------------------------------
st.session_state.setdefault("speed", 2.0)
st.session_state.setdefault("running", True)
if "sim" not in st.session_state:
    st.session_state.sim = Sim(seed=11)
sim = st.session_state.sim

with st.sidebar:
    st.title("🤖 SAR-1 PATHFINDER")
    st.caption("Collapsed-structure search & rescue · humanoid explorer console")
    st.markdown(badge(sim.state), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Battery", f"{sim.battery:.0f}%")
    c2.metric("Comms", f"{sim.signal:.0f}%")
    c3, c4 = st.columns(2)
    c3.metric("T+", f"{int(sim.t // 60):02d}:{int(sim.t % 60):02d}")
    c4.metric("Distance", f"{sim.distance:.1f} m")
    st.divider()
    cc1, cc2 = st.columns(2)
    if cc1.button("⏸ Pause" if st.session_state.running else "▶ Resume", use_container_width=True):
        st.session_state.running = not st.session_state.running
    if cc2.button("🏠 Return to base", use_container_width=True):
        sim.cmd_return("operator recall", "operator")
    sim.auto_explore = st.checkbox("Battery-aware auto-explore", value=sim.auto_explore)
    sim.red_override = st.checkbox("Test override: allow flagged-zone routing",
                                   value=sim.red_override,
                                   help="Bypasses the safety hold — controlled tests only.")
    st.session_state.speed = st.select_slider("Sim speed", options=[1.0, 2.0, 4.0],
                                              value=float(st.session_state.speed))
    if st.button("🏁 End mission", use_container_width=True):
        sim.end_mission("Operator ended mission")
    st.divider()
    with st.expander("🎬 Record live demo video (≤60 s)", expanded=False):
        secs = st.selectbox("Duration", [15, 30, 60], index=1)
        if st.button("● Start recording", use_container_width=True):
            prog = st.progress(0, text="Recording live simulation…")
            data, mime, fn = record_video(sim, secs, fps=12,
                                          speed=max(2.0, float(st.session_state.speed)), prog=prog)
            prog.empty()
            st.session_state.demo = (data, mime, fn)
        if st.session_state.get("demo"):
            d, m, f = st.session_state.demo
            st.video(d)
            st.download_button("⬇ Download recording", d, file_name=f, mime=m,
                               use_container_width=True)
    st.divider()
    st.markdown("**🛡️ Safety rules (enforced)**")
    st.caption("• NEVER enter sensor-flagged UNSTABLE zones without operator approval.\n"
               "• EVERY autonomous decision is logged.\n"
               "• Auto-return at 20% battery · return-to-last-link on comms loss.")
    st.divider()
    st.markdown("**Latest decisions**")
    for d in list(sim.decisions)[-6:][::-1]:
        st.caption(f"`{d['subsystem']}` {d['decision'][:84]}")

# ----------------------------------------------------------------------------
# Header + tabs
# ----------------------------------------------------------------------------
st.markdown("## 🚨 Live Operation — Collapsed Structure Interior")
h1, h2, h3, h4, h5, h6 = st.columns(6)
h1.metric("State", sim.state)
h2.metric("Battery", f"{sim.battery:.0f}%")
h3.metric("Comms", f"{sim.signal:.0f}% ({sim.mode_comms})")
h4.metric("Map known", f"{explored_pct(sim):.0f}%")
h5.metric("Survivors", f"{sum(s['detected'] for s in sim.survivors)} / {len(sim.survivors)}")
h6.metric("Falls", sim.falls)

tab3d, tac, sens, surv, logt, exp = st.tabs(
    ["🧊 3D SLAM Map", "🗺️ Tactical — click to send", "📡 Sensor Feeds",
     "🧍 Survivors", "📜 Decision Log", "📤 Export"])

with tab3d:
    st.plotly_chart(fig3d(sim), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption("🟦 floor · 🟫 rubble · 🟪 slope · ⬛ wall · ⬛void · 🔴 translucent red = "
               "sensor-flagged UNSAFE zone · ◆ = survivor likelihood marker · camera state is "
               "preserved while the map grows (uirevision).")

with tac:
    st.caption("👆 **Click any explored point to send SAR-1 there.** The footstep planner scores "
               "rubble / gaps / slopes for stability and re-plans automatically when the floor shifts.")
    tac_img = render_tactical(sim, px=9)
    if HAVE_CLICK:
        click = streamlit_image_coordinates(tac_img, key="tacmap")
        if isinstance(click, dict) and "x" in click:
            key_pt = (int(click["x"]), int(click["y"]))
            if st.session_state.get("_last_click") != key_pt:
                st.session_state._last_click = key_pt
                gx = min(N - 1, int(click["x"] // 9))
                gy = min(N - 1, int(click["y"] // 9))
                res = sim.request_goal(gx, gy)
                MSGS = {"ok": ("success", f"🎯 Goal ({gx},{gy}) accepted — footstep planner engaged."),
                        "approval": ("warning", "⚠️ Goal lies inside a sensor-flagged UNSTABLE zone — approval required below."),
                        "blocked": ("error", "⛔ Goal rejected: no foothold possible (wall / void)."),
                        "unmapped": ("info", "🌫 Target not yet mapped by SLAM — explore closer first."),
                        "invalid": ("error", "Invalid target.")}
                st.session_state.notice = MSGS[res]
    else:
        st.image(tac_img, caption="Install streamlit-image-coordinates to enable click-to-move")
        g1, g2, g3 = st.columns([1, 1, 2])
        gx = g1.number_input("goal x", 0, N - 1, 30)
        gy = g2.number_input("goal y", 0, N - 1, 30)
        if g3.button("Send robot"):
            res = sim.request_goal(int(gx), int(gy))
    st.button("🧭 Send to best frontier (autonomous)",
              on_click=lambda: (sim.set_goal(f, "operator") if (f := sim.best_frontier()) else None))
    if sim.pending_goal:
        with st.container(border=True):
            st.error(f"🚧 **SAFETY HOLD** — target ({sim.pending_goal[0]},{sim.pending_goal[1]}) is inside a "
                     f"structurally **UNSAFE** zone flagged by shoring sensors. The robot is holding position.")
            b1, b2, _ = st.columns([1, 1, 2])
            if b1.button("✅ Approve entry", type="primary"):
                sim.approve_goal()
                st.session_state.notice = ("success", "Entry approved — proceeding with heightened stability margins.")
            if b2.button("⛔ Deny"):
                sim.deny_goal()
                st.session_state.notice = ("info", "Entry denied — robot holding position.")
    if st.session_state.get("notice"):
        kind, msg = st.session_state.notice
        getattr(st, kind)(msg)
        st.session_state.notice = None
    if sim.goal:
        st.info(f"Active goal: ({sim.goal[0]}, {sim.goal[1]}) · source: {sim.goal_source} · "
                f"steps remaining: {len(sim.path)} · weakest stability on route: {sim.path_weak:.2f}")

with sens:
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**🌡 Thermal (FLIR)**")
        st.image(thermal_image(sim), use_container_width=True,
                 caption="Warm blobs = candidate heat signatures within 6 m cone")
    with s2:
        st.markdown("**🎙 Mic array (4-ch)**")
        levels, tap, bearing = audio_levels(sim)
        for i, lv in enumerate(levels, 1):
            st.progress(float(np.clip(lv, 0, 1)), text=f"MIC {i}")
        if tap:
            st.warning(f"🔊 Rhythmic tapping / voice candidate — bearing ≈ {bearing:.0f}°")
        else:
            st.caption("Ambient noise only.")
    with s3:
        st.markdown("**📡 LiDAR**")
        st.plotly_chart(lidar_fig(sim), use_container_width=True)
    st.dataframe(pd.DataFrame({
        "Sensor": ["Stereo depth cameras", "LiDAR", "Thermal camera", "Microphone array", "IMU"],
        "Status": ["OK"] * 5,
        "Rate": ["30 fps", "10 Hz", "9 Hz", "48 kHz", "200 Hz"]}),
        hide_index=True, use_container_width=True)

with surv:
    ranked = sorted(sim.survivors, key=lambda s: -s["conf"])
    if not any(s["detected"] for s in ranked):
        st.info("No detections yet — autonomous sweep in progress…")
    for i, s in enumerate(ranked, 1):
        with st.container(border=True):
            ca, cb, cc = st.columns([2, 2, 3])
            status = "NOT DETECTED" if not s["detected"] else ("CONFIRMED" if s["conf"] > 0.7 else "CANDIDATE")
            ca.markdown(f"**#{i} · Survivor S{s['id']}**")
            ca.caption(status)
            cb.metric("Likelihood", f"{s['conf']:.0%}")
            cb.progress(float(s["conf"]))
            labels = ["thermal", "audio", "visual"]
            for name, ev in zip(labels, s["evid"]):
                cc.progress(float(np.clip(ev, 0, 1)), text=f"evidence: {name}")
            fused = [n for n, ev in zip(["thermal blob", "audio tap/voice", "visual pose"], s["evid"]) if ev > 0.05]
            cc.caption("Fusing: " + (" + ".join(fused) if fused else "no evidence yet")
                       + f" · grid ({int(s['pos'][0])},{int(s['pos'][1])})")

with logt:
    df = pd.DataFrame(list(sim.decisions))[::-1].reset_index(drop=True)
    subs = ["ALL"] + sorted(df["subsystem"].unique().tolist()) if len(df) else ["ALL"]
    sub = st.selectbox("Filter subsystem", subs)
    shown = df if sub == "ALL" else df[df["subsystem"] == sub]
    st.dataframe(shown, use_container_width=True, height=380)
    st.download_button("⬇ Download decision log (CSV)", df.to_csv(index=False).encode(),
                       "mission_decisions.csv", "text/csv")
    st.caption(f"{len(df)} decisions logged — `level` distinguishes operator actions from autonomous decisions.")

with exp:
    st.markdown("### 📤 Post-mission export (structural engineers)")
    st.caption("Point cloud includes per-voxel height, RGB terrain class, stability score and unsafe flag. "
               "Generate artifacts on demand, then download.")
    e1, e2 = st.columns(2)
    if e1.button("🧊 Generate 3D point cloud (.ply)"):
        st.session_state.ply = build_ply(sim)
    if st.session_state.get("ply"):
        e1.download_button("⬇ Download sar1_map.ply", st.session_state.ply.encode(),
                           "sar1_map.ply", "application/octet-stream", use_container_width=True)
    if e2.button("🗂 Generate full grid (.csv)"):
        st.session_state.gridcsv = build_grid_csv(sim)
    if st.session_state.get("gridcsv"):
        e2.download_button("⬇ Download sar1_grid.csv", st.session_state.gridcsv,
                           "sar1_grid.csv", "text/csv", use_container_width=True)
    e3, e4 = st.columns(2)
    e3.download_button("⬇ Survivor report (.csv)", survivors_csv(sim),
                       "sar1_survivors.csv", "text/csv", use_container_width=True)
    e4.download_button("⬇ Mission summary (.json)",
                       json.dumps(build_summary(sim), indent=2).encode(),
                       "sar1_mission_summary.json", "application/json", use_container_width=True)

if sim.mission_over:
    st.success("🏁 Mission complete — export artifacts are available in the Export tab.")

# ----------------------------------------------------------------------------
# Live loop — the simulation auto-starts on page load (no blank dashboard)
# ----------------------------------------------------------------------------
if st.session_state.running and not sim.mission_over:
    sim.step(0.25 * float(st.session_state.speed))
    time.sleep(0.22)
    st.rerun()