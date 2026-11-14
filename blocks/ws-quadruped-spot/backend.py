# ── EsaSim block ──────────────────────────────────────────────────────
# Quadruped · Spot (ws-quadruped-spot) · version 6 · machine block, Python: def run(ctx)
# Boston Dynamics Spot in MuJoCo: learns to stand, walk and climb (PPO, GPU or CPU), then patrols a warehouse, hearing and seeing incidents and raising alarms
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-14
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Policies · several · optional · Spot's policies (.pt): walking, stand-up, look-around. Learn: carries on from the newest. Walk, Patrol: the newest of each
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Policy · spot_walk_[course_]…, spot_standup_… or spot_look_<date>_<NN>.pt: the best policy (network, scaling, optimizer, score)
#   curve       Table (csv)               Learning curve · training.csv (Learn to walk), standup.csv (Learn to stand) or look.csv (Learn to look around)
#   report      Files (md, json, csv)     Patrol report · Patrol: incident_report.md, incidents.json, patrol_log.csv (once a second: where, what it was doing, battery)
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    What to do = "Learn to walk" of Learn to stand | Learn to look around | Learn to walk | Walk | Patr…
#   robot       select    Robot = "Boston Dynamics Spot" of Boston Dynamics Spot
#   accelerator select    Run on = "auto" of auto | NVIDIA GPU | CPU
#   terrain     select    Ground = "Flat floor" of Flat floor | Obstacle course
#   robots      number    Robots learning at once = 4096
#   minutes     number    Minutes = 90
#   max_speed   number    Top speed asked (m/s) = 1.2
#   push        number    Pushes up to (m/s) = 1
#   replay_every number    Replay every (s) = 20
#   seed        number    Seed = 1
#   training    select    Training = "Continue" of Continue | Start over
#   speed       number    Walking speed (m/s) = 0.8
#   heading     number    Heading (degrees) = 0
#   policy_file text      Walking policy = ""
#   after_fall  select    After a fall = "Get up on its own" of Get up on its own | Put back on its feet
#   areas       textarea  Areas to patrol = "all"
#   incidents   select    Incidents = "Demo (about one every 90 s)" of Demo (about one every 90 s) | Realistic (about one eve…
#   start_time  text      Patrol starts at = "22:00"
#   pace        select    Pace = "Real time" of Real time | As fast as it can
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch, mujoco>=3.3, mujoco-warp, warp-lang (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Quadruped robot training and warehouse surveillance: the Boston Dynamics Spot (MuJoCo Menagerie, 12 joints, 32.9 kg)
learns, in MuJoCo (the physics engine robotics research uses), to stand and get up from a fall, to walk where it is told
(a trot, any speed and turn), and to cross obstacles (slopes, stairs, curbs, rough ground, ice) with an elevation map of
the ground ahead; then it patrols a warehouse (docks, racking aisles, staging, a mezzanine office up stairs and a ramp),
hearing and seeing what happens there and raising the alarm for intruders. Many robots learn at once on the NVIDIA GPU
(MuJoCo Warp), or on the CPU, with PPO.

Modes: Learn to stand (lying where a fall left it: on its side, on its back; it learns to roll over and get up), Learn to look
around (feet planted, it aims its front cameras up, down, left, right and lower with its legs), Learn
to walk (flat floor or the obstacle course), Walk (the newest policy drives the robot live in the 3D view: speed, way
round, push, reset), Patrol the warehouse (a session of surveillance: the round through the areas asked for, incidents,
alarms; it writes an incident report and a patrol log). Without a walking policy the patrol runs animated, and says so."""
import math


# the robot: Boston Dynamics Spot (MuJoCo Menagerie, Clearpath Robotics, BSD-3), collision shapes as primitives,
# meshes left out (the view draws them), compressed (zlib, base64)
SPOT_XML_Z = '''
eNrVWctyozoQ/RUVe2wk3lVxlre8uKv8gEvBsqUJIK7AmXi+/rYknn5MSCqecTZBhn6cPt1qOuKhOPyQmUSF3LJ85dSVbDasprUonMeHTBaVyJlCtNzn
bOUouhW0dNDy8UFWjZAlakTB6oZVK8dbeF7gIFE2bK9oI9XKEUWVi0w0O1o3DspkCSaqo6KF2NIcRBsGgmClXjmgmdeb8Z3EuNmyHT3kTb9AWU7r2uIE
gHsmC1TLXLGdQUAQNo6aYwW+7JrudqIUzbH7vRXFyvEdtFMi0760YoKMMvwBmb2Sh8qIAIBK1sJE+qJveZ6DXl4NOFFypkSjgBrrankGcicHkBZRXYES
c1AtfjED2I8cBC5gjTzkegvfj0K4o4RUFyBfjex34QCuZU/jpRVgZXDhO8HyLSppAQ4gEYoKSHWp5M+VEyZgqMwkVEhKcIef+AgHixBpT62j1tZPqfLt
s9we2+it0V0upeqDh4ABIERrQ6pyWrIpGV4Lf2Six2WVLGYH2evosTGDQxQDPNe6AUMaUWuoYvmrqIFGLvLtuKgGBN4iiCB9O8XYDwl1PUQBNVruN8+0
ZhvzxBgXJVONoHlrANBHfhoEyE0XCSY4Yq4X6og9L8XYj1MHFcarTxYJVAHsrH1rQmtjH4PuyfWUDRvEBvLSEfIs34bqCgLtEBPjNg2dETIdnrEG1c1a
a6I4jKMfVSk+FeWMbjcZLXqTGqP24p9pjTjf5Rsuql6HpHFioIXw1zkl0OQ/9OPANzKYJAFOEHAZh8TXXDrovwNtIMsIIxfrjW/5xIsoOaPT8wiJQsMk
LDEOYrvEkJo0MkCnOd5wIJK+CW3QstHudMAFsP1U7zG7AGcU+mS5144AFlUFbQ7KkqBxdlszl7UJHKDSrDnoJrlTWWc38lDkXaDskLP9kJcFxl5ygS7j
CgfEj+3OIYHfrqH2oOQ6dsjCD87ZIXEYppHpGrEfBqmlKQkSHF7k5thx4wH5E26SNEmg7AhkN8T4i6hJPJR4p8UPOBou9rwr/YxWNXA1KsDAOJBFIzVQ
Q8uoM3Y3XF2JF3jPR7yDTtg2aHKRekLCOA1QsIi82O50zXwC9slQp3pb9WUKpsILZRp4oecbgJ6fpqnFCnUapH5wKRUv5ZVUkEWcktREB3XvkS9KBSbg
h1xKRs2HvnyeCzLOhXZo+4UNj1hu2zSMzP7zrzN5mZ60IX1vo2XOX6FDERD7ZrJvo+llnHB1oTe5c5qTO3Qn90+1J3Uv7UlN25M7pz/duD2pee0p8CP/
pu1Jfaw9nXan3zUn9S2ak5rXnMIoSG7cnNRtmtPTjOb09BXNiY8HJ/e9yeni4PSnWhO/l8mJ39/kxO9kcuK3nJz4t5ic+LzJKYi9KL5tc+K3mZzWMyan
9ZdMTlxdak7ujO7k/vH/6/i9DE78/gYnru6kO91ucOLfYnDi8wanICHJjf+r47cZnNYzBqf15wen7jI6hNQHpxD24wN7y/LDliF9Gw+ngPonGZ22LN+V
VHMl+WybfGRz2SPusjU6ip4cV04OrEwZ9b+W7+scJzrHOTq6Ngedl/J9HTXBpmZhUxNsahY2NcGm5mDjE974LN74hDc+izc+4Y3P4o1PeOOzeOMT3vgs
3viEN97zthxKr2Zlra/7o5JIb9X2tNga0HeNxivLZQZfhOAY/lwKHlGYDUUJUkaaZhmDj0vyirx5bAR38MmI/dL9EMnnH+2XFBB39M/R2bVdHqpXljX6
Q0Onan3O0d3n8nkK0hiALvkxA1ZhMAD0X9XuD7G6I68NSM8Gf67dOW7kIeMtredS5ulHAD71yk+fADjWvg5wLPVRgOs+uvUnGJxoXwU4kfowwD669ScY
nGhfBziWGgAuu/37wo7Go1l12x82IEwzk89eyI41ZrSBly3M4fqb4vtreK83KjdTy8fUlmNELyVj9eaZ6Q9rfx/XciBtaT+WP/4PtV8d1A==
'''


# ── the warehouse: its layout, the way round it, what the robot hears and sees, what happens in it ────────────────────
# A 42 x 25 m hall (x from -6 to 36, y from -5 to 20, metres; z up). Along its south wall four loading docks with roller
# doors and the dock aisle; three double rows of pallet racking (2.2 m deep, 8 m high, a cross aisle in the middle) with
# aisles A, B and C between them; a staging area of pallets along the north wall; to the east an open floor with
# pallets and a pallet truck, and a mezzanine 0.84 m up (the office on it) reached by a ramp (15 %) and by 7 steps of
# 12 cm; the robot's charging dock in the north-west corner. In the physics the walls, the racking, the stacks and the
# office walls are blocks of the heightfield (1.9 m tall: the robot cannot see or step over them), the mezzanine, its
# stairs and ramp are its real heights, a cable protector (5 cm) crosses the dock aisle.
import heapq
import math

WALL_H = 1.9                  # (the heightfield's tallest: the blocks the robot cannot pass)


def make_warehouse():
    import numpy as np
    hall = {"x0": -6.0, "x1": 36.0, "y0": -5.0, "y1": 20.0, "height": 9.0}
    boxes = []

    def box(x0, x1, y0, y1, h, kind):
        boxes.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "h": h, "kind": kind})
    t = 0.3                                                   # the walls
    box(hall["x0"] - t, hall["x1"] + t, hall["y0"] - t, hall["y0"], WALL_H, "wall")
    box(hall["x0"] - t, hall["x1"] + t, hall["y1"], hall["y1"] + t, WALL_H, "wall")
    box(hall["x0"] - t, hall["x0"], hall["y0"], hall["y1"], WALL_H, "wall")
    box(hall["x1"], hall["x1"] + t, hall["y0"], hall["y1"], WALL_H, "wall")
    racks = []
    for name, (y0, y1) in (("R1", (0.8, 3.0)), ("R2", (6.0, 8.2)), ("R3", (11.2, 13.4))):
        for x0, x1 in ((2.0, 12.3), (14.7, 24.0)):
            racks.append({"name": name, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "levels": 4, "bay": 2.7, "height": 8.0})
            box(x0, x1, y0, y1, WALL_H, "rack")
    stacks = []                                               # staging: pallets of goods along the north wall
    for i, x in enumerate(np.arange(2.2, 20.5, 1.6)):
        h = [1.2, 0.9, 1.5, 0.6, 1.3, 1.1, 0.8, 1.4, 1.0, 1.2, 0.7, 1.5][i % 12]
        stacks.append({"x": round(float(x), 2), "y": 18.9, "w": 1.2, "d": 1.0, "h": h})
        box(float(x) - 0.6, float(x) + 0.6, 18.4, 19.4, min(WALL_H, h), "stack")
    pallets = [{"x": 33.6, "y": 0.6, "yaw": 0.2, "h": 0.15, "load": 0.0}, {"x": 33.5, "y": -2.2, "yaw": 0.0, "h": 0.15, "load": 1.1},
               {"x": 34.6, "y": 8.4, "yaw": -0.3, "h": 0.15, "load": 0.8}, {"x": 34.6, "y": 4.0, "yaw": 1.57, "h": 0.15, "load": 0.0}]
    for p in pallets:                                         # (as blocks: their footprint, the load's height)
        box(p["x"] - 0.65, p["x"] + 0.65, p["y"] - 0.55, p["y"] + 0.55, max(p["h"], p["load"]), "pallet")
    # clutter to walk over (what the obstacle course taught it): aisle B half blocked by two pallets left on the floor
    # (14 cm), cartons fallen from the racks; aisle C a pallet and spilled cartons; it steps over them or onto them
    clutter = [{"x0": 4.6, "x1": 5.8, "y0": 8.35, "y1": 9.35, "h": 0.14, "what": "pallet"},
               {"x0": 5.1, "x1": 6.1, "y0": 9.5, "y1": 10.7, "h": 0.14, "what": "pallet"},
               {"x0": 7.4, "x1": 7.8, "y0": 9.15, "y1": 9.5, "h": 0.2, "what": "carton"},
               {"x0": 8.3, "x1": 8.75, "y0": 10.0, "y1": 10.35, "h": 0.18, "what": "carton"},
               {"x0": 8.9, "x1": 9.3, "y0": 8.7, "y1": 9.05, "h": 0.2, "what": "carton"},
               {"x0": 7.8, "x1": 8.45, "y0": 10.45, "y1": 10.95, "h": 0.06, "what": "flat"},
               {"x0": 17.6, "x1": 18.8, "y0": 13.75, "y1": 14.75, "h": 0.14, "what": "pallet"},
               {"x0": 19.3, "x1": 19.7, "y0": 14.9, "y1": 15.25, "h": 0.2, "what": "carton"},
               {"x0": 20.0, "x1": 20.45, "y0": 14.2, "y1": 14.55, "h": 0.18, "what": "carton"},
               {"x0": 20.6, "x1": 21.2, "y0": 14.7, "y1": 15.15, "h": 0.06, "what": "flat"},
               {"x0": 21.4, "x1": 21.8, "y0": 14.05, "y1": 14.4, "h": 0.2, "what": "carton"}]
    for c in clutter:
        box(c["x0"], c["x1"], c["y0"], c["y1"], c["h"], "clutter")
    # a roller conveyor across the east floor, from the dock wall to the mezzanine: crossed by a step-over bridge
    conveyor = {"x0": 28.0, "x1": 28.7, "y0": -5.0, "y1": 12.0, "h": 0.75, "gap": [1.3, 2.7]}
    box(conveyor["x0"], conveyor["x1"], conveyor["y0"], conveyor["gap"][0], conveyor["h"], "conveyor")
    box(conveyor["x0"], conveyor["x1"], conveyor["gap"][1], conveyor["y1"], conveyor["h"], "conveyor")
    plat = {"x0": 27.0, "x1": 36.0, "y0": 12.0, "y1": 20.0, "h": 0.84}          # the mezzanine, and the office on it
    box(plat["x0"], plat["x1"], plat["y0"], plat["y1"], plat["h"], "mezzanine")
    disp = {"x0": -6.0, "x1": -3.6, "y0": 6.0, "y1": 12.0, "h": 0.48, "name": "dispatch"}    # a dispatch platform at the west end
    box(disp["x0"], disp["x1"], disp["y0"], disp["y1"], disp["h"], "platform")
    office = {"x0": 29.0, "x1": 36.0, "y0": 15.0, "y1": 20.0, "door": [31.0, 32.6]}
    box(29.0, 29.15, 15.0, 20.0, WALL_H, "office")
    box(29.0, 31.0, 15.0, 15.15, WALL_H, "office")
    box(32.6, 36.0, 15.0, 15.15, WALL_H, "office")
    box(29.15, 30.6, 18.6, 20.0, 1.0, "desk")                                 # (a desk and a cabinet in it)
    box(34.4, 36.0, 15.15, 16.0, 1.6, "desk")
    rails = []                                                # railings round the mezzanine's open edges (gaps: the stairs, the ramp)
    for x0, x1, y0, y1 in ((27.0, 27.08, 12.0, 13.0), (27.0, 27.08, 15.0, 20.0), (27.0, 30.0, 12.0, 12.08), (32.0, 36.0, 12.0, 12.08)):
        rails.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": plat["h"]})
        box(x0, x1, y0, y1, WALL_H, "rail")
    for x0, x1, y0, y1 in ((-3.68, -3.6, 6.0, 8.15), (-3.68, -3.6, 9.85, 12.0), (-4.2, -3.6, 6.0, 6.08), (-6.0, -3.6, 11.92, 12.0)):
        rails.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": disp["h"]})
        box(x0, x1, y0, y1, WALL_H, "rail")
    stairs = [{"x0": 25.2, "y": 14.0, "dir": 1, "n": 7, "rise": 0.12, "tread": 0.3, "landing": 1.6, "width": 2.0, "name": "mezzanine", "label": "STAIRS  7 × 12 cm"},
              {"x0": 26.07, "y": 2.0, "dir": 1, "n": 7, "rise": 0.12, "tread": 0.28, "landing": 1.2, "width": 1.3, "name": "crossover", "label": "STEP-OVER  7 × 12 cm", "label_at": [27.3, 3.5]},
              {"x0": -1.5, "y": 9.0, "dir": -1, "n": 4, "rise": 0.12, "tread": 0.3, "landing": 1.2, "width": 1.6, "name": "dispatch", "label": "STEPS  4 × 12 cm", "label_at": [-1.6, 10.6]},
              {"x0": 8.0, "y": -0.1, "dir": 1, "n": 1, "rise": 0.05, "tread": 0.3, "landing": 0.3, "width": 1.8, "name": "curb", "label": "CABLE PROTECTOR  5 cm", "label_at": [8.2, -1.6]}]
    ramps = [{"x0": 30.0, "x1": 32.0, "y0": 6.5, "y1": 12.0, "h0": 0.0, "h1": 0.84, "axis": "y", "label": "RAMP  15 %"},
             {"x0": -6.0, "x1": -4.2, "y0": 2.2, "y1": 6.0, "h0": 0.0, "h1": 0.48, "axis": "y", "label": "RAMP  12.5 %"}]
    docks = [{"x": x, "w": 3.0, "open": 0.0} for x in (0.0, 6.0, 12.0, 18.0)]
    areas = [
        {"key": "docks", "name": "Loading docks", "path": [[-3.0, -2.6], [5.0, -2.6], [13.0, -2.6], [22.0, -2.6]], "look": [[12.0, -2.6]]},
        {"key": "aisle_a", "name": "Aisle A", "path": [[0.5, 4.5], [13.5, 4.5], [25.0, 4.5]], "look": [[13.5, 4.5]]},
        {"key": "aisle_b", "name": "Aisle B", "path": [[25.0, 9.7], [13.5, 9.7], [0.5, 9.7]], "look": [[13.5, 9.7]]},
        {"key": "aisle_c", "name": "Aisle C", "path": [[0.5, 14.9], [13.5, 14.9], [25.0, 14.9]], "look": [[13.5, 14.9]]},
        {"key": "staging", "name": "Staging area", "path": [[21.0, 17.2], [11.0, 17.2], [0.5, 17.2]], "look": [[11.0, 17.2]]},
        {"key": "east", "name": "East floor", "path": [[26.0, -2.5], [31.0, -3.0], [31.0, 5.0], [26.0, 9.0]], "look": [[31.0, 3.0]]},
        {"key": "dispatch", "name": "Dispatch platform", "path": [[-0.4, 9.0], [-4.8, 9.0], [-5.1, 7.2], [-5.1, 1.2]], "look": [[-4.8, 10.6]]},
        {"key": "office", "name": "Mezzanine office", "path": [[31.0, 5.5], [31.0, 13.5], [31.8, 14.4], [31.8, 16.2], [33.5, 17.5]], "look": [[33.0, 17.2]]},
        {"key": "stairs", "name": "Mezzanine stairs", "path": [[23.8, 14.0], [28.0, 14.0], [28.5, 13.0], [28.0, 14.0], [23.8, 14.0]], "look": []},
    ]
    return {"kind": "warehouse", "hall": hall, "boxes": boxes, "racks": racks, "stacks": stacks, "pallets": pallets, "platform": plat,
            "office": office, "rails": rails, "stairs": stairs, "clutter": clutter, "conveyor": conveyor, "platforms": [plat, disp], "ramps": ramps, "docks": docks, "areas": areas, "patches": [], "ramp": None,
            "dock": {"x": -4.4, "y": 18.2, "yaw": 0.0}, "start": [-3.6, 18.2, 0.0], "area": [-6.0, 36.0, -5.0, 20.0]}


WAREHOUSE = make_warehouse()
AREA_NAMES = [a["name"] for a in WAREHOUSE["areas"]]


# ── the way round: a grid of the floor (20 cm), what the robot can walk on, A* between two places ─────────────────────
class FloorMap:
    def __init__(self, H, x0, y0, res, lay=WAREHOUSE, clearance=0.42, flat_only=False):
        """H: the heightfield's heights (rows along y), at res metres from (x0, y0); walkable: the floor, and the
        mezzanine, its stairs and its ramp (everything else above 25 cm is in the way); kept clearance from what is"""
        import numpy as np
        step = max(1, int(round(0.2 / res)))
        self.res, self.x0, self.y0 = res * step, x0, y0
        G = H[::step, ::step]
        self.h = G
        ny, nx = G.shape
        self.nx, self.ny = nx, ny
        X = x0 + np.arange(nx) * self.res
        Y = y0 + np.arange(ny) * self.res
        XX, YY = np.meshgrid(X, Y)
        walk = np.zeros(G.shape, dtype=bool)
        for p in lay.get("platforms") or [lay["platform"]]:
            walk |= (XX >= p["x0"]) & (XX <= p["x1"]) & (YY >= p["y0"]) & (YY <= p["y1"]) & (G < p["h"] + 0.05)
        for st in lay["stairs"]:
            ln = 2 * (st["n"] - 1) * st["tread"] + st["landing"]
            xa, xb = sorted([st["x0"], st["x0"] + st["dir"] * ln])
            walk |= (XX >= xa) & (XX <= xb) & (np.abs(YY - st["y"]) <= st["width"] / 2) & (G < 1.0)
        for r in lay["ramps"]:
            walk |= (XX >= r["x0"]) & (XX <= r["x1"]) & (YY >= r["y0"]) & (YY <= r["y1"])
        hall = lay["hall"]
        inside = (XX > hall["x0"]) & (XX < hall["x1"]) & (YY > hall["y0"]) & (YY < hall["y1"])
        blocked = ((G > 0.25) & ~walk) | (G > 1.0) | ~inside
        if flat_only:        # (a robot that has learned the flat floor only: no stairs, ramps, platforms or clutter on its way)
            blocked |= (G > 0.03) & inside
        # the sides of the stairs and ramps, where they are high: no way on or off there (their stringers, railings)
        for st in lay["stairs"]:
            if st["name"] in ("mezzanine", "crossover", "dispatch"):
                ln = 2 * (st["n"] - 1) * st["tread"] + st["landing"]
                xa, xb = sorted([st["x0"], st["x0"] + st["dir"] * ln])
                d_ = np.abs(YY - st["y"])
                blocked |= (XX >= xa) & (XX <= xb) & (d_ > st["width"] / 2 - 0.05) & (d_ <= st["width"] / 2 + 0.3) & (G > 0.15)
                blocked |= (XX >= xa) & (XX <= xb) & (d_ > st["width"] / 2) & (d_ <= st["width"] / 2 + 0.3)
        for r in lay["ramps"]:
            u = np.clip((YY - r["y0"]) / (r["y1"] - r["y0"]), 0, 1)
            high = (r["h0"] + (r["h1"] - r["h0"]) * u) > 0.15
            alongy = (YY >= r["y0"]) & (YY <= r["y1"]) & high
            blocked |= alongy & (((XX >= r["x0"] - 0.3) & (XX < r["x0"])) | ((XX > r["x1"]) & (XX <= r["x1"] + 0.3)))
        self.solid = blocked.copy()                     # (what blocks the view: anything taller than the robot's eyes)
        self.sight = (G > 0.7) & ~walk | (G > 1.0)
        # keep clear of what is in the way (the robot is about 1.1 x 0.5 m): grow it by the clearance
        k = int(math.ceil(clearance / self.res))
        grown = blocked.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dx * dx + dy * dy > k * k:
                    continue
                grown |= np.roll(np.roll(blocked, dy, 0), dx, 1)
        # the stairs and the ramp keep their own width (their railings would close them)
        for st in lay["stairs"]:
            if st["name"] in ("mezzanine", "crossover", "dispatch"):
                ln = 2 * (st["n"] - 1) * st["tread"] + st["landing"]
                xa, xb = sorted([st["x0"], st["x0"] + st["dir"] * ln])
                xa, xb = xa - 1.4, xb + 1.4
                core = (XX >= xa) & (XX <= xb) & (np.abs(YY - st["y"]) <= st["width"] / 2 - 0.45)
                grown &= ~(core & ~blocked)
        for r in lay["ramps"]:
            core = (XX >= r["x0"] + 0.45) & (XX <= r["x1"] - 0.45) & (YY >= r["y0"] - 1.0) & (YY <= r["y1"] + 1.0)
            grown &= ~(core & ~blocked)
        self.free = ~grown

    def cell(self, x, y):
        return (min(self.nx - 1, max(0, int(round((x - self.x0) / self.res)))), min(self.ny - 1, max(0, int(round((y - self.y0) / self.res)))))

    def xy(self, c):
        return (self.x0 + c[0] * self.res, self.y0 + c[1] * self.res)

    def nearest_free(self, c, r=15):
        if self.free[c[1], c[0]]:
            return c
        best, bd = c, 1e9
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = c[0] + dx, c[1] + dy
                if 0 <= x < self.nx and 0 <= y < self.ny and self.free[y, x] and dx * dx + dy * dy < bd:
                    best, bd = (x, y), dx * dx + dy * dy
        return best

    def path(self, a, b):
        """the shortest way from a to b (x, y), as points about 0.6 m apart, or None"""
        s, g = self.nearest_free(self.cell(*a)), self.nearest_free(self.cell(*b))
        free, nx, ny = self.free, self.nx, self.ny
        openh = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        nb = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while openh:
            _, c = heapq.heappop(openh)
            if c == g:
                break
            for dx, dy, w in nb:
                n_ = (c[0] + dx, c[1] + dy)
                if not (0 <= n_[0] < nx and 0 <= n_[1] < ny) or not free[n_[1], n_[0]]:
                    continue
                nc = cost[c] + w
                if nc < cost.get(n_, 1e18):
                    cost[n_] = nc
                    came[n_] = c
                    heapq.heappush(openh, (nc + math.hypot(g[0] - n_[0], g[1] - n_[1]), n_))
        if g not in came:
            return None
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = came[c]
        cells.reverse()
        pts = [self.xy(c) for c in cells[::3]] + [self.xy(cells[-1])]
        return self.smooth(pts)

    def line_free(self, a, b, grid=None):
        grid = self.free if grid is None else grid
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.res * 0.5)) + 1
        for i in range(n + 1):
            x = a[0] + (b[0] - a[0]) * i / n
            y = a[1] + (b[1] - a[1]) * i / n
            c = self.cell(x, y)
            if grid is self.free and not grid[c[1], c[0]]:
                return False
            if grid is not self.free and grid[c[1], c[0]]:
                return False
        return True

    def smooth(self, pts):
        """shortcuts: a point is dropped when the way past it is clear"""
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        # (again points every 1 m at most, for the follower)
        dense = [out[0]]
        for p in out[1:]:
            q = dense[-1]
            d = math.hypot(p[0] - q[0], p[1] - q[1])
            k = max(1, int(d / 1.0))
            for i in range(1, k + 1):
                dense.append((q[0] + (p[0] - q[0]) * i / k, q[1] + (p[1] - q[1]) * i / k))
        return dense

    def visible(self, a, b, eye=0.7):
        """a clear line of sight from a to b over what is lower than the robot's eyes"""
        return self.line_free(a, b, self.sight)

    def walls_between(self, a, b):
        """how many blocks of racking or wall a sound goes through (each takes about 6 dB)"""
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / self.res) + 1
        k, inside = 0, False
        for i in range(n + 1):
            c = self.cell(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            s = bool(self.sight[c[1], c[0]])
            if s and not inside:
                k += 1
            inside = s
        return k


def follow(path, i, x, y, yaw, speed, look=1.1):
    """pure pursuit: the forward speed and the turn toward a point look metres along the way; the point reached"""
    while i < len(path) - 1 and math.hypot(path[i][0] - x, path[i][1] - y) < look:
        i += 1
    tx, ty = path[i]
    want = math.atan2(ty - y, tx - x)
    err = math.atan2(math.sin(want - yaw), math.cos(want - yaw))
    d_end = math.hypot(path[-1][0] - x, path[-1][1] - y)
    if abs(err) > 0.9:                                    # (facing away: turn on the spot first)
        return 0.0, max(-0.9, min(0.9, 1.8 * err)), i, d_end
    v = speed * max(0.25, math.cos(err)) * min(1.0, 0.4 + d_end / 1.5)
    return v, max(-0.9, min(0.9, 1.8 * err)), i, d_end


# ── what happens in the warehouse: incidents, people, a forklift, sounds ──────────────────────────────────────────────
# Each incident starts at a time, somewhere; it makes sounds (a level in dB at 1 m, for a while) and may move (a
# person, a forklift, along a way). What it really is decides what the robot should do: a carton fallen from a shelf, a
# dock door rattling, staff driving a forklift (a badge) are logged; a person without a badge in the building is an
# intruder: the robot raises the alarm.
KINDS = {
    "carton": {"what": "a carton fallen from a shelf", "threat": False, "db": 78, "dur": 1.5, "label": "Carton fallen"},
    "door": {"what": "a dock door rattling in the wind", "threat": False, "db": 70, "dur": 6.0, "label": "Dock door rattling"},
    "forklift": {"what": "a forklift driven by staff (badge seen)", "threat": False, "db": 68, "dur": 45.0, "label": "Forklift, staff"},
    "intruder": {"what": "a person without a badge", "threat": True, "db": 52, "dur": 150.0, "label": "INTRUDER"},
    "glass": {"what": "glass broken in the office, a person inside without a badge", "threat": True, "db": 88, "dur": 120.0, "label": "Break-in, office"},
}


def plan_incidents(minutes, seed, fmap, kinds=None, first=(100, 160), gap=(170, 290), threat_from=None):
    """a session's incidents: (as it is) about one every 4 minutes after the first 2, a mix of harmless ones and (from a
    third of the way in) intrusions; (for a demo) closer together; each with where it happens and, for those that
    move, its way"""
    import random
    rng = random.Random(int(seed) * 7919 + 11)
    T = minutes * 60.0
    kinds = kinds or ["carton", "door", "forklift", "intruder", "carton", "glass", "forklift", "intruder", "door", "carton"]
    threat_from = T / 3 if threat_from is None else threat_from
    t = rng.uniform(*first)
    out = []
    racks = WAREHOUSE["racks"]
    docks = WAREHOUSE["docks"]
    i = 0
    while t < T - 60 and i < 40:
        k = kinds[i % len(kinds)]
        if k in ("intruder", "glass") and t < threat_from:
            k = "carton"
        ev = {"id": i, "kind": k, "t0": round(t, 1), **KINDS[k]}
        if k == "carton":
            r = racks[rng.randrange(len(racks))]
            side = rng.choice([-1, 1])
            ev["x"], ev["y"] = rng.uniform(r["x0"] + 0.8, r["x1"] - 0.8), (r["y0"] - 0.45) if side < 0 else (r["y1"] + 0.45)
        elif k == "door":
            d = docks[rng.randrange(len(docks))]
            ev["x"], ev["y"] = d["x"], -4.6
        elif k == "forklift":
            y = rng.choice([4.5, 9.7, 14.9])
            a, b = (rng.choice([[-1.5, y], [26.0, y]]), None)
            b = [26.0, y] if a[0] < 0 else [-1.5, y]
            ev["way"] = [a, b, a]
            ev["x"], ev["y"] = a
            ev["speed"] = 1.6
        elif k == "intruder":
            d = docks[rng.randrange(len(docks))]
            start = [d["x"], -4.3]
            hide = rng.choice([[13.5, 4.5], [7.0, 9.7], [18.0, 14.9], [13.5, 9.7], [4.0, 17.2]])
            way = fmap.path(start, hide) or [start, hide]
            ev["way"] = [list(p) for p in way]
            ev["exit"] = start
            ev["x"], ev["y"] = start
            ev["speed"] = 1.1
        elif k == "glass":
            ev["x"], ev["y"] = 34.0, 18.0
            ev["way"] = [[34.0, 18.0], [32.5, 17.2], [34.5, 16.6], [34.0, 18.0]]
            ev["exit"] = [35.4, 19.4]          # (the broken window)
            ev["speed"] = 0.5
        out.append(ev)
        t += rng.uniform(*gap)
        i += 1
    return out


class Actor:
    """a person, a forklift or a carton on the floor, moving along its way (for incidents that move)"""

    def __init__(self, ev):
        self.ev = ev
        self.kind = {"intruder": "person", "glass": "person", "forklift": "forklift", "carton": "carton", "door": "door"}[ev["kind"]]
        self.x, self.y, self.yaw = float(ev["x"]), float(ev["y"]), 0.0
        self.way = [list(p) for p in ev.get("way", [])]
        self.wi = 1
        self.moving = False
        self.badge = ev["kind"] == "forklift"
        self.gone = False
        self.fleeing = False
        self.wait = 0.0
        self.near_t = 0.0

    def step(self, dt, robot_xy, fmap):
        self.moving = False
        if self.gone or not self.way:
            return
        # (an intruder keeps going while the robot keeps its distance; it runs for the way out when the robot comes
        # close, or when it has been found hiding and watched for 20 s)
        d_r = math.hypot(robot_xy[0] - self.x, robot_xy[1] - self.y)
        self.near_t = self.near_t + dt if (d_r < 5.0 and (self.wait > 1e6 or self.ev["kind"] == "glass")) else 0.0
        if self.ev["kind"] in ("intruder", "glass") and not self.fleeing and (d_r < 2.6 or self.near_t > 20.0):
            self.fleeing = True
            self.wait = 0.0
            ex = self.ev["exit"]
            way = [[self.x, self.y], list(ex)] if self.ev["kind"] == "glass" else ([[self.x, self.y]] + [list(p) for p in (fmap.path((self.x, self.y), ex) or [ex])])
            self.way = way
            self.wi = 1
        if self.wait > 0:
            self.wait -= dt
            return
        if self.wi >= len(self.way):
            if self.fleeing:
                self.gone = True
            elif self.ev["kind"] in ("forklift", "glass"):
                self.wi = 1
                self.way = self.way[::-1]
            return
        tx, ty = self.way[self.wi]
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        sp = self.ev.get("speed", 1.0) * (1.6 if self.fleeing else 1.0)
        if d < 0.05:
            self.wi += 1
            if self.ev["kind"] == "intruder" and not self.fleeing and self.wi >= len(self.way):
                self.wait = 1e9            # (hiding: still, crouched)
            return
        step = min(d, sp * dt)
        self.x += dx / d * step
        self.y += dy / d * step
        want = math.atan2(dy, dx)
        self.yaw += math.atan2(math.sin(want - self.yaw), math.cos(want - self.yaw)) * min(1.0, 6 * dt)
        self.moving = True

    def item(self):
        return {"id": self.ev["id"], "kind": self.kind, "x": round(self.x, 2), "y": round(self.y, 2), "yaw": round(self.yaw, 2),
                "moving": bool(self.moving), "badge": self.badge, "crouch": bool(self.wait > 1e6)}


def heard(src_xy, db, robot_xy, fmap):
    """the level (dB) a sound of db (at 1 m) reaches the robot with: spreading, and about 6 dB for each block of racking or
    wall it goes through"""
    d = max(1.0, math.hypot(src_xy[0] - robot_xy[0], src_xy[1] - robot_xy[1]))
    return db - 20.0 * math.log10(d) - 6.0 * fmap.walls_between(robot_xy, src_xy)


def seen(target_xy, robot_xy, robot_yaw, fmap, fov=math.radians(110), rng=14.0):
    """in the robot's cameras: within its field of view and range, with a clear line of sight"""
    dx, dy = target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1]
    d = math.hypot(dx, dy)
    if d > rng:
        return False
    if d > 0.6:
        err = math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw), math.cos(math.atan2(dy, dx) - robot_yaw))
        if abs(err) > fov / 2:
            return False
    return fmap.visible(robot_xy, target_xy)


# ── the robot: Boston Dynamics Spot (MuJoCo Menagerie) in EsaSim's conventions ───────────────────────────────────────
# Its root body is "pelvis" (Spot's body), its legs fl, fr, hl, hr (front/hind, left/right), each a hip abduction (hx),
# a hip swing (hy) and a knee (kn); its feet the spheres FL, FR, HL, HR, with sites foot_FL… and touch sensors; the IMU
# site at its centre, a camera site at its front. Collision shapes are primitives (a box, capsules, the feet); its 3D
# look in the view comes from the Menagerie meshes (models/robot-spot.js).
def _xml(z):
    import base64
    import zlib
    return zlib.decompress(base64.b64decode(z)).decode()


ROBOTS = {
    "spot": {"name": "Boston Dynamics Spot", "short": "Spot", "xml": lambda: _xml(SPOT_XML_Z), "h0": 0.43, "view": "robot-spot",
             "camera": {"body": "pelvis", "pos": [0.44, 0.0, 0.03], "pitch": 0.35, "fov_h": 100.0, "aspect": 1.7778, "name": "its front stereo cameras"},
             "credit": "Boston Dynamics Spot: MuJoCo Menagerie (Clearpath Robotics, BSD-3)"},
}
ROBOT_OF = {"boston dynamics spot": "spot", "spot": "spot", "spot mini": "spot"}
ROBOT = "spot"
NU = 12
H0 = 0.43                 # its body's height standing (under its own weight)
FEET_GEOMS = ["FL", "FR", "HL", "HR"]
LEGS = ["fl", "fr", "hl", "hr"]
MODEL_IN_USE = None
MODEL_XML = None


def robot_id(p):
    return ROBOT_OF.get(str(p.get("robot") or "Boston Dynamics Spot").strip().lower(), "spot")


def use_robot(rid, mujoco):
    global ROBOT, NU, H0, OBS_BASE, PRIV_BASE, OBS_DIM, PRIV_DIM, MODEL_IN_USE, FIELDS
    ROBOT, R = rid, ROBOTS[rid]
    MODEL_IN_USE = R["xml"]()
    m = mujoco.MjModel.from_xml_string(MODEL_IN_USE)
    NU, H0 = int(m.nu), float(R["h0"])
    OBS_BASE = 3 + 3 + 3 + 3 + NU + NU + NU + 4
    PRIV_BASE = OBS_BASE + 1 + 4 + 4 + 12 + 3 + 3
    OBS_DIM, PRIV_DIM = OBS_BASE + N_SCAN, PRIV_BASE + N_SCAN
    FIELDS = make_fields(NU)
    return R


def hs(v):
    """a height given for the humanoid G1 (its pelvis 0.755 m up), for this robot"""
    return v * H0 / 0.755


# ── the physics: MuJoCo, many robots at once ──────────────────────────────────────────────────────────────────────
# Two ways to run the same model: MuJoCo Warp on the NVIDIA GPU (thousands of robots, the whole step on the card, one
# CUDA graph per control step) or MuJoCo on the CPU (one simulation per robot, spread over the cores). Both give the
# same views as PyTorch tensors: qpos, qvel, ctrl and the sensors, on the device the learning runs on.
CTRL_DT, SIM_DT = 0.02, 0.004           # the policy at 50 Hz, the physics at 250 Hz
N_SUB = int(round(CTRL_DT / SIM_DT))


# ── the ground: a rectangular circuit (x along its first side from the start line, y to the left, metres) ──────────
# The robot follows the circuit's line round a 38 x 18 m rectangle with rounded corners (2.5 m), counterclockwise:
#   side 1 (y = 0, x from -4 to 34): a 4 x 4 m square of bumps (0-4 cm) at 4-8 m · a 12.5 % slope (7°) up to a 0.5 m
#     plateau at 14-17 m and down to 21 m · a 4 x 4 m square of blocks (0.4 m tiles, 0-5 cm) at 24-28 m
#   side 3 (y = 18, walked toward -x): a staircase 3 m wide, 10 steps up (12 cm, 30 cm deep) from x = 22, a 2 m
#     landing 1.2 m up, 10 steps down
#   side 2 (x = 34): a 4 x 4 m square of 64 moving tiles · side 4 (x = -4, walked toward -y): a 4 x 4 m square of
#     smooth ice (friction 0.08, its top 6 mm up: a heightfield of its own that only the feet touch) at y = 7-11
# Inside it, a training yard: four staircases of 10 steps up and 10 down, with 6, 9, 12 and 15 cm risers.
# The same few numbers go to the 3D view, which builds the same ground from them. In MuJoCo: one heightfield, 5 cm a
# point (the stairs' edges on its points).
HF_X0, HF_X1, HF_Y0, HF_Y1, HF_RES, HF_BASE = -8.0, 38.0, -7.0, 22.0, 0.05, 0.05
CIRCUIT = {"x0": -4.0, "x1": 34.0, "y0": 0.0, "y1": 18.0, "r": 2.5}


def make_path(c=CIRCUIT, ds=0.1):
    """the circuit's line: points every ds metres, counterclockwise from the start line (0, 0); heading and curvature"""
    import numpy as np
    x0, x1, y0, y1, r = c["x0"], c["x1"], c["y0"], c["y1"], c["r"]
    pts = []
    def line(a, b):
        for t in np.linspace(0, 1, 400, endpoint=False):
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    def arc(cx, cy, a0):
        for t in np.linspace(0, 1, 200, endpoint=False):
            a = a0 + t * np.pi / 2
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    line((x0 + r, y0), (x1 - r, y0)); arc(x1 - r, y0 + r, -np.pi / 2)
    line((x1, y0 + r), (x1, y1 - r)); arc(x1 - r, y1 - r, 0.0)
    line((x1 - r, y1), (x0 + r, y1)); arc(x0 + r, y1 - r, np.pi / 2)
    line((x0, y1 - r), (x0, y0 + r)); arc(x0 + r, y0 + r, np.pi)
    P = np.array(pts)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    S = np.concatenate([[0.0], np.cumsum(seg)])
    L = S[-1]
    s0 = np.interp(-x0 - r, [0, x1 - x0 - 2 * r], [0, x1 - x0 - 2 * r])    # the start line: x = 0 on side 1
    s = (s0 + np.arange(0, L, ds)) % L
    Pc = np.vstack([P, P[:1]])
    Q = np.stack([np.interp(s, S, Pc[:, 0]), np.interp(s, S, Pc[:, 1])], -1)
    d = np.roll(Q, -1, axis=0) - np.roll(Q, 1, axis=0)
    head = np.arctan2(d[:, 1], d[:, 0])
    dh = np.angle(np.exp(1j * (np.roll(head, -1) - np.roll(head, 1))))
    curv = dh / (2 * ds)
    return Q, head, curv, float(L)


def make_course():
    import numpy as np
    r = np.random.default_rng(20261112)
    coarse = r.uniform(-1, 1, (11, 11))                       # bumps: a 0.4 m swell and a 0.1 m grain
    xi = np.linspace(0, 10, 41)
    up = np.array([[np.interp(x, np.arange(11), row) for x in xi] for row in coarse])
    up = np.array([np.interp(xi, np.arange(11), up[:, j]) for j in range(41)]).T
    bumps = 16 * up + r.uniform(-6, 6, (41, 41))
    bumps = np.round(bumps - bumps.min()).astype(int)            # (all above the floor: 0 to about 4 cm)
    blocks = r.integers(0, 51, (10, 10))
    stairs = [{"x0": 22.0, "y": 18.0, "dir": -1, "n": 10, "rise": 0.12, "tread": 0.3, "landing": 2.0, "width": 3.0, "name": "circuit"}]
    for k, rise in enumerate((0.06, 0.09, 0.12, 0.15)):          # the training yard, inside the circuit
        stairs.append({"x0": 2.0, "y": round(6.6 + 2.2 * k, 2), "dir": 1, "n": 10, "rise": rise, "tread": 0.3, "landing": 1.2, "width": 1.8, "name": "yard"})
    Q, head, curv, L = make_path()
    k4 = max(1, int(round(0.25 / 0.1)))
    tiles = {"x0": 32.0, "y0": 7.0, "cell": 0.5, "n": 8, "half": 0.15, "amp": [0.01, 0.04], "base": [0.0, 0.03], "period": [1.5, 4.0]}
    return {"ice": {"x0": -6.0, "y0": 7.0, "size": 4.0, "top": 0.006, "mu": 0.08},
            "ramp": {"x": [10.0, 14.0, 17.0, 21.0], "h": 0.5, "half": 3.0, "bank": 1.5},
            "patches": [{"kind": "bumps", "x0": 4.0, "y0": -2.0, "size": 4.0, "cell": 0.1, "n": 41, "mm": bumps.ravel().tolist()},
                        {"kind": "blocks", "x0": 24.0, "y0": -2.0, "size": 4.0, "cell": 0.4, "n": 10, "mm": blocks.ravel().tolist()}],
            "stairs": stairs, "tiles": tiles, "circuit": dict(CIRCUIT, length=round(L, 2)),
            "path": [[round(float(x), 3), round(float(y), 3)] for x, y in Q[::k4]]}


COURSE = make_course()
PATH = make_path()              # (points every 10 cm, heading, curvature, length)


# ── the playground: an open floor with things to walk over, driven by hand (Walk, Ground: Playground) ─────────────
# From the start (1.5, 9), facing +x: gravel (0-2 cm) at x 6-10, y 2-6 and stepping stones (0.4 m, 0-8 cm) at
# x 6-10, y 12-16 · a sandbox (5 x 5 m, its sand 3 cm over the floor, soft: the feet sink in) at x 13-18, y 2-7 and a
# puddle (5 x 5 m, 10 cm of water that drags at the legs) at x 13-18, y 11-16 · ice (4 x 4 m) at x 21-25, y 2-6 and
# two curbs (6 and 10 cm) across x 21.5 and 23.5, y 11-16 · steps (4 up of 10 cm, a landing, 4 down) from x 27, y 7.5-10.5.
def make_playground():
    import numpy as np
    r = np.random.default_rng(20261113)
    gravel = np.round(r.uniform(0, 20, (41, 41))).astype(int)               # (mm: grains of up to 2 cm)
    stones = (r.uniform(0, 1, (10, 10)) < 0.55) * r.integers(30, 81, (10, 10))  # (some 0.4 m slabs raised 3-8 cm, the rest flush)
    return {"kind": "playground", "ramp": None,
            "patches": [{"kind": "bumps", "x0": 6.0, "y0": 2.0, "size": 4.0, "cell": 0.1, "n": 41, "mm": gravel.ravel().tolist(), "label": "GRAVEL  0–2 cm"},
                        {"kind": "blocks", "x0": 6.0, "y0": 12.0, "size": 4.0, "cell": 0.4, "n": 10, "mm": stones.ravel().tolist(), "label": "STEPPING STONES  3–8 cm"}],
            "stairs": [{"x0": 21.5, "y": 13.5, "dir": 1, "n": 1, "rise": 0.06, "tread": 0.3, "landing": 0.3, "width": 5.0, "name": "curb", "label": "CURB  6 cm", "label_at": [21.65, 10.5]},
                       {"x0": 23.5, "y": 13.5, "dir": 1, "n": 1, "rise": 0.10, "tread": 0.3, "landing": 0.3, "width": 5.0, "name": "curb", "label": "CURB  10 cm", "label_at": [23.65, 16.5]},
                       {"x0": 27.0, "y": 9.0, "dir": 1, "n": 4, "rise": 0.10, "tread": 0.3, "landing": 1.5, "width": 3.0, "name": "steps"}],
            "ice": {"x0": 21.0, "y0": 2.0, "size": 4.0, "top": 0.006, "mu": 0.08},
            "sand": {"x0": 13.0, "y0": 2.0, "size": 5.0, "top": 0.03, "mu": 0.55},
            "water": {"x0": 13.0, "y0": 11.0, "size": 5.0, "depth": 0.10, "drag": 1.5},
            "start": [1.5, 9.0, 0.0], "area": [-2.0, 32.0, -1.0, 19.0]}


PLAYGROUND = make_playground()


def course_of(terrain):
    """the ground's layout for a terrain: the circuit's, the playground's, the warehouse's, or none (the flat floor)"""
    return COURSE if terrain == "circuit" else PLAYGROUND if terrain == "playground" else WAREHOUSE if terrain == "warehouse" else None


def stair_profile(u, st):
    """a staircase's height at distance u (m) from its first riser, along the way it climbs"""
    import numpy as np
    n, r, t, L = st["n"], st["rise"], st["tread"], st["landing"]
    up = (n - 1) * t
    return np.where(u < 0, 0.0, np.where(u < up, (np.floor(u / t + 1e-6) + 1) * r, np.where(u < up + L, n * r,
                    np.where(u < 2 * up + L, (n - 1 - np.floor((u - up - L) / t + 1e-6)) * r, 0.0))))


def course_height(X, Y, course=COURSE):
    """the ground's height (m) at X, Y (numpy arrays)"""
    import numpy as np
    X, Y = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))
    H = np.zeros(X.shape)
    rp = course.get("ramp")
    if rp:
        xs = rp["x"]
        prof = np.interp(X, xs, [0.0, rp["h"], rp["h"], 0.0], left=0.0, right=0.0)
        H += prof * np.clip((rp["half"] + rp["bank"] - np.abs(Y)) / rp["bank"], 0.0, 1.0)
    for pa in course["patches"]:
        n, c = pa["n"], pa["cell"]
        g = np.array(pa["mm"], dtype=float).reshape(n, n) / 1000.0        # [row along y, column along x]
        u, v = (X - pa["x0"]) / c, (Y - pa["y0"]) / c
        if pa["kind"] == "bumps":
            inside = (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1)
            u0 = np.clip(np.floor(u), 0, n - 2).astype(int); v0 = np.clip(np.floor(v), 0, n - 2).astype(int)
            fu, fv = np.clip(u - u0, 0, 1), np.clip(v - v0, 0, 1)
            h = (g[v0, u0] * (1 - fu) * (1 - fv) + g[v0, u0 + 1] * fu * (1 - fv) + g[v0 + 1, u0] * (1 - fu) * fv + g[v0 + 1, u0 + 1] * fu * fv)
        else:
            inside = (u >= 0) & (u < n) & (v >= 0) & (v < n)
            h = g[np.clip(np.floor(v), 0, n - 1).astype(int), np.clip(np.floor(u), 0, n - 1).astype(int)]
        H = H + np.where(inside, h, 0.0)
    for st in course.get("stairs", []):
        inside = np.abs(Y - st["y"]) <= st["width"] / 2 + 1e-6
        H = H + np.where(inside, stair_profile((X - st["x0"]) * st["dir"], st), 0.0)
    for rp_ in course.get("ramps", []):          # straight ramps between two heights, along x or along y
        u = ((X - rp_["x0"]) / (rp_["x1"] - rp_["x0"])) if rp_.get("axis", "x") == "x" else ((Y - rp_["y0"]) / (rp_["y1"] - rp_["y0"]))
        inside = (X >= rp_["x0"]) & (X <= rp_["x1"]) & (Y >= rp_["y0"]) & (Y <= rp_["y1"])
        H = np.where(inside, np.maximum(H, rp_["h0"] + (rp_["h1"] - rp_["h0"]) * np.clip(u, 0.0, 1.0)), H)
    for b in course.get("boxes", []):            # solid blocks (walls, racks, a platform, pallets): their tops
        inside = (X >= b["x0"]) & (X <= b["x1"]) & (Y >= b["y0"]) & (Y <= b["y1"])
        H = np.where(inside, np.maximum(H, b["h"]), H)
    return H


def ground_grid(terrain):
    """the heightfield's heights (rows along y, columns along x)"""
    import numpy as np
    xs = np.linspace(HF_X0, HF_X1, int(round((HF_X1 - HF_X0) / HF_RES)) + 1)
    ys = np.linspace(HF_Y0, HF_Y1, int(round((HF_Y1 - HF_Y0) / HF_RES)) + 1)
    c = course_of(terrain)
    if c is None:
        return np.zeros((len(ys), len(xs)))
    X, Y = np.meshgrid(xs, ys)
    return course_height(X, Y, c)


def tiles_xml(t=None):
    """the moving tiles: one box each, moved up and down by the task (mocap bodies; only the feet touch them)"""
    t = t or COURSE["tiles"]
    out = []
    for j in range(t["n"]):
        for i in range(t["n"]):
            x, y = t["x0"] + (i + 0.5) * t["cell"], t["y0"] + (j + 0.5) * t["cell"]
            out.append('<body name="tile_%d_%d" mocap="true" pos="%.3f %.3f %.3f"><geom type="box" size="%.3f %.3f %.3f" contype="2" conaffinity="0" '
                       'condim="3" friction="0.8 0.02 0.01" group="1" /></body>' % (i, j, x, y, -t["half"], t["cell"] / 2 - 0.005, t["cell"] / 2 - 0.005, t["half"]))
    return "".join(out)


def ice_xml(xml, terrain, c=None):
    """the ice: a flat heightfield a few millimetres up that only the feet touch (contact pairs of their own, with the
    ice's low friction), the same kind of contact as the course's ground; on the flat floor, none"""
    c = c or (course_of(terrain) or {}).get("ice")
    if not c:
        return xml
    half = c["size"] / 2
    n = int(round(c["size"] / 0.1)) + 1           # (a point every 10 cm: small cells make good contacts)
    xml = xml.replace("</asset>", '<hfield name="ice" nrow="%d" ncol="%d" size="%.3f %.3f %.4f 0.01" /></asset>' % (n, n, half, half, c["top"]), 1)   # (its heights: set by load_model)
    xml = xml.replace("</worldbody>", '<geom name="ice" type="hfield" hfield="ice" pos="%.3f %.3f 0" contype="0" conaffinity="0" group="1" /></worldbody>'
                      % (c["x0"] + half, c["y0"] + half), 1)
    pairs = "".join('<pair name="%s_ice" geom1="%s" geom2="ice" condim="3" friction="%.3f %.3f" />' % (f, f, c["mu"], c["mu"]) for f in FEET_GEOMS)
    return xml.replace("</contact>", pairs + "</contact>", 1)


def sand_xml(xml, terrain):
    """the sand: a flat heightfield its depth over the floor that only the feet touch, soft (a slow, damped contact
    that lets them sink in, harder the deeper they go), the floor under it; only on the playground"""
    c = (course_of(terrain) or {}).get("sand")
    if not c:
        return xml
    half = c["size"] / 2
    n = int(round(c["size"] / 0.1)) + 1
    xml = xml.replace("</asset>", '<hfield name="sand" nrow="%d" ncol="%d" size="%.3f %.3f %.4f 0.01" /></asset>' % (n, n, half, half, c["top"]), 1)
    xml = xml.replace("</worldbody>", '<geom name="sand" type="hfield" hfield="sand" pos="%.3f %.3f 0" contype="0" conaffinity="0" group="1" /></worldbody>'
                      % (c["x0"] + half, c["y0"] + half), 1)
    pairs = "".join('<pair name="%s_sand" geom1="%s" geom2="sand" condim="3" friction="%.3f %.3f" solref="%s" solimp="%s" />' % (f, f, c["mu"], c["mu"], SAND_SOLREF, SAND_SOLIMP)
                    for f in FEET_GEOMS)
    return xml.replace("</contact>", pairs + "</contact>", 1)


SAND_SOLREF, SAND_SOLIMP = "0.06 1.2", "0.2 0.95 0.03 0.5 2"


def load_model(mujoco, terrain="circuit"):
    import numpy as np
    m = mujoco.MjModel.from_xml_string(sand_xml(ice_xml((MODEL_IN_USE or MODEL_XML).replace("</worldbody>", tiles_xml() + "</worldbody>"), terrain), terrain))
    m.opt.timestep = SIM_DT
    H = ground_grid(terrain)
    hid = m.hfield("terrain").id
    assert H.shape == (int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid]))
    top = float(m.hfield_size[hid][2])
    a = int(m.hfield_adr[hid])
    m.hfield_data[a:a + H.size] = np.clip((H + HF_BASE) / top, 0.0, 1.0).ravel().astype(np.float32)
    for nm in ("ice", "sand"):          # the ice and the sand: level, at their full height
        if (course_of(terrain) or {}).get(nm):
            ih = m.hfield(nm).id
            ia = int(m.hfield_adr[ih])
            m.hfield_data[ia:ia + int(m.hfield_nrow[ih]) * int(m.hfield_ncol[ih])] = 1.0
    if course_of(terrain) is None:      # the flat floor only: the course's heightfield out of the way (fewer contacts)
        m.geom_pos[m.geom("terrain").id][2] = -5.0
    return m


class WarpSim:
    """MuJoCo Warp: n worlds on the GPU"""

    def __init__(self, mjm, n, device="cuda:0", log=None):
        import warp as wp
        import mujoco_warp as mjw
        import torch
        wp.config.quiet = True
        wp.init()
        self.wp, self.mjw, self.dev, self.torch = wp, mjw, device, torch
        self.n = n
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(mjm)
            try:
                self.m.opt.warn_overflow = 0
            except Exception:
                pass
            self.d = mjw.make_data(mjm, nworld=n, nconmax=96, njmax=480)    # (room for the contacts on rough ground)
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.sens = wp.to_torch(self.d.sensordata)
        self.tq = wp.to_torch(self.d.actuator_force)         # each motor's torque (N m)
        self.mocap = wp.to_torch(self.d.mocap_pos) if mjm.nmocap else None      # the moving tiles' places (nworld, ntiles, 3)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)                          # forces applied to bodies (nworld, nbody, 6: force, torque)
        self.tdev = self.qpos.device
        self._qacc = wp.to_torch(self.d.qacc_warmstart) if hasattr(self.d, "qacc_warmstart") else None
        self.graph = None
        self.cuda = str(device).startswith("cuda")
        t = __import__("time").time()
        with wp.ScopedDevice(device):
            self.mjw.forward(self.m, self.d)
            for _ in range(2):
                self.mjw.step(self.m, self.d)          # (the kernels compile here, the first time on a machine)
            wp.synchronize()
            if self.cuda:
                with wp.ScopedCapture() as cap:
                    for _ in range(N_SUB):
                        self.mjw.step(self.m, self.d)
                self.graph = cap.graph
        if log:
            log("MuJoCo Warp ready on %s: %d robots (kernels built in %.0f s)." % (device, n, __import__("time").time() - t))
        torch.cuda.synchronize() if self.cuda else None

    def step(self):
        wp = self.wp
        if self.cuda:
            self.torch.cuda.synchronize()        # (what PyTorch wrote into the arrays first, then the physics)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            with wp.ScopedDevice(self.dev):
                for _ in range(N_SUB):
                    self.mjw.step(self.m, self.d)
        if self.cuda:
            wp.synchronize_device(self.dev)

    def kick(self, dv):
        """add dv (n, 2) to the robots' horizontal speed: a push"""
        self.qvel[:, :2] += dv.to(self.qvel.dtype)

    def set_tiles(self, z):
        """the moving tiles' centres' heights (n, ntiles), for the next step"""
        if self.mocap is not None:
            self.mocap[:, :, 2] = z.to(self.mocap.dtype)

    def set_lift(self, body, fz):
        """an upward force fz (n, newtons) on one body of each robot, until changed (a help while it learns to stand up)"""
        self.xfrc[:, body, 2] = fz.to(self.xfrc.dtype)

    def set_rows(self, idx, qpos, qvel, ctrl):
        """robots idx start again from qpos, qvel (their motors aimed at ctrl)"""
        self.qpos[idx] = qpos.to(self.qpos.dtype)
        self.qvel[idx] = qvel.to(self.qvel.dtype)
        self.ctrl[idx] = ctrl.to(self.ctrl.dtype)
        if self._qacc is not None:
            self._qacc[idx] = 0.0

    def refresh(self):
        """the sensors after set_rows (they are computed by the step; a forward pass brings them up to date)"""
        if self.cuda:
            self.torch.cuda.synchronize()
        with self.wp.ScopedDevice(self.dev):
            self.mjw.forward(self.m, self.d)
        if self.cuda:
            self.wp.synchronize_device(self.dev)

    def sync(self):
        if self.cuda:
            self.wp.synchronize()


class CpuSim:
    """MuJoCo on the CPU: one simulation per robot, the robots shared out over threads (MuJoCo releases the GIL)"""

    def __init__(self, mjm, n, threads=None, log=None):
        import os
        import mujoco
        import numpy as np
        import torch
        from concurrent.futures import ThreadPoolExecutor
        self.mj, self.np, self.torch = mujoco, np, torch
        self.m = mjm
        self.n = n
        self.datas = [mujoco.MjData(mjm) for _ in range(n)]
        nt = max(1, min(n, threads or (os.cpu_count() or 2)))
        self.ex = ThreadPoolExecutor(nt)
        self.parts = [list(range(i * n // nt, (i + 1) * n // nt)) for i in range(nt)]
        self.qpos_np = np.zeros((n, mjm.nq))
        self.qvel_np = np.zeros((n, mjm.nv))
        self.ctrl_np = np.zeros((n, mjm.nu))
        self.sens_np = np.zeros((n, mjm.nsensordata))
        self.tq_np = np.zeros((n, mjm.nu))
        self.tq = torch.zeros(n, mjm.nu)
        self.tile_z = None
        self.lift = None             # (body, forces) an upward help on one body
        self.pushes = {}             # robot: [body, point in the body's frame, force (N, world), substeps left] (a push by hand)
        self.water = None            # a puddle: {x0, y0, size, depth, drag}: the water pulls back on the parts of the legs in it
        self._wv = np.zeros(6)
        self._wet = {}               # robot: the bodies the water pulls on now
        self.qpos = torch.zeros(n, mjm.nq)
        self.qvel = torch.zeros(n, mjm.nv)
        self.ctrl = torch.zeros(n, mjm.nu)
        self.sens = torch.zeros(n, mjm.nsensordata)
        self.tdev = torch.device("cpu")
        if log:
            log("MuJoCo on the CPU: %d robots over %d threads." % (n, nt))

    def _work(self, part):
        mj, m, D = self.mj, self.m, self.datas
        tz, lift = self.tile_z, self.lift
        for i in part:
            d = D[i]
            d.ctrl[:] = self.ctrl_np[i]
            if tz is not None:
                d.mocap_pos[:, 2] = tz[i]
            if lift is not None:
                d.xfrc_applied[lift[0], 2] = lift[1][i]
            pu = self.pushes.get(i)
            if self.water is not None:
                self._wade(i, d)
            if pu is None:
                for _ in range(N_SUB):
                    mj.mj_step(m, d)
            else:                     # a push: the force at the point pushed (it moves with the body), its turning effect about the body's centre
                b, loc, F = pu[0], pu[1], pu[2]
                for _ in range(N_SUB):
                    if pu[3] > 0:
                        pw = d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc
                        d.xfrc_applied[b, :3] = F
                        d.xfrc_applied[b, 3:] = self.np.cross(pw - d.xipos[b], F)
                        pu[3] -= 1
                    elif pu[3] == 0:
                        d.xfrc_applied[b, :] = 0.0
                        pu[3] = -1
                    mj.mj_step(m, d)
                if pu[3] < 0:
                    d.xfrc_applied[b, :] = 0.0
                    self.pushes.pop(i, None)
            self.qpos_np[i] = d.qpos
            self.qvel_np[i] = d.qvel
            self.sens_np[i] = d.sensordata
            self.tq_np[i] = d.actuator_force

    def step(self):
        self.ctrl_np[:] = self.ctrl.numpy()
        list(self.ex.map(self._work, self.parts))
        self._pull()

    def _pull(self):
        self.qpos.copy_(self.torch.from_numpy(self.qpos_np))
        self.qvel.copy_(self.torch.from_numpy(self.qvel_np))
        self.sens.copy_(self.torch.from_numpy(self.sens_np))
        self.tq.copy_(self.torch.from_numpy(self.tq_np))

    def set_rows(self, idx, qpos, qvel, ctrl):
        qp, qv, cu = qpos.double().numpy(), qvel.double().numpy(), ctrl.double().numpy()
        for k, i in enumerate(idx.tolist()):
            d = self.datas[i]
            self.mj.mj_resetData(self.m, d)
            d.qpos[:] = qp[k]
            d.qvel[:] = qv[k]
            d.ctrl[:] = cu[k]
            self.mj.mj_forward(self.m, d)
            self.qpos_np[i] = d.qpos
            self.qvel_np[i] = d.qvel
            self.sens_np[i] = d.sensordata
            self.ctrl_np[i] = cu[k]
        self.ctrl[idx] = ctrl.float()
        self._pull()

    def refresh(self):
        pass

    def _wade(self, i, d):
        """the water's drag on the feet in it (quadratic, on what of each is under: a 0.02 m² face per foot, times the
        drag setting), set once per control step (20 ms)"""
        w, np_, mj = self.water, self.np, self.mj
        P = d.xipos
        if not hasattr(self, "_feet"):
            self._feet = np_.array([int(self.m.geom_bodyid[self.m.geom(f).id]) for f in FEET_GEOMS])
        F = self._feet                   # (the feet: what is in 10 cm of water)
        under = F[(P[F, 0] >= w["x0"]) & (P[F, 0] <= w["x0"] + w["size"]) & (P[F, 1] >= w["y0"]) & (P[F, 1] <= w["y0"] + w["size"]) & (P[F, 2] < w["depth"] + 0.05)]
        if i in self._wet:
            d.xfrc_applied[self._wet[i], :] = 0.0
        idx = under
        for b in idx:
            mj.mj_objectVelocity(self.m, d, mj.mjtObj.mjOBJ_BODY, int(b), self._wv, 0)
            v = self._wv[3:]
            frac = min(1.0, max(0.0, (w["depth"] + 0.05 - P[b, 2]) / 0.1))
            d.xfrc_applied[b, :3] = -0.5 * 1000.0 * 0.02 * w["drag"] * frac * np_.linalg.norm(v) * v
        self._wet[i] = idx

    def push_at(self, i, body, local, force, seconds=0.1):
        """push robot i by hand: force (N, world) at a point of a body (in its frame), for a moment"""
        self.pushes[i] = [int(body), self.np.asarray(local, dtype=float), self.np.asarray(force, dtype=float), max(1, int(round(seconds / SIM_DT)))]

    def kick(self, dv):
        d_ = dv.double().numpy()
        for i in (abs(d_).sum(-1) > 0).nonzero()[0].tolist():
            self.datas[i].qvel[0:2] += d_[i]
            self.qvel_np[i] = self.datas[i].qvel
        self._pull()

    def set_tiles(self, z):
        self.tile_z = z.double().numpy() if self.m.nmocap else None

    def set_lift(self, body, fz):
        self.lift = (int(body), fz.double().numpy())

    def sync(self):
        pass


def make_sim(mjm, n, want, log=None):
    """(sim, label): the GPU through MuJoCo Warp when it can, else the CPU"""
    import torch
    w = str(want or "auto").lower()
    if w.startswith("nvidia") or w == "auto":
        err = None
        if torch.cuda.is_available():
            try:
                sim = WarpSim(mjm, n, "cuda:0", log)
                return sim, "%s (MuJoCo Warp)" % torch.cuda.get_device_name(0)
            except ImportError as e:
                err = "MuJoCo Warp is not installed on this worker (pip install mujoco-warp): %s" % e
            except Exception as e:      # noqa: BLE001
                err = "MuJoCo Warp could not start on the GPU: %s" % e
        else:
            err = "this PyTorch cannot use an NVIDIA GPU (install the CUDA build: pip install torch --index-url https://download.pytorch.org/whl/cu124)"
        if w.startswith("nvidia"):
            raise RuntimeError("NVIDIA GPU asked, but " + err)
        if log:
            log("On the CPU instead: " + err)
    n = min(n, 512)
    return CpuSim(mjm, n, log=log), "CPU (MuJoCo, %d robots)" % n


# ── the task: walk where asked (a speed ahead, a speed aside, a turn), on any ground ─────────────────────────────────
# What the policy sees (as the robot itself could): its speed and turning (from the IMU in its body), where gravity
# points, the command, its 12 joint angles and speeds, what it did last, a clock for the gait (a trot: the diagonal
# pairs of legs, front-left with hind-right and front-right with hind-left, half a step apart), and an elevation map of
# the ground around and ahead of it (as its cameras give it). What it does: 12 joint targets around its standing pose
# (the motors' own PD loops follow them). To walk a way it follows a line (or the circuit's), a guide turning it back
# onto it; in training the commands also come at random, so it learns to turn on the spot, to step sideways, to stand.
# The rewards (as legged_gym and MuJoCo Playground have them for quadrupeds): follow the speed and turn, keep the body
# level and steady, trot (feet in the air in turn, lifted, no slipping), stay near the standing pose, no falling.
import math

ACTION_SCALE = 0.3
OBS_BASE = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 4      # what it senses of itself (52)
PRIV_BASE = OBS_BASE + 1 + 4 + 4 + 12 + 3 + 3     # and what only the critic knows (79)
SCAN_X = [-0.6 + 0.15 * i for i in range(14)]     # the elevation map: 14 x 5 points from 0.6 m behind its centre to 1.35 m ahead
SCAN_Y = [-0.3, -0.15, 0.0, 0.15, 0.3]
N_SCAN = len(SCAN_X) * len(SCAN_Y)
OBS_DIM = OBS_BASE + N_SCAN
PRIV_DIM = PRIV_BASE + N_SCAN
EP_STEPS = 1000                      # 20 s
SENSORS = ["local_linvel", "gyro", "upvector", "global_linvel", "global_angvel", "accel"] + \
    ["%s_%s" % (f, k) for f in ("FL", "FR", "HL", "HR") for k in ("pos", "vel", "touch")]
SCALES = {"tracking_lin_vel": 1.5, "tracking_ang_vel": 0.8, "lin_vel_z": -2.0, "ang_vel_xy": -0.05, "orientation": -5.0, "alive": 0.5,
          "feet_air_time": 1.0, "feet_slip": -0.1, "feet_phase": 0.5, "stand_still": -0.5, "termination": -100.0, "collision": -1.0,
          "torques": -2e-5, "action_rate": -0.01, "dof_pos_limits": -1.0, "pose": -0.05, "hip_abd": -0.2}
PAIR = [0, 1, 1, 0]                  # each foot's place in the trot: FL and HR together, FR and HL together


def quat_rotate_inv(q, v):
    """v (world) into the frame of the unit quaternions q (w, x, y, z)"""
    import torch
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def yaw_of(q):
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rz_bezier(phi, h):
    """the wanted foot height along the gait cycle (lift and set down smoothly)"""
    import torch
    x = (phi + math.pi) / (2 * math.pi)
    bez = lambda a, b, s: a + (b - a) * (s ** 3 + 3 * (s ** 2 * (1 - s)))   # noqa: E731
    return torch.where(x <= 0.5, bez(0.0, h, 2 * x), bez(h, 0.0, 2 * x - 1))


class WalkEnv:
    def __init__(self, sim, mjm, seed=0, vmax=1.2, push=1.0, noise=1.0, train=True, terrain="circuit", scan=True):
        import torch
        self.sim, self.m, self.n = sim, mjm, sim.n
        dev = self.dev = sim.tdev
        self.g = torch.Generator(device=dev).manual_seed(seed)
        f = dict(device=dev, dtype=torch.float32)
        key = mjm.key("home")
        self.q0 = torch.tensor(key.qpos, **f)
        self.default = self.q0[7:].clone()
        self.vmax, self.push_max, self.noise, self.train = vmax, push, noise, train
        lo, hi = torch.tensor(mjm.jnt_range[1:, 0], **f), torch.tensor(mjm.jnt_range[1:, 1], **f)
        c, r = (lo + hi) / 2, hi - lo
        self.soft_lo, self.soft_hi = c - 0.5 * r * 0.95, c + 0.5 * r * 0.95
        names = [mjm.joint(i).name for i in range(1, mjm.njnt)]
        self.hx_i = torch.tensor([i for i, nm in enumerate(names) if nm.endswith("_hx")], device=dev)
        self.sadr = {}
        for s_ in SENSORS:
            sid = mjm.sensor(s_).id
            a, d = int(mjm.sensor_adr[sid]), int(mjm.sensor_dim[sid])
            self.sadr[s_] = slice(a, a + d)
        self.weight = float(mjm.body_subtreemass[mjm.body("pelvis").id]) * 9.81
        n = self.n
        z = lambda *s: torch.zeros(n, *s, **f)   # noqa: E731
        self.cmd = z(3)
        self.last_act, self.act = z(NU), z(NU)
        self.phase, self.phase_dt = z(2), z()
        self.air = z(4)
        self.last_contact = torch.zeros(n, 4, dtype=torch.bool, device=dev)
        self.swing_peak = z(4)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.push_every = torch.zeros(n, dtype=torch.long, device=dev)
        self.mode_line = torch.ones(n, dtype=torch.bool, device=dev)
        self.line_y, self.line_yaw = z(), z()
        self.speed = z()
        self.ep_ret, self.ep_dist = z(), z()
        self.t_cmd = torch.zeros(n, dtype=torch.long, device=dev)
        self.fell = torch.zeros(n, dtype=torch.bool, device=dev)
        self.terrain = terrain
        self.Hg = torch.tensor(ground_grid(terrain), device=dev, dtype=torch.float32)
        lay = course_of(terrain) or {}
        self.ice = lay.get("ice")
        self.flats = [c_ for c_ in (lay.get("ice"), lay.get("sand")) if c_]
        for c_ in self.flats:
            i0, i1 = int(round((c_["x0"] - HF_X0) / HF_RES)), int(round((c_["x0"] + c_["size"] - HF_X0) / HF_RES))
            j0, j1 = int(round((c_["y0"] - HF_Y0) / HF_RES)), int(round((c_["y0"] + c_["size"] - HF_Y0) / HF_RES))
            self.Hg[j0:j1 + 1, i0:i1 + 1] = torch.clamp(self.Hg[j0:j1 + 1, i0:i1 + 1], min=float(c_["top"]))
        self.sole = 0.036                    # (the foot sites are the feet's centres: their spheres' radius)
        Q, head, curv, L = PATH
        self.P, self.PH, self.PK = torch.tensor(Q, **f), torch.tensor(head, **f), torch.tensor(curv, **f)
        self.M = self.P.shape[0]
        self.on_path = torch.zeros(n, dtype=torch.bool, device=dev)
        self.level = torch.zeros(n, dtype=torch.long, device=dev)       # (the course's curriculum: 0 flat … 4 the hardest; see spawn_circuit)
        self.flat_idx = None
        self.pidx = torch.zeros(n, dtype=torch.long, device=dev)
        self.pdir = torch.ones(n, **f)
        self.win = torch.arange(-10, 31, device=dev)
        self.scan = scan
        gx, gy = torch.meshgrid(torch.tensor(SCAN_X, **f), torch.tensor(SCAN_Y, **f), indexing="ij")
        self.sx, self.sy = gx.reshape(-1), gy.reshape(-1)
        self.scan_bias = z()
        self.yard = [s_ for s_ in COURSE["stairs"] if s_["name"] == "yard"]
        T = self.tiles = COURSE["tiles"]
        nt = T["n"] * T["n"]
        self.tile_a, self.tile_b, self.tile_w, self.tile_p = z(nt), z(nt), z(nt), z(nt)
        self.tile_h = z(nt)
        self.tt = z()
        self.tiles_on = terrain == "circuit"
        self.stairs_main = [s_ for s_ in COURSE["stairs"] if s_["name"] == "circuit"][0]

    def contacts(self):
        """each foot on the ground (its touch sensor: more than 1 N), FL, FR, HL, HR"""
        import torch
        return torch.stack([self.S(f + "_touch")[:, 0] > 1.0 for f in ("FL", "FR", "HL", "HR")], -1)

    def feet_pos(self):
        import torch
        return torch.stack([self.S(f + "_pos") for f in ("FL", "FR", "HL", "HR")], 1)       # (n, 4, 3)

    def feet_height(self):
        """each foot's sole above the ground under it"""
        import torch
        p = self.feet_pos()
        return torch.stack([p[:, k, 2] - self.ground(p[:, k, 0], p[:, k, 1]) for k in range(4)], -1) - self.sole

    def height(self):
        """its body's height above the ground under it"""
        q = self.sim.qpos.float()
        return q[:, 2] - self.ground(q[:, 0], q[:, 1])

    def sample_cmd(self, idx):
        import torch
        k = idx.numel()
        if k == 0:
            return
        self.speed[idx] = self.U(k, 0.0, self.vmax) * (torch.rand(k, device=self.dev, generator=self.g) > 0.1)
        free = torch.rand(k, device=self.dev, generator=self.g) < (0.35 if self.train else 0.0)       # (a third of the time: driven freely)
        self.mode_line[idx] = ~free
        rnd = torch.stack([self.U(k, -0.6, self.vmax), self.U(k, -0.5, 0.5) * (torch.rand(k, device=self.dev, generator=self.g) < 0.5),
                           self.U(k, -1.0, 1.0)], -1)
        rnd = rnd * (torch.rand(k, 1, device=self.dev, generator=self.g) > 0.12)                      # (and now and then: stand still)
        self.cmd[idx] = torch.where(free.unsqueeze(-1), rnd, self.cmd[idx])
        self.t_cmd[idx] = (self.U(k, 0.0, 300.0)).long()                                               # (changed every 4 to 10 s)

    def reset(self, idx, at=None):
        """robots idx start again (at: [x, y, yaw] to put one there, standing)"""
        import torch
        k = idx.numel()
        if k == 0:
            return
        qpos = self.q0.unsqueeze(0).repeat(k, 1)
        qvel = torch.zeros(k, self.m.nv, device=self.dev)
        if self.train and at is None:
            self.update_levels(idx)
        if self.train:
            if self.terrain == "circuit":
                yaw = self.spawn_circuit(idx, qpos)
            else:
                qpos[:, 0:2] += self.U(k, -0.5, 0.5, 2)
                yaw = self.U(k, -math.pi, math.pi)
            qpos[:, 7:] = qpos[:, 7:] + self.U(k, -0.25, 0.25, NU)
            qvel[:, 0:6] = self.U(k, -0.5, 0.5, 6)
        else:
            yaw = torch.zeros(k, device=self.dev)
        if not self.train:
            self.on_path[idx] = False
        if at is not None:
            qpos[:, 0], qpos[:, 1], yaw = float(at[0]), float(at[1]), torch.full((k,), float(at[2]), device=self.dev)
        if self.train:
            self.new_tiles(idx)
        self.tiles_now()
        gmax = self.ground(qpos[:, 0], qpos[:, 1], w=idx)
        for dx, dy in ((0.33, 0.17), (0.33, -0.17), (-0.27, 0.17), (-0.27, -0.17)):
            gmax = torch.maximum(gmax, self.ground(qpos[:, 0] + dx * torch.cos(yaw) - dy * torch.sin(yaw), qpos[:, 1] + dx * torch.sin(yaw) + dy * torch.cos(yaw), w=idx))
        for dx, dy in ((0.45, 0.22), (0.45, -0.22), (-0.4, 0.22), (-0.4, -0.22), (0.0, 0.25), (0.0, -0.25)):    # (wider: the legs' noise, the shins)
            gmax = torch.maximum(gmax, self.ground(qpos[:, 0] + dx * torch.cos(yaw) - dy * torch.sin(yaw), qpos[:, 1] + dx * torch.sin(yaw) + dy * torch.cos(yaw), w=idx))
        qpos[:, 2] = qpos[:, 2] + gmax + 0.04
        qpos[:, 3] = torch.cos(yaw / 2)
        qpos[:, 6] = torch.sin(yaw / 2)
        qpos[:, 7:] = torch.maximum(torch.minimum(qpos[:, 7:], self.soft_hi), self.soft_lo)
        self.sim.set_rows(idx, qpos, qvel, qpos[:, 7:])
        yard_line = getattr(self, "_yard_line", None)
        if yard_line is not None:
            self.line_yaw[idx] = yard_line[1]
            self.line_y[idx] = yard_line[0]
            self._yard_line = None
        else:
            self.line_yaw[idx] = yaw + (self.U(k, -0.4, 0.4) if self.train else 0.0)
            self.line_y[idx] = -(qpos[:, 0] * torch.sin(self.line_yaw[idx])) + qpos[:, 1] * torch.cos(self.line_yaw[idx]) + (self.U(k, -0.5, 0.5) if self.train else 0.0)
        self.scan_bias[idx] = self.U(k, -0.02, 0.02) if self.train else 0.0
        f_ = self.U(k, 1.6, 2.0) if self.train else torch.full((k,), 1.8, device=self.dev)     # (a trot of about 1.8 steps a second)
        self.phase_dt[idx] = 2 * math.pi * CTRL_DT * f_
        self.phase[idx] = torch.tensor([0.0, math.pi], device=self.dev)
        self.last_act[idx] = 0.0
        self.act[idx] = 0.0
        self.air[idx] = 0.0
        self.last_contact[idx] = False
        self.swing_peak[idx] = 0.0
        self.steps[idx] = 0
        self.push_every[idx] = (self.U(k, 5.0, 10.0) / CTRL_DT).long()
        self.ep_ret[idx] = 0.0
        self.ep_dist[idx] = 0.0
        self.fell[idx] = False
        self.sample_cmd(idx)

    def ground_scan(self, noisy):
        """the elevation map: around and ahead of it, turned with it, as heights below its body's usual height"""
        import torch
        q = self.sim.qpos.float()
        yaw = yaw_of(q[:, 3:7])
        c, s_ = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        px = q[:, 0:1] + c * self.sx - s_ * self.sy
        py = q[:, 1:2] + s_ * self.sx + c * self.sy
        h = (q[:, 2:3] - H0 - self.ground(px, py)).clamp(-1.0, 1.0)
        if noisy and self.noise > 0:
            h = h + self.scan_bias.unsqueeze(-1) + (2 * torch.rand(h.shape, device=self.dev, generator=self.g) - 1) * 0.02 * self.noise
        return h

    def obs(self, noisy=True):
        import torch
        q, v = self.sim.qpos.float(), self.sim.qvel.float()
        quat = q[:, 3:7]
        grav = quat_rotate_inv(quat, torch.tensor([[0.0, 0.0, -1.0]], device=self.dev).expand(self.n, 3))
        lin, gyro = self.S("local_linvel"), self.S("gyro")
        jp, jv = q[:, 7:] - self.default, v[:, 6:]
        k = self.noise if noisy else 0.0
        nz = lambda x, s: x + (2 * torch.rand(x.shape, device=self.dev, generator=self.g) - 1) * s * k if k > 0 else x   # noqa: E731
        ph = torch.cat([torch.cos(self.phase), torch.sin(self.phase)], -1)
        actor = torch.cat([nz(lin, 0.1), nz(gyro, 0.2), nz(grav, 0.05), self.cmd, nz(jp, 0.02), nz(jv, 1.5), self.last_act, ph], -1)
        sc = self.ground_scan(noisy=False)
        if self.scan:
            actor = torch.cat([actor, self.ground_scan(noisy=noisy) if (noisy and k > 0) else sc], -1)
        contact = self.contacts().float()
        feet_v = torch.cat([self.S(f + "_vel") for f in ("FL", "FR", "HL", "HR")], -1)
        priv = torch.cat([torch.cat([lin, gyro, grav, self.cmd, jp, jv, self.last_act, ph], -1), self.height().unsqueeze(-1), contact, self.air,
                          feet_v, self.S("global_angvel"), self.S("global_linvel"), sc], -1)
        return actor, priv

    def step(self, action):
        import torch
        sim, n = self.sim, self.n
        a = action.clamp(-5.0, 5.0)
        self.guide()
        if self.train and self.push_max > 0:
            due = (self.steps + 1) % self.push_every.clamp(min=1) == 0
            if bool(due.any()):
                th = self.U(n, 0.0, 2 * math.pi)
                mag = self.U(n, 0.1, self.push_max) * due
                sim.kick(torch.stack([torch.cos(th) * mag, torch.sin(th) * mag], -1))
        sim.ctrl.copy_(torch.maximum(torch.minimum(self.default + a * ACTION_SCALE, self.soft_hi), self.soft_lo).to(sim.ctrl.dtype))
        if self.tiles_on:
            self.tt += CTRL_DT
            self.tiles_now()
        sim.step()
        q, v = sim.qpos.float(), sim.qvel.float()
        contact = self.contacts()
        contact_f = contact | self.last_contact
        first = (self.air > 0) & contact_f
        self.air += CTRL_DT
        feet_z = self.feet_height()
        up = self.S("upvector")[:, 2]
        h = self.height()
        # the end: the body down on the ground, tipped over, or the physics gone wrong
        fell = (h < 0.2) | (up < 0.4) | torch.isnan(q).any(-1) | torch.isnan(v).any(-1)
        lin, gyro = self.S("local_linvel"), self.S("gyro")
        cmd = self.cmd
        jq = q[:, 7:]
        r = {}
        r["tracking_lin_vel"] = torch.exp(-((cmd[:, :2] - lin[:, :2]) ** 2).sum(-1) / 0.25)
        r["tracking_ang_vel"] = torch.exp(-(cmd[:, 2] - gyro[:, 2]) ** 2 / 0.25)
        r["lin_vel_z"] = self.S("global_linvel")[:, 2] ** 2
        r["ang_vel_xy"] = (gyro[:, :2] ** 2).sum(-1)
        grav = quat_rotate_inv(q[:, 3:7], torch.tensor([[0.0, 0.0, -1.0]], device=self.dev).expand(n, 3))
        r["orientation"] = (grav[:, :2] ** 2).sum(-1)
        air = ((self.air - 0.25) * first).clamp(max=0.3)
        moving = cmd.norm(dim=-1) > 0.05
        r["feet_air_time"] = air.sum(-1) * moving
        fv = torch.stack([self.S(f + "_vel")[:, :2].norm(dim=-1) for f in ("FL", "FR", "HL", "HR")], -1)
        r["feet_slip"] = (fv * contact).sum(-1)
        rz = rz_bezier(self.phase[:, PAIR], 0.09)
        r["feet_phase"] = torch.exp(-((feet_z - rz) ** 2).sum(-1) / 0.01) * moving
        still = ~moving
        r["stand_still"] = (jq - self.default).abs().sum(-1) * still
        r["termination"] = fell.float()
        r["alive"] = (~fell).float()
        r["collision"] = (h < 0.25).float()                                   # (the body or the legs down near the ground)
        r["torques"] = (sim.tq.float() ** 2).sum(-1)
        r["action_rate"] = ((a - self.last_act) ** 2).sum(-1)
        r["dof_pos_limits"] = ((self.soft_lo - jq).clamp(min=0) + (jq - self.soft_hi).clamp(min=0)).sum(-1)
        r["pose"] = ((jq - self.default) ** 2).sum(-1)
        r["hip_abd"] = (jq[:, self.hx_i] - self.default[self.hx_i]).abs().sum(-1) * (cmd[:, 1].abs() < 0.1)
        rew = (sum(SCALES[k_] * v_ for k_, v_ in r.items()) * CTRL_DT).clamp(min=0.0)
        rew = torch.nan_to_num(rew, nan=-1.0)
        self.steps += 1
        self.t_cmd += 1
        self.phase = torch.fmod(self.phase + self.phase_dt.unsqueeze(-1) + math.pi, 2 * math.pi) - math.pi
        self.last_act = a
        self.air = self.air * (~contact)
        self.last_contact = contact
        self.ep_ret += rew
        wh = self.path_heading()
        self.ep_dist += (v[:, 0] * torch.cos(wh) + v[:, 1] * torch.sin(wh)) * CTRL_DT
        timeout = self.steps >= EP_STEPS
        if self.train:
            new_cmd = (self.t_cmd >= 500).nonzero(as_tuple=False).flatten()
            self.sample_cmd(new_cmd)
        self.fell = fell
        loads = torch.stack([self.S(f + "_touch")[:, 0] for f in ("FL", "FR", "HL", "HR")], -1) / self.weight
        info = {"fell": fell, "timeout": timeout & ~fell, "terms": r, "lin": lin, "gyro": gyro, "contact": contact, "feet_z": feet_z, "forces": loads}
        return rew, fell | timeout, info

    def ground(self, x, y, w=None):
        """the ground's height under x, y (bilinear between the heightfield's points; flat outside it), the moving tiles
        included (x, y: one row per robot, or w: the robots they belong to)"""
        import torch
        g = self.ground_static(x, y)
        if not self.tiles_on:
            return g
        T = self.tiles
        n = T["n"]
        u, v = (x - T["x0"]) / T["cell"], (y - T["y0"]) / T["cell"]
        inside = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        tid = (v.clamp(0, n - 1e-4).floor().long() * n + u.clamp(0, n - 1e-4).floor().long())
        if w is None:
            th = self.tile_h if tid.dim() == 2 else self.tile_h
            th = th.gather(1, tid) if tid.dim() == 2 else th.gather(1, tid.unsqueeze(-1)).squeeze(-1)
        else:
            th = self.tile_h[w].gather(1, tid.reshape(len(w), -1)).reshape(tid.shape)
        return torch.where(inside, torch.maximum(g, th), g)

    def tiles_now(self, idx=None):
        """the tiles' tops at the tiles' clock: a + a sin(w t + p) above their resting heights b (0 to 2a over b)"""
        import torch
        h = self.tile_b + self.tile_a * (1.0 + torch.sin(self.tile_w * self.tt.unsqueeze(-1) + self.tile_p))
        self.tile_h = h if self.tiles_on else torch.full_like(h, -5.0)
        self.sim.set_tiles(self.tile_h - self.tiles["half"])

    def new_tiles(self, idx, seed=None):
        """robots idx get a new pattern of tiles (in training: every episode)"""
        import torch
        T, k = self.tiles, idx.numel()
        if k == 0:
            return
        nt = T["n"] * T["n"]
        g = self.g if seed is None else torch.Generator(device=self.dev).manual_seed(seed)
        R = lambda a, b: a + (b - a) * torch.rand(k, nt, device=self.dev, generator=g)   # noqa: E731
        self.tile_a[idx] = R(*T["amp"])
        self.tile_b[idx] = R(*T["base"])
        self.tile_w[idx] = 2 * math.pi / R(*T["period"])
        self.tile_p[idx] = R(0.0, 2 * math.pi)
        self.tt[idx] = 0.0

    def ground_static(self, x, y):
        """the heightfield's ground under x, y"""
        import torch
        H = self.Hg
        nr, nc = H.shape
        gx = ((x - HF_X0) / HF_RES).clamp(0, nc - 1.0001)
        gy = ((y - HF_Y0) / HF_RES).clamp(0, nr - 1.0001)
        i0, j0 = gx.floor().long(), gy.floor().long()
        fx, fy = gx - i0, gy - j0
        return (H[j0, i0] * (1 - fx) * (1 - fy) + H[j0, i0 + 1] * fx * (1 - fy) + H[j0 + 1, i0] * (1 - fx) * fy + H[j0 + 1, i0 + 1] * fx * fy)

    def S(self, name):
        return self.sim.sens[:, self.sadr[name]].float()

    def U(self, k, a, b, *shape):
        import torch
        return a + (b - a) * torch.rand(k, *shape, device=self.dev, generator=self.g)

    # ---- commands
    def path_heading(self):
        """the way to walk: along the circuit's line (either way round) or along its straight line"""
        import torch
        ph = self.PH[self.pidx] + (self.pdir < 0).float() * math.pi
        return torch.where(self.on_path, ph, self.line_yaw)

    def guide(self):
        """the line follower: the turn and the sideways step that bring it back onto its line and its heading
        (a straight line, or the circuit's line: its nearest point found near the last one, the heading a little ahead)"""
        import torch
        q = self.sim.qpos
        x, y = q[:, 0].float(), q[:, 1].float()
        yaw = yaw_of(q[:, 3:7].float())
        cand = (self.pidx.unsqueeze(-1) + self.win.unsqueeze(0) * self.pdir.long().unsqueeze(-1)) % self.M
        d2 = (self.P[cand, 0] - x.unsqueeze(-1)) ** 2 + (self.P[cand, 1] - y.unsqueeze(-1)) ** 2
        self.pidx = torch.where(self.on_path, cand.gather(1, d2.argmin(-1, keepdim=True)).squeeze(-1), self.pidx)
        ahead = (self.pidx + 3 * self.pdir.long()) % self.M
        p_head = self.PH[ahead] + (self.pdir < 0).float() * math.pi
        nx, ny = -torch.sin(self.PH[self.pidx]), torch.cos(self.PH[self.pidx])
        p_off = ((x - self.P[self.pidx, 0]) * nx + (y - self.P[self.pidx, 1]) * ny) * self.pdir     # left of the way it walks (m)
        cy, sy = torch.cos(self.line_yaw), torch.sin(self.line_yaw)
        l_off = -(x * sy) + y * cy - self.line_y
        want = torch.where(self.on_path, p_head, self.line_yaw)
        off = torch.where(self.on_path, p_off, l_off)
        dyaw = torch.atan2(torch.sin(want - yaw), torch.cos(want - yaw))
        ff = torch.where(self.on_path, self.PK[self.pidx] * self.pdir * self.speed, torch.zeros_like(off))   # (round the corners)
        wz = (2.0 * dyaw - 0.6 * off * torch.cos(dyaw) + ff).clamp(-1.0, 1.0)
        vy = (-0.8 * off).clamp(-0.3, 0.3) * (self.speed > 0.05)
        wz = wz * ((self.speed > 0.05) | (dyaw.abs() > 0.15))
        line_cmd = torch.stack([self.speed, vy, wz], -1)
        self.cmd = torch.where(self.mode_line.unsqueeze(-1), line_cmd, self.cmd)
        self.yaw_err, self.line_off = dyaw, off

    # ---- reset
    def update_levels(self, idx):
        """the curriculum: a robot that walked its episode through without falling and covered ground goes up a level
        (harder ground next time); one that fell in the first half goes down one"""
        import torch
        if idx.numel() == 0 or not self.train or self.terrain != "circuit":
            return
        done_ep = self.steps[idx] > 0
        good = done_ep & ~self.fell[idx] & (self.ep_dist[idx] > 4.0)
        bad = done_ep & self.fell[idx] & (self.steps[idx] < EP_STEPS // 2)
        lv = self.level[idx] + good.long() - bad.long()
        self.level[idx] = lv.clamp(0, 4)

    def spawn_circuit(self, idx, qpos):
        """where robots idx start in training on the circuit, by their level (a curriculum, as in legged_gym): 0 flat
        stretches of the circuit; 1 the 6 cm yard staircase, the ice, side 1's slopes and rough ground; 2 the 9 cm
        staircase, the moving tiles, side 1; 3 the 12 cm staircase and the circuit's stairs; 4 the 15 cm staircase and
        anywhere. One in five starts a level lower (not to forget). On the circuit either way round."""
        import torch
        import numpy as np
        k = idx.numel()
        M = self.M
        if self.flat_idx is None:          # (the circuit's points with flat ground 1.5 m before and after them)
            P = self.P.cpu().numpy()
            PH = self.PH.cpu().numpy()
            hs = np.zeros(len(P))
            for d in np.arange(-1.5, 1.51, 0.25):
                for sd in (-0.4, 0.0, 0.4):
                    hs = np.maximum(hs, np.abs(course_height(P[:, 0] + d * np.cos(PH) - sd * np.sin(PH), P[:, 1] + d * np.sin(PH) + sd * np.cos(PH), COURSE)))
            ic = COURSE.get("ice")
            if ic:
                hs = np.where((P[:, 0] > ic["x0"] - 1.5) & (P[:, 0] < ic["x0"] + ic["size"] + 1.5) & (P[:, 1] > ic["y0"] - 1.5) & (P[:, 1] < ic["y0"] + ic["size"] + 1.5), 1.0, hs)
            fl = np.nonzero(hs < 0.01)[0]
            self.flat_idx = torch.tensor(fl if len(fl) else np.arange(len(P)), device=self.dev, dtype=torch.long)
        lv = self.level[idx] - (torch.rand(k, device=self.dev, generator=self.g) < 0.2).long()
        lv = lv.clamp(0, 4)
        u = torch.rand(k, device=self.dev, generator=self.g)
        # on the circuit: a point of its line
        i_any = (torch.rand(k, device=self.dev, generator=self.g) * M).long() % M
        st = self.stairs_main
        mid = st["x0"] + st["dir"] * ((st["n"] - 1) * st["tread"] + st["landing"] / 2)
        s_st = self.nearest_index(torch.tensor([mid], device=self.dev), torch.tensor([st["y"]], device=self.dev))
        i_st = (s_st + (self.U(k, -1.0, 1.0) * 80).long()) % M          # within 8 m of the landing: the stairs and their approaches
        i_s1 = (self.nearest_index(torch.tensor([2.0], device=self.dev), torch.tensor([0.0], device=self.dev)) + (torch.rand(k, device=self.dev, generator=self.g) * 280).long()) % M
        tl = self.tiles
        i_tl = (self.nearest_index(torch.tensor([tl["x0"] + tl["cell"] * tl["n"] / 2], device=self.dev), torch.tensor([tl["y0"] + tl["cell"] * tl["n"] / 2], device=self.dev))
                + (self.U(k, -1.0, 1.0) * 50).long()) % M                # within 5 m of the moving tiles' centre
        ic = COURSE.get("ice") or {"x0": -6.0, "y0": 7.0, "size": 4.0}
        i_ic = (self.nearest_index(torch.tensor([ic["x0"] + ic["size"] / 2], device=self.dev), torch.tensor([ic["y0"] + ic["size"] / 2], device=self.dev))
                + (self.U(k, -1.0, 1.0) * 40).long()) % M                # within 4 m of the ice's centre
        i_fl = self.flat_idx[(torch.rand(k, device=self.dev, generator=self.g) * len(self.flat_idx)).long().clamp(max=len(self.flat_idx) - 1)]
        v = torch.rand(k, device=self.dev, generator=self.g)
        pi = torch.where(lv == 0, i_fl,
             torch.where(lv == 1, torch.where(v < 0.5, i_ic, i_s1),
             torch.where(lv == 2, torch.where(v < 0.5, i_tl, i_s1),
             torch.where(lv == 3, i_st, i_any))))
        dirn = torch.where(torch.rand(k, device=self.dev, generator=self.g) < 0.5, 1.0, -1.0)
        side = self.U(k, -0.3, 0.3)
        hx, hy = torch.cos(self.PH[pi]), torch.sin(self.PH[pi])
        px, py = self.P[pi, 0] - side * hy, self.P[pi, 1] + side * hx
        yaw = self.PH[pi] + (dirn < 0).float() * math.pi + self.U(k, -0.3, 0.3)
        # or in the yard: on one of its staircases, walking along it either way
        yard = (lv > 0) & (u < 0.5)                                       # (half of levels 1 to 4: on the yard's staircase of that level)
        ys = torch.tensor([s_["y"] for s_ in self.yard], device=self.dev)
        which = (lv - 1).clamp(0, len(self.yard) - 1)
        y0 = self.yard[0]
        ln = 2 * (y0["n"] - 1) * y0["tread"] + y0["landing"]
        yx = self.U(k, y0["x0"] - 2.0, y0["x0"] + ln + 2.0)
        yy = ys[which] + self.U(k, -0.25, 0.25)
        yback = torch.rand(k, device=self.dev, generator=self.g) < 0.5
        yyaw = yback.float() * math.pi + self.U(k, -0.2, 0.2)
        qpos[:, 0] = torch.where(yard, yx, px)
        qpos[:, 1] = torch.where(yard, yy, py)
        self.on_path[idx] = ~yard
        self.pidx[idx] = pi
        self.pdir[idx] = dirn
        lyaw = yback.float() * math.pi
        self._yard_line = (torch.where(yard, -(0.0) * torch.sin(lyaw) + ys[which] * torch.cos(lyaw), self.line_y[idx]),
                           torch.where(yard, lyaw, yaw))
        return torch.where(yard, yyaw, yaw)

    def nearest_index(self, x, y):
        """the circuit's point nearest to each x, y"""
        d2 = (self.P[:, 0].unsqueeze(0) - x.unsqueeze(-1)) ** 2 + (self.P[:, 1].unsqueeze(0) - y.unsqueeze(-1)) ** 2
        return d2.argmin(-1)

    def put_on_path(self, idx, i, dirn):
        """robots idx follow the circuit from its point i, either way round (dirn +1 counterclockwise)"""
        self.on_path[idx] = True
        self.pidx[idx] = i
        self.pdir[idx] = float(dirn)

    def reset_all(self):
        import torch
        self.reset(torch.arange(self.n, device=self.dev))
        self.sim.refresh()

    # ---- what it senses


def quat_mul(a, b):
    """the product of quaternions a and b (w, x, y, z), row by row"""
    import torch
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


STAND_EP = 400                       # 8 s to get up and stay up
STAND_SCALES = {"height": 2.0, "upright": 3.0, "pose": 2.0, "feet": 1.0, "still": 1.0, "action_rate": -0.01, "joint_vel": -2e-4,
                "dof_pos_limits": -1.0, "torques": -1e-5}


def is_standing(h, up):
    """on its feet: its body at walking height and level"""
    return (h > 0.32) & (up > 0.85)


class StandEnv(WalkEnv):
    """Learning to stand and to self-right: each episode starts with the robot lying where a fall left it (dropped from a
    random pose, any way round: on its side, on its back), and it has 8 s to get onto its feet and stay there. Rewards:
    its body level and high, then once up its standing pose, the four feet down and stillness. (An upward help on its
    body can make it easier at first; for a quadruped it starts at none.)"""

    def __init__(self, sim, mjm, seed=0, noise=1.0, train=True):
        super().__init__(sim, mjm, seed=seed, vmax=0.0, push=0.0, noise=noise, train=train, terrain="flat", scan=True)
        import torch
        self.torso = mjm.body("pelvis").id
        self.assist = 0.0
        self.bank = None
        n = self.n
        self.up_steps = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.t_up = torch.full((n,), -1.0, device=self.dev)
        self.start_standing = torch.zeros(n, dtype=torch.bool, device=self.dev)

    def build_bank(self, rounds=3, log=None):
        """lying poses: every robot dropped from 0.6-0.9 m in a random pose and turn, holding random joint targets, 2 s to land"""
        import torch
        n, dev = self.n, self.dev
        idx = torch.arange(n, device=dev)
        keep = []
        span = self.soft_hi - self.soft_lo
        for _ in range(rounds):
            qpos = self.q0.unsqueeze(0).repeat(n, 1)
            qpos[:, 0:2] = self.U(n, -0.3, 0.3, 2)
            qpos[:, 2] = self.U(n, 0.6, 0.9)
            quat = torch.randn(n, 4, device=dev, generator=self.g)
            qpos[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True)
            qpos[:, 7:] = torch.maximum(torch.minimum(self.default + self.U(n, -0.35, 0.35, NU) * span, self.soft_hi), self.soft_lo)
            qvel = torch.zeros(n, self.m.nv, device=dev)
            qvel[:, 0:6] = self.U(n, -1.0, 1.0, 6)
            targets = torch.maximum(torch.minimum(self.default + self.U(n, -0.3, 0.3, NU) * span, self.soft_hi), self.soft_lo)
            self.sim.set_rows(idx, qpos, qvel, targets)
            self.sim.set_lift(self.torso, torch.zeros(n, device=dev))
            for _ in range(100):
                self.sim.step()
            q = self.sim.qpos.float().clone()
            up = self.S("upvector")[:, 2]
            ok = ~torch.isnan(q).any(-1) & ~is_standing(self.height(), up) & (self.sim.qvel.float().abs().amax(-1) < 5.0)
            keep.append(q[ok])
        self.bank = torch.cat(keep)
        if log:
            upside = float((torch.tensor([0.0]) if not len(self.bank) else (1 - 2 * (self.bank[:, 4] ** 2 + self.bank[:, 5] ** 2) < 0).float().mean()))
            log("%d lying poses made by dropping the robot (%d rounds): %.0f%% of them on its back, the rest on its side or its belly." % (len(self.bank), rounds, 100 * upside))
        return len(self.bank)

    def reset(self, idx, at=None):
        import torch
        k = idx.numel()
        if k == 0:
            return
        dev = self.dev
        if at is not None or self.bank is None or not len(self.bank):
            qpos = self.q0.unsqueeze(0).repeat(k, 1)
            stand = torch.ones(k, dtype=torch.bool, device=dev)
        else:
            qpos = self.bank[torch.randint(len(self.bank), (k,), device=dev, generator=self.g)].clone()
            stand = torch.rand(k, device=dev, generator=self.g) < (0.12 if self.train else 0.0)
            qpos[stand] = self.q0
        yaw = self.U(k, -math.pi, math.pi)
        qz = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        qpos[:, 3:7] = quat_mul(qz, qpos[:, 3:7])
        qpos[:, 0:2] = self.U(k, -0.5, 0.5, 2)
        qvel = torch.zeros(k, self.m.nv, device=dev)
        self.sim.set_rows(idx, qpos, qvel, qpos[:, 7:])
        self.cmd[idx] = 0.0
        self.mode_line[idx] = False
        self.speed[idx] = 0.0
        self.phase[idx] = torch.tensor([0.0, math.pi], device=dev)
        self.phase_dt[idx] = 0.0
        self.last_act[idx] = 0.0
        self.act[idx] = 0.0
        self.air[idx] = 0.0
        self.last_contact[idx] = False
        self.swing_peak[idx] = 0.0
        self.steps[idx] = 0
        self.ep_ret[idx] = 0.0
        self.ep_dist[idx] = 0.0
        self.fell[idx] = False
        self.up_steps[idx] = 0
        self.t_up[idx] = -1.0
        self.start_standing[idx] = stand
        self.scan_bias[idx] = self.U(k, -0.02, 0.02) if self.train else 0.0

    def step(self, action):
        import torch
        sim = self.sim
        a = action.clamp(-5.0, 5.0)
        sim.ctrl.copy_(torch.maximum(torch.minimum(self.default + a * ACTION_SCALE * 2.5, self.soft_hi), self.soft_lo).to(sim.ctrl.dtype))   # (wide moves: rolling over takes them)
        sim.set_lift(self.torso, self.assist * self.weight * (self.height() < 0.3).float())
        sim.step()
        q, v = sim.qpos.float(), sim.qvel.float()
        h = self.height()
        up = self.S("upvector")[:, 2]
        standing = is_standing(h, up)
        self.up_steps = torch.where(standing, self.up_steps + 1, torch.zeros_like(self.up_steps))
        newly = (self.up_steps == 25) & (self.t_up < 0)
        self.t_up = torch.where(newly, self.steps.float() * CTRL_DT, self.t_up)
        contact = self.contacts()
        jq, jv = q[:, 7:], v[:, 6:]
        st = standing.float()
        r = {}
        r["height"] = ((h - 0.1) / 0.3).clamp(0.0, 1.0) * (up > 0.0)
        r["upright"] = ((up + 1.0) / 2.0) ** 2
        r["pose"] = torch.exp(-((jq - self.default) ** 2).sum(-1) / 1.0) * st
        r["feet"] = contact.float().mean(-1) * (up > 0.7)
        r["still"] = torch.exp(-(self.S("global_linvel") ** 2).sum(-1) / 0.1) * st
        r["action_rate"] = ((a - self.last_act) ** 2).sum(-1)
        r["joint_vel"] = (jv ** 2).sum(-1)
        r["dof_pos_limits"] = ((self.soft_lo - jq).clamp(min=0) + (jq - self.soft_hi).clamp(min=0)).sum(-1)
        r["torques"] = (sim.tq.float() ** 2).sum(-1)
        rew = sum(STAND_SCALES[k_] * v_ for k_, v_ in r.items()) * CTRL_DT
        bad = torch.isnan(q).any(-1) | torch.isnan(v).any(-1)
        rew = torch.nan_to_num(rew, nan=0.0)
        self.steps += 1
        self.last_act = a
        self.last_contact = contact
        self.ep_ret += rew
        timeout = self.steps >= STAND_EP
        self.fell = bad
        loads = torch.stack([self.S(f + "_touch")[:, 0] for f in ("FL", "FR", "HL", "HR")], -1) / self.weight
        info = {"fell": bad, "timeout": timeout & ~bad, "terms": r, "standing": standing, "height": h, "upright": up,
                "lin": self.S("local_linvel"), "gyro": self.S("gyro"), "contact": contact, "forces": loads}
        return rew, timeout | bad, info




# ── looking around: its feet planted, it turns, tilts and lowers its body to aim its front cameras ───────────────────
LOOK_EP = 600                        # 12 s, a new aim every 1.5 to 4 s
LOOK_RANGE = {"pitch": 0.45, "yaw": 0.5, "low": -0.15, "high": 0.05}     # rad (up is negative), rad, m
LOOK_SCALES = {"pitch": 2.0, "yaw": 2.0, "height": 1.0, "level": 1.0, "feet": 1.0, "slip": -1.0, "drift": -6.0, "lin": -2.0,
               "action_rate": -0.01, "joint_vel": -2e-4, "dof_pos_limits": -1.0, "torques": -1e-5}


def wrap(a):
    import torch
    return torch.atan2(torch.sin(a), torch.cos(a))


class LookEnv(StandEnv):
    """Learning to look around: it stands, its four feet where they are, and is asked a body attitude: pitch (nose up
    or down, up to 26°), yaw (turned left or right of where its feet face, up to 29°), height (down 15 cm to up 5 cm).
    It must reach it with its legs alone, without its feet slipping or stepping, and hold it; a new one every 1.5 to
    4 s. The command goes in the walking command's three places: (pitch, yaw, height change)."""

    def __init__(self, sim, mjm, seed=0, noise=1.0, train=True):
        super().__init__(sim, mjm, seed=seed, noise=noise, train=train)
        import torch
        n, dev = self.n, self.dev
        self.yaw0 = torch.zeros(n, device=dev)
        self.feet0 = torch.zeros(n, 4, 3, device=dev)
        self.need_feet = torch.ones(n, dtype=torch.bool, device=dev)
        self.t_look = torch.zeros(n, dtype=torch.long, device=dev)
        self.err_sum = torch.zeros(n, 3, device=dev)
        self.track_sum = torch.zeros(n, device=dev)

    def sample_look(self, idx):
        import torch
        k = idx.numel()
        if k == 0:
            return
        R = LOOK_RANGE
        c = torch.stack([self.U(k, -R["pitch"], R["pitch"]), self.U(k, -R["yaw"], R["yaw"]), self.U(k, R["low"], R["high"])], -1)
        r = torch.rand(k, device=self.dev, generator=self.g)
        c = torch.where((r < 0.15).unsqueeze(-1), torch.zeros_like(c), c)                                 # (now and then: straight ahead)
        one = torch.randint(0, 3, (k,), device=self.dev, generator=self.g)
        only = (r >= 0.15) & (r < 0.4)                                                                  # (or one of the three alone)
        mask = torch.nn.functional.one_hot(one, 3).float()
        c = torch.where(only.unsqueeze(-1), c * mask, c)
        self.cmd[idx] = c
        self.t_look[idx] = self.U(k, 75, 200).long()

    def reset(self, idx, at=None):
        import torch
        k = idx.numel()
        if k == 0:
            return
        dev = self.dev
        qpos = self.q0.unsqueeze(0).repeat(k, 1)
        yaw = self.U(k, -math.pi, math.pi)
        qz = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        qpos[:, 3:7] = qz
        qpos[:, 0:2] = self.U(k, -0.5, 0.5, 2)
        qpos[:, 2] = qpos[:, 2] + 0.01
        self.sim.set_rows(idx, qpos, torch.zeros(k, self.m.nv, device=dev), qpos[:, 7:])
        self.yaw0[idx] = yaw
        self.need_feet[idx] = True
        self.mode_line[idx] = False
        self.speed[idx] = 0.0
        self.phase[idx] = torch.tensor([0.0, math.pi], device=dev)
        self.phase_dt[idx] = 0.0
        self.last_act[idx] = 0.0
        self.act[idx] = 0.0
        self.air[idx] = 0.0
        self.last_contact[idx] = True
        self.swing_peak[idx] = 0.0
        self.steps[idx] = 0
        self.ep_ret[idx] = 0.0
        self.ep_dist[idx] = 0.0
        self.fell[idx] = False
        self.up_steps[idx] = 0
        self.t_up[idx] = -1.0
        self.start_standing[idx] = True
        self.scan_bias[idx] = self.U(k, -0.02, 0.02) if self.train else 0.0
        self.err_sum[idx] = 0.0
        self.track_sum[idx] = 0.0
        self.sample_look(idx)
        self.cmd[idx] = self.cmd[idx] * (self.U(k, 0, 1) > 0.5).float().unsqueeze(-1)      # (half start straight ahead)

    def attitude(self):
        """its body's pitch (nose down positive), yaw from where its feet face, height change from standing"""
        import torch
        q = self.sim.qpos.float()
        grav = quat_rotate_inv(q[:, 3:7], torch.tensor([[0.0, 0.0, -1.0]], device=self.dev).expand(self.n, 3))
        pitch = torch.asin(grav[:, 0].clamp(-1.0, 1.0))
        yaw = wrap(yaw_of(q[:, 3:7]) - self.yaw0)
        return pitch, yaw, self.height() - H0, grav

    def step(self, action):
        import torch
        sim = self.sim
        self.t_look -= 1
        due = (self.t_look <= 0).nonzero(as_tuple=False).flatten()
        if self.train:
            self.sample_look(due)
        a = action.clamp(-5.0, 5.0)
        sim.ctrl.copy_(torch.maximum(torch.minimum(self.default + a * ACTION_SCALE * 1.5, self.soft_hi), self.soft_lo).to(sim.ctrl.dtype))
        sim.step()
        q, v = sim.qpos.float(), sim.qvel.float()
        feet = self.feet_pos()
        if bool(self.need_feet.any()):
            self.feet0 = torch.where(self.need_feet.view(-1, 1, 1), feet, self.feet0)
            self.need_feet[:] = False
        pitch, yaw, dh, grav = self.attitude()
        ep, ey, eh = pitch - self.cmd[:, 0], wrap(yaw - self.cmd[:, 1]), dh - self.cmd[:, 2]
        contact = self.contacts()
        jq, jv = q[:, 7:], v[:, 6:]
        feet_v = torch.stack([self.S(f + "_vel")[:, :2] for f in ("FL", "FR", "HL", "HR")], 1)
        r = {}
        r["pitch"] = torch.exp(-ep ** 2 / 0.01)
        r["yaw"] = torch.exp(-ey ** 2 / 0.01)
        r["height"] = torch.exp(-eh ** 2 / 0.0016)
        r["level"] = torch.exp(-grav[:, 1] ** 2 / 0.005)                          # (no roll)
        r["feet"] = contact.float().mean(-1)
        r["slip"] = ((feet_v ** 2).sum(-1) * contact.float()).sum(-1)
        r["drift"] = (feet[:, :, :2] - self.feet0[:, :, :2]).norm(dim=-1).mean(-1)
        r["lin"] = (self.S("global_linvel")[:, :2] ** 2).sum(-1)
        r["action_rate"] = ((a - self.last_act) ** 2).sum(-1)
        r["joint_vel"] = (jv ** 2).sum(-1)
        r["dof_pos_limits"] = ((self.soft_lo - jq).clamp(min=0) + (jq - self.soft_hi).clamp(min=0)).sum(-1)
        r["torques"] = (sim.tq.float() ** 2).sum(-1)
        rew = sum(LOOK_SCALES[k_] * v_ for k_, v_ in r.items()) * CTRL_DT
        h = self.height()
        up = self.S("upvector")[:, 2]
        bad = torch.isnan(q).any(-1) | torch.isnan(v).any(-1)
        fell = bad | (h < 0.2) | (up < 0.5)
        rew = torch.where(fell, torch.full_like(rew, -2.0), torch.nan_to_num(rew, nan=0.0))
        settled = (self.steps > 50).float()
        self.err_sum += torch.stack([ep.abs(), ey.abs(), eh.abs()], -1) * settled.unsqueeze(-1)
        self.track_sum += (r["pitch"] + r["yaw"] + r["height"]) / 3.0 * settled
        self.steps += 1
        self.last_act = a
        self.last_contact = contact
        self.ep_ret += rew
        timeout = self.steps >= LOOK_EP
        self.fell = fell
        loads = torch.stack([self.S(f + "_touch")[:, 0] for f in ("FL", "FR", "HL", "HR")], -1) / self.weight
        info = {"fell": fell, "timeout": timeout & ~fell, "terms": r, "height": h, "upright": up, "contact": contact, "forces": loads,
                "lin": self.S("local_linvel"), "gyro": self.S("gyro")}
        return rew, timeout | fell, info


# ── learning: PPO with an actor (the policy: what the robot senses → joint targets) and a critic (how good a moment is,
# from more than the robot can sense: the privileged state of the simulation) ─────────────────────────────────────────
import math


class Norm:
    """running mean and variance of what goes into a network"""

    def __init__(self, torch, n, dev):
        self.t = torch
        self.mean = torch.zeros(n, device=dev)
        self.var = torch.ones(n, device=dev)
        self.count = torch.full((n,), 1e-4, device=dev)          # (one count per input: new inputs learn their scale fast)

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-2)).clamp(-5.0, 5.0)     # (what never changes stays near 0)

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count.cpu()}

    def load(self, st, dev):
        torch = self.t
        n = self.mean.shape[0]
        m, v, c = st["mean"].to(dev), st["var"].to(dev), st["count"]
        c = c.to(dev) if torch.is_tensor(c) else torch.full((m.shape[0],), float(c), device=dev)
        k = m.shape[0]               # (a policy with fewer inputs: the new ones start unscaled, with no count)
        self.mean[:k], self.var[:k], self.count[:k] = m, v, c
        if k < n:
            self.mean[k:], self.var[k:], self.count[k:] = 0.0, 1.0, 1e-4


def mlp(nn, i, sizes, o):
    layers, d = [], i
    for h in sizes:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def make_ac(torch, obs, priv, act, hidden=(512, 256, 128)):
    import torch.nn as nn

    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = mlp(nn, obs, hidden, act)
            self.critic = mlp(nn, priv, hidden, 1)
            self.log_std = nn.Parameter(torch.full((act,), math.log(0.25)))    # exploration: joint targets jitter by 0.25 x 0.5 rad at first (more and even a good walker falls)
            with torch.no_grad():     # a new policy starts by holding the default pose (its last layer near zero)
                self.actor[-1].weight.mul_(0.01)
                self.actor[-1].bias.zero_()
    return AC()


class PPO:
    def __init__(self, torch, dev, n_env, obs, priv, act, horizon=24, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5,
                 minibatches=4, entropy=0.001, desired_kl=0.01, schedule="fixed"):
        self.t, self.dev = torch, dev
        self.ac = make_ac(torch, obs, priv, act).to(dev)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.lr = lr
        self.on = Norm(torch, obs, dev)
        self.pn = Norm(torch, priv, dev)
        self.n, self.T, self.obs_dim, self.priv_dim, self.act_dim = n_env, horizon, obs, priv, act
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.ent, self.kl_want = gamma, lam, clip, epochs, minibatches, entropy, desired_kl
        self.schedule = schedule
        T, n = horizon, n_env
        z = lambda *s: torch.zeros(*s, device=dev)   # noqa: E731
        self.b_obs, self.b_priv, self.b_act = z(T, n, obs), z(T, n, priv), z(T, n, act)
        self.b_lp, self.b_val, self.b_rew, self.b_done, self.b_mu = z(T, n), z(T, n), z(T, n), z(T, n), z(T, n, act)

    def act(self, obs, priv, t=None, deterministic=False):
        torch = self.t
        with torch.no_grad():
            x = self.on(obs)
            mu = self.ac.actor(x)
            if deterministic:
                return mu
            std = self.ac.log_std.exp()
            a = mu + std * torch.randn_like(mu)
            if t is not None:
                v = self.ac.critic(self.pn(priv)).squeeze(-1)
                lp = (-0.5 * ((a - mu) / std) ** 2 - self.ac.log_std - 0.9189385).sum(-1)
                self.b_obs[t], self.b_priv[t], self.b_act[t], self.b_lp[t], self.b_val[t], self.b_mu[t] = obs, priv, a, lp, v, mu
            return a

    def record(self, t, rew, done, timeout, priv_next):
        """the reward of step t; a run cut by the clock (not a fall) is worth what the critic says of where it stopped"""
        torch = self.t
        with torch.no_grad():
            if bool(timeout.any()):
                v_next = self.ac.critic(self.pn(priv_next)).squeeze(-1)
                rew = rew + self.gamma * v_next * timeout.float()
            self.b_rew[t], self.b_done[t] = rew, done.float()

    def update(self, priv_last, critic_only=False):
        """one learning step on the batch gathered (critic_only: the policy holds still while the critic learns new ground)"""
        torch = self.t
        T, n = self.T, self.n
        with torch.no_grad():
            last_v = self.ac.critic(self.pn(priv_last)).squeeze(-1)
            adv = torch.zeros_like(self.b_rew)
            gae = torch.zeros(n, device=self.dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else self.b_val[t + 1]
                nonterm = 1.0 - self.b_done[t]
                delta = self.b_rew[t] + self.gamma * nv * nonterm - self.b_val[t]
                gae = delta + self.gamma * self.lam * nonterm * gae
                adv[t] = gae
            ret = adv + self.b_val
        N = T * n
        obs, priv, act = self.b_obs.reshape(N, -1), self.b_priv.reshape(N, -1), self.b_act.reshape(N, -1)
        lp_old, val_old, mu_old = self.b_lp.reshape(N), self.b_val.reshape(N), self.b_mu.reshape(N, -1)
        adv, ret = adv.reshape(N), ret.reshape(N)
        xo, xp = self.on(obs), self.pn(priv)
        size = N // self.mb
        stats = {"kl": 0.0, "vloss": 0.0, "ploss": 0.0}
        std_old = self.ac.log_std.detach().exp()
        stop, stop_ok = False, True
        for _ in range(self.epochs):
            if stop:
                break
            perm = torch.randperm(N, device=self.dev)
            for i in range(self.mb):
                idx = perm[i * size:(i + 1) * size]
                mu = self.ac.actor(xo[idx])
                ls = self.ac.log_std
                std = ls.exp()
                v = self.ac.critic(xp[idx]).squeeze(-1)
                # how far the policy has moved from the one that gathered this batch (KL)
                with torch.no_grad():
                    kl = (torch.log(std / std_old) + (std_old ** 2 + (mu_old[idx] - mu) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean()
                    if self.schedule == "adaptive":          # the step size follows it (as rsl_rl can)
                        if kl > 2 * self.kl_want:
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif 0 < kl < self.kl_want / 2:
                            self.lr = min(1e-2, self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                if kl > 4 * self.kl_want and stop_ok and not critic_only:
                    stop = True                                 # moved far enough on this batch: no more steps on it
                    break
                a_ = adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                nlp = (-0.5 * ((act[idx] - mu) / std) ** 2 - ls - 0.9189385).sum(-1)
                ratio = (nlp - lp_old[idx]).exp()
                pl = -torch.min(ratio * a_, ratio.clamp(1 - self.clip, 1 + self.clip) * a_).mean()
                vc = val_old[idx] + (v - val_old[idx]).clamp(-self.clip, self.clip)
                vl = torch.max((v - ret[idx]) ** 2, (vc - ret[idx]) ** 2).mean()
                ent = (ls + 1.4189385).sum()
                loss = vl if critic_only else pl + 1.0 * vl - self.ent * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()
                stats["kl"], stats["vloss"], stats["ploss"] = float(kl), float(vl.detach()), float(pl.detach())
        with torch.no_grad():     # (the input scaling follows what it sees, after the update that used the old one)
            self.on.update(obs)
            self.pn.update(priv)
        stats["std"] = float(self.ac.log_std.detach().exp().mean())
        stats["lr"] = self.lr
        return stats

    def state(self):
        return {"ac": self.ac.state_dict(), "obs_norm": self.on.state(), "priv_norm": self.pn.state(), "lr": self.lr, "opt": self.opt.state_dict()}

    def load(self, st):
        sd = st["ac"]
        same = sd["actor.0.weight"].shape[1] == self.obs_dim and sd["critic.0.weight"].shape[1] == self.priv_dim
        self.ac.load_state_dict(widen(self.t, sd, self.obs_dim, self.priv_dim))
        self.on.load(st["obs_norm"], self.dev)
        self.pn.load(st["priv_norm"], self.dev)
        self.lr = float(st.get("lr") or self.lr)
        # the optimizer's running moments too, when the network is the same shape: training carries on where it was
        if same and st.get("opt"):
            try:
                self.opt.load_state_dict(st["opt"])
            except Exception:      # noqa: BLE001
                pass


def widen(torch, sd, obs, priv):
    """a policy's weights for more inputs than it had: the new inputs' weights start at zero (it acts as before)"""
    sd = dict(sd)
    for key, want in (("actor.0.weight", obs), ("critic.0.weight", priv)):
        w = sd[key]
        if w.shape[1] < want:
            sd[key] = torch.cat([w, torch.zeros(w.shape[0], want - w.shape[1], dtype=w.dtype)], 1)
    return sd


def load_policy(path, dev="cpu"):
    """the actor of a saved policy: obs → joint targets (deterministic), with its input scaling"""
    import torch
    st = torch.load(path, map_location=dev, weights_only=False)
    ac = make_ac(torch, st["obs_dim"], st["priv_dim"], st["act_dim"]).to(dev)
    ac.load_state_dict(st["ac"])
    ac.eval()
    on = Norm(torch, st["obs_dim"], dev)
    on.load(st["obs_norm"], dev)

    def act(obs):
        with torch.no_grad():
            return ac.actor(on(obs))
    return act, st


# ── what the 3D view gets (esasim.humanoid/1, with four feet): the robot once (its body tree), then frames ─────────────
FORMAT = "esasim.quadruped-policy/1"
TASK = "walk"
FEET = ["FL", "FR", "HL", "HR"]


def make_fields(nu):
    return (["t", "x", "y", "z", "qw", "qx", "qy", "qz"] + ["j%d" % k for k in range(nu)]
            + ["cvx", "cvy", "cwz", "vx", "vy", "wz", "c0", "c1", "c2", "c3", "f0", "f1", "f2", "f3"]
            + ["h", "gx", "gy", "gz", "ax", "ay", "az"] + ["tq%d" % k for k in range(nu)] + ["tt"])


FIELDS = make_fields(12)
# c0..c3: each foot on the ground (FL, FR, HL, HR); f0..f3: its load in body weights; h: the ground under the body (m);
# gx..gz: the body's gyro (rad/s); ax..az: its accelerometer (m/s²); tq: each motor's torque (N m)


GROUND_NAME = {"flat": "the flat floor", "course": "the straight test course", "circuit": "the obstacle course",
               "playground": "the playground", "warehouse": "the warehouse"}


def make_zones(c=COURSE):
    """the parts of the obstacle course, as boxes: (key, x0, x1, y0, y1, kind, what it is; %s: up or down)"""
    z = []
    for pa in c["patches"]:
        mx = max(pa["mm"]) / 10.0
        what = {"GRAVEL": "Gravel", "STEPPING": "Stepping stones"}.get(str(pa.get("label") or "").split(" ")[0], pa["kind"])
        z.append((pa["kind"], pa["x0"], pa["x0"] + pa["size"], pa["y0"], pa["y0"] + pa["size"], "rough",
                  "Rough ground: %s up to %.0f cm" % (what.lower() if pa.get("label") else what, mx)))
    rp = c.get("ramp")
    if rp:
        pct = 100.0 * rp["h"] / (rp["x"][1] - rp["x"][0])
        w = rp["half"]
        z.append(("slope_a", rp["x"][0], rp["x"][1], -w, w, "slope", "Slope %s: " + "%.1f %%%% (%.0f°)" % (pct, math.degrees(math.atan(pct / 100)))))
        z.append(("plateau", rp["x"][1], rp["x"][2], -w, w, "flat", "On the plateau, %.1f m up" % rp["h"]))
        z.append(("slope_b", rp["x"][2], rp["x"][3], -w, w, "slope", "Slope %s: " + "%.1f %%%%" % pct))
    sd = c.get("sand")
    if sd:
        z.append(("sand", sd["x0"], sd["x0"] + sd["size"], sd["y0"], sd["y0"] + sd["size"], "sand", "Sand: soft, the feet sink in"))
    wa = c.get("water")
    if wa:
        z.append(("water", wa["x0"], wa["x0"] + wa["size"], wa["y0"], wa["y0"] + wa["size"], "water", "Water, %.0f cm deep: it drags at the legs" % (100 * wa["depth"])))
    for st in c["stairs"]:
        if st["name"] == "curb":
            x0_, x1_ = sorted([st["x0"] - 0.4 * st["dir"], st["x0"] + st["dir"] * (st["landing"] + 0.4)])
            z.append(("curb%.0f" % (st["rise"] * 100), x0_, x1_, st["y"] - st["width"] / 2, st["y"] + st["width"] / 2, "curb", "A curb, %.0f cm" % (100 * st["rise"])))
            continue
        if st["name"] not in ("circuit", "steps"):
            continue
        up = (st["n"] - 1) * st["tread"]
        u_edges = [0.0, up, up + st["landing"], 2 * up + st["landing"]]
        xs = [st["x0"] + st["dir"] * u for u in u_edges]
        y0, y1 = st["y"] - st["width"] / 2, st["y"] + st["width"] / 2
        txt = "Stairs %s: " + "%d steps of %.0f cm" % (st["n"], 100 * st["rise"])
        z.append(("stairs_a", min(xs[0], xs[1]), max(xs[0], xs[1]), y0, y1, "stairs", txt))
        z.append(("landing", min(xs[1], xs[2]), max(xs[1], xs[2]), y0, y1, "flat", "On the landing, %.1f m up" % (st["n"] * st["rise"])))
        z.append(("stairs_b", min(xs[2], xs[3]), max(xs[2], xs[3]), y0, y1, "stairs", txt))
    ic = c.get("ice")
    if ic:
        z.append(("ice", ic["x0"], ic["x0"] + ic["size"], ic["y0"], ic["y0"] + ic["size"], "ice",
                  "Ice: smooth, almost no grip (friction %.2f)" % ic["mu"]))
    tl = c.get("tiles")
    if tl:
        S = tl["cell"] * tl["n"]
        z.append(("tiles", tl["x0"], tl["x0"] + S, tl["y0"], tl["y0"] + S, "rough",
                  "Moving tiles: each rises and falls by up to %.0f cm" % (200 * tl["amp"][1])))
    return z


ZONES = make_zones()
ZONES_PLAY = make_zones(PLAYGROUND)


def terrain_of(p, default):
    t = str(p.get("terrain") or default).lower()
    if "warehouse" in t:
        return "warehouse"
    if "playground" in t or "play" in t:
        return "playground"
    return "circuit" if ("circuit" in t or "course" in t or "rough" in t or "stairs" in t or "obstacle" in t) else "flat"


def robot_info(m, terrain="circuit", scan=True):
    """the body tree for the view: each body's parent, its place and turn on its parent, its joint (axis, slot in qpos)"""
    bodies = []
    for b in range(1, m.nbody):
        if m.body_mocapid[b] >= 0:
            continue                   # (the moving tiles: the view draws them from their rhythms)
        j = int(m.body_jntadr[b]) if m.body_jntnum[b] > 0 else -1
        jt = int(m.jnt_type[j]) if j >= 0 else -1
        bodies.append({"name": m.body(b).name, "parent": int(m.body_parentid[b]) - 1,
                       "pos": [round(float(x), 6) for x in m.body_pos[b]], "quat": [round(float(x), 6) for x in m.body_quat[b]],
                       "jtype": "free" if jt == 0 else "hinge" if jt == 3 else None,
                       "axis": [round(float(x), 4) for x in m.jnt_axis[j]] if jt == 3 else None,
                       "qadr": int(m.jnt_qposadr[j]) if j >= 0 else None})
    joints = [m.joint(i).name for i in range(1, m.njnt)]
    limits = [[round(float(m.jnt_range[i][0]), 3), round(float(m.jnt_range[i][1]), 3)] for i in range(1, m.njnt)]
    tmax = [round(float(abs(m.jnt_actfrcrange[int(m.actuator_trnid[a][0])][1])), 1) for a in range(m.nu)]
    R = ROBOTS[ROBOT]
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    feet = []
    for f in FEET:                     # each foot: the link it is on, how high that link's origin stands over the floor
        g = m.geom(f).id
        b = int(m.geom_bodyid[g])
        r = float(m.geom_size[g][0])
        feet.append({"name": f, "body": m.body(b).name, "sole": round(float(d.xpos[b][2] - d.geom_xpos[g][2] + r), 3),
                     "ahead": round(float(d.geom_xpos[g][0] - d.xpos[b][0]), 3), "size": [round(2 * r, 3), round(2 * r, 3)], "round": True})
    return {"model": ROBOT, "legs": 4, "feet": feet, "name": R["name"], "short": R["short"], "glb": R["view"], "credit": R["credit"], "h0": H0,
            "mass": round(float(m.body_subtreemass[m.body("pelvis").id]), 1),
            "bodies": bodies, "joints": joints, "limits": limits, "torque_max": tmax,
            "default": [round(float(x), 4) for x in m.key("knees_bent").qpos[7:]], "fields": FIELDS,
            "course": course_of(terrain), "scan": {"x": SCAN_X, "y": SCAN_Y, "base": round(H0, 3)} if scan else None,
            "camera": R["camera"]}


def frame_row(env, i, t):
    import torch
    q = env.sim.qpos[i].float().tolist()
    c = env.cmd[i].tolist()
    lin = env.S("local_linvel")[i].tolist()
    gy = env.S("gyro")[i].tolist()
    ct = [int(v) for v in env.contacts()[i].tolist()]
    ld = [round(float(env.S(f + "_touch")[i, 0]) / env.weight, 2) for f in FEET]
    r3, r4 = (lambda v: round(v, 3)), (lambda v: round(v, 4))
    return ([round(t, 3), r3(q[0]), r3(q[1]), r3(q[2])] + [r4(v) for v in q[3:7]] + [r3(v) for v in q[7:7 + NU]]
            + [round(c[0], 2), round(c[1], 2), round(c[2], 2), round(lin[0], 2), round(lin[1], 2), round(gy[2], 2)] + ct + ld
            + [r3(float(env.ground(env.sim.qpos[i:i + 1, 0].float(), env.sim.qpos[i:i + 1, 1].float(), w=torch.tensor([i], device=env.dev))[0]))] + [round(v, 2) for v in gy]
            + [round(v, 1) for v in env.S("accel")[i].tolist()] + [round(v, 1) for v in env.sim.tq[i].float().tolist()] + [round(float(env.tt[i]), 3)])




def tiles_item(env, i):
    """robot i's moving tiles, for the view (it moves them on the frames' tile clock tt)"""
    r = lambda t: [round(v, 4) for v in t[i].tolist()]   # noqa: E731
    return {"a": r(env.tile_a), "b": r(env.tile_b), "w": r(env.tile_w), "p": r(env.tile_p)}


def chunks(rows, budget=40000):
    """the frames in pieces small enough for a telemetry item (at most 60 KB each)"""
    import json
    per = max(1, len(json.dumps(rows[0]))) * 1.15 if rows else 1
    size = max(10, int(budget / per))
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size], i + size >= len(rows)


# ── policies to carry on from ─────────────────────────────────────────────────────────────────────────────────────
def policy_stamp(path, name):
    """when a policy was made: the date in its name (humanoid_walk_20261112_1926_64.pt), else its file's time"""
    import os
    import re
    import time
    m = re.search(r"(\d{8})_(\d{4})", str(name))
    if m:
        try:
            return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))
        except ValueError:
            pass
    return os.path.getmtime(path)


def find_policies(ctx, task, terrain, checkpoint):
    """The policies a training can carry on from: those wired in (the dataset), and those that stopped runs left on this
    machine (the checkpoint in their task folder, next to this one). Those that learned this ground come first, then the
    one with the most experience (the furthest along), then the newest."""
    import glob
    import os
    import torch
    out, others, foreign = [], [0], [0]

    def consider(path, name, where):
        try:
            st = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:     # noqa: BLE001
            ctx.log("Skipped %s: %s" % (name, e))
            return
        if st.get("format") != FORMAT:
            foreign[0] += 1         # (a checkpoint of another block, or of an older version of this one)
            return
        if (st.get("task") or TASK) != task:
            return                  # (a policy of the other skill: walking, or standing up)
        if (st.get("robot") or "spot") != ROBOT:
            others[0] += 1          # (another robot's: kept as it is, for that robot)
            return
        out.append({"same": 1 if (st.get("terrain") or "flat") == terrain else 0, "steps": int(st.get("steps") or 0),
                    "stamp": policy_stamp(path, name), "name": str(name), "where": where, "st": st, "path": path})
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            consider(f.path, f.name, "dataset")
    here = os.getcwd()
    jobs = os.path.dirname(os.path.dirname(here))
    if os.path.basename(jobs).lower() == "jobs":
        paths = sorted(glob.glob(os.path.join(jobs, "*", "*", checkpoint)), key=os.path.getmtime, reverse=True)[:30]
        for pth in paths:
            if os.path.dirname(os.path.abspath(pth)) == os.path.abspath(here):
                continue
            consider(pth, "the policy run #%s left on this machine" % os.path.basename(os.path.dirname(os.path.dirname(pth))), "machine")
    out.sort(key=lambda x: (x["same"], x["steps"], x["stamp"]), reverse=True)
    if others[0]:
        ctx.log("%d policies of other robots left as they are (each robot carries on from its own)." % others[0])
    if foreign[0]:
        ctx.log("Skipped %d checkpoints made by other blocks or older versions of this one." % foreign[0])
    return out


def start_over(p):
    return str(p.get("training") or "").lower().startswith("start") or bool(p.get("from_scratch"))


# ── learning to walk ─────────────────────────────────────────────────────────────────────────────────────────────────
def learn_run(ctx):
    import collections
    import csv
    import os
    import re
    import shutil
    import time
    import mujoco
    import torch
    p = ctx.params
    n = int(p.get("robots") or 4096)
    minutes = float(p.get("minutes") or 90)
    vmax = float(p.get("max_speed") or 1.2)
    replay_every = float(p.get("replay_every") or 20)
    seed = int(p.get("seed") or 1)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    terrain = terrain_of(p, "flat")
    if terrain in ("playground", "warehouse"):
        ctx.log("The warehouse is for the patrol (Surveillance): it learns on the obstacle course, whose ground has its kinds of obstacles (slopes, stairs, curbs) and more.")
        terrain = "circuit"
    torch.manual_seed(seed)
    mjm = load_model(mujoco, terrain)
    sim, label = make_sim(mjm, n, p.get("accelerator") or "auto", ctx.log)
    n = sim.n
    dev = sim.tdev
    env = WalkEnv(sim, mjm, seed=seed, vmax=vmax, push=float(p.get("push") if p.get("push") not in (None, "") else 1.0), terrain=terrain, scan=True)
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, NU, horizon=24)
    steps_done, warm = 0, 0
    found = [] if start_over(p) else find_policies(ctx, TASK, terrain, "policy_checkpoint.pt")
    if start_over(p):
        ctx.log("Starting over (Training: Start over): a new policy that learns from nothing.")
    elif found:
        pick = found[0]
        st = pick["st"]
        algo.load(st)
        steps_done = int(st.get("steps") or 0)
        if (st.get("terrain") or "flat") != terrain or int(st.get("obs_dim") or 0) < OBS_DIM:
            warm = 25          # new ground or new inputs: the critic learns what they are worth first, the walk is left as it is
        if int(st.get("obs_dim") or 0) < OBS_DIM:
            ctx.log("It learned blind: it now gets an elevation map of the ground ahead (from its cameras), which it learns to use.")
        ctx.log("Continuing from %s (walk score %.0f%% on %s, %.0f million steps of experience)." % (pick["name"], float(st.get("score") or 0), GROUND_NAME.get(st.get("terrain") or "flat", "?"), steps_done / 1e6))
    else:
        ctx.log("No policy to carry on from yet: a new one learns from nothing.")
    # The run's time limit (Stop after): learning ends a few minutes before it, so the policy is written and kept
    # rather than lost when the worker stops the block.
    left = ctx.time_left() if hasattr(ctx, "time_left") else None
    if left is not None and left / 60.0 - 3.0 < minutes:
        cut = max(1.0, left / 60.0 - 3.0)
        ctx.log("The run stops after %.0f minutes (Stop after): learning ends at %.0f minutes instead of %.0f, so the policy is kept. Raise Stop after to learn longer." % (ctx.max_minutes, cut, minutes))
        minutes = cut
    ctx.log("The %s learning to walk on %s, on %s: %d robots at once, for up to %.0f minutes." % (ROBOTS[ROBOT]["name"], GROUND_NAME[terrain], label, n, minutes))
    env.reset_all()
    a_obs, a_priv = env.obs()
    ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, terrain)})
    curve = open(ctx.output("curve", "training.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "steps", "walk_score", "fell_pct", "speed_err", "episode_s", "reward", "steps_per_s"])
    hist = collections.deque(maxlen=max(200, min(4000, n)))    # finished episodes: (fell, length s, return, distance)
    track = collections.deque(maxlen=200)                       # (tracking, speed error) per iteration
    t_start, last_tel, last_replay = time.time(), 0.0, -1e9
    it, best, stop_reason = 0, -1.0, "time"
    out_path = os.path.join(os.getcwd(), "policy_checkpoint.pt")
    rec, rec_ep = [], 0
    sim_t0 = 0.0

    def score():
        if len(hist) < 50 or not track:
            return 0.0
        fall = sum(h[0] for h in hist) / len(hist)
        tr = sum(x[0] for x in track) / len(track)
        return 100.0 * tr * (1.0 - fall)

    def save(note):
        st = algo.state()
        st.update({"format": FORMAT, "task": TASK, "robot": ROBOT, "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": NU, "score": score(),
                   "note": note, "steps": steps_done, "device": label, "minutes": (time.time() - t_start) / 60.0,
                   "action_scale": ACTION_SCALE, "default": env.default.cpu(), "vmax": vmax, "ctrl_dt": CTRL_DT, "terrain": terrain})
        torch.save(st, out_path)

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed > minutes * 60:
                break
            if any(isinstance(m.get("data") if isinstance(m, dict) else None, dict) and str(m["data"].get("control", "")).lower() in ("finish now", "finish")
                   for m in ctx.messages(ch_ctl)):
                stop_reason = "asked"
                break
            t0 = time.time()
            tr_sum, err_sum = 0.0, 0.0
            for t in range(algo.T):
                act = algo.act(a_obs, a_priv, t)
                rew, done, info = env.step(act)
                o2, p2 = env.obs()
                algo.record(t, rew, done, info["timeout"], p2)
                tr_sum += float(info["terms"]["tracking_lin_vel"].mean())
                err_sum += float((env.cmd[:, 0] - info["lin"][:, 0]).abs().mean())
                # one robot, replayed in 3D from time to time
                if it % 1 == 0:
                    if int(env.steps[0]) % 2 == 0:
                        rec.append(frame_row(env, 0, sim_t0 + float(env.steps[0]) * CTRL_DT))
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=False).flatten()
                    fell = info["fell"][idx].tolist()
                    lens = env.steps[idx].tolist()
                    rets = env.ep_ret[idx].tolist()
                    dist = env.ep_dist[idx].tolist()
                    for a_, b_, c_, d_ in zip(fell, lens, rets, dist):
                        hist.append((1.0 if a_ else 0.0, b_ * CTRL_DT, c_, d_))
                    if bool(done[0]):
                        if rec and time.time() - last_replay > replay_every and len(rec) > 25:
                            last_replay = time.time()
                            rec_ep += 1
                            res = {"fell": bool(info["fell"][0]), "seconds": round(float(env.steps[0]) * CTRL_DT, 1), "distance": round(float(env.ep_dist[0]), 1)}
                            lab = "Episode %d: %s after %.0f s, %.1f m along its line" % (rec_ep, "fell" if res["fell"] else "still walking", res["seconds"], res["distance"])
                            for k, part, last in chunks(rec):
                                ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "training", "ep": rec_ep, "seq": k, "last": last, "frames": part,
                                                       "result": res, "label": lab, "tiles": tiles_item(env, 0) if env.tiles_on else None, **({"path": int(env.pdir[0])} if bool(env.on_path[0]) else {"line": [round(float(env.line_y[0]), 3), round(float(env.line_yaw[0]), 4)]})})
                        rec = []
                    env.reset(idx)
                    sim.refresh()
                    o2, p2 = env.obs()
                a_obs, a_priv = o2, p2
            stats = algo.update(a_priv, critic_only=it < warm)
            if warm and it == warm - 1:
                ctx.log("The critic has learned the new ground (%d iterations): the policy starts to adapt." % warm)
            it += 1
            steps_done += algo.T * n
            sps = algo.T * n / max(1e-6, time.time() - t0)
            track.append((tr_sum / algo.T, err_sum / algo.T))
            sc = score()
            if sc > best and len(hist) >= 50 and it > 20:
                best = sc
                save("best so far")
            if time.time() - last_tel > 2.0 or it == 1:
                last_tel = time.time()
                fall = 100.0 * sum(h[0] for h in hist) / max(1, len(hist))
                eps = sum(h[1] for h in hist) / max(1, len(hist))
                ret = sum(h[2] for h in hist) / max(1, len(hist))
                dist = sum(h[3] for h in hist) / max(1, len(hist))
                err = track[-1][1]
                row = {"iteration": it, "minutes": round(elapsed / 60, 2), "steps_m": round(steps_done / 1e6, 2), "walk_score": round(sc, 1),
                       "fell": round(fall, 1), "speed_err": round(err, 3), "episode_s": round(eps, 1), "reward": round(ret, 2), "distance": round(dist, 1),
                       "steps_per_s": round(sps), "std": round(stats["std"], 3), "kl": round(stats["kl"], 4), "lr": stats["lr"]}
                if terrain == "circuit":
                    row["level"] = round(float(env.level.float().mean()), 2)          # (the curriculum: 0 flat … 4 the hardest)
                    row["hardest_pct"] = round(100.0 * float((env.level >= 4).float().mean()), 1)
                ctx.telemetry(ch_train, row)
                cw.writerow([row["minutes"], it, steps_done, row["walk_score"], row["fell"], row["speed_err"], row["episode_s"], row["reward"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "walk score %.0f%% · falls %.0f%% · %.0f M steps%s" % (sc, fall, steps_done / 1e6, (" · level %.1f of 4" % row["level"]) if "level" in row else ""))
    finally:
        curve.close()
    if best < 0 or score() >= best:
        save("final")
    name = "%s_walk_%s%s_%02d.pt" % (ROBOT, "course_" if terrain == "circuit" else "", time.strftime("%Y%m%d_%H%M"), int(round(max(0.0, best, score()))))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s (walk score %.0f%%: how well it keeps the asked speed, times the share of runs it does not fall)." % (name, max(0.0, best, score())))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ({"time": "time is up", "asked": "finished on request"}[stop_reason], it, steps_done / 1e6, (time.time() - t_start) / 60))
    return {"walk_score": round(max(0.0, best, score()), 1), "steps_m": round(steps_done / 1e6, 1)}


# ── learning to stand up after a fall ───────────────────────────────────────────────────────────────────────────────
ASSIST0 = 0.0          # the help it starts with (a quadruped rolls over and gets up without one)


def stand_run(ctx):
    import collections
    import csv
    import math
    import os
    import shutil
    import time
    import mujoco
    import torch
    p = ctx.params
    n = int(p.get("robots") or 4096)
    minutes = float(p.get("minutes") or 90)
    replay_every = float(p.get("replay_every") or 20)
    seed = int(p.get("seed") or 1)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    torch.manual_seed(seed)
    mjm = load_model(mujoco, "flat")
    sim, label = make_sim(mjm, n, p.get("accelerator") or "auto", ctx.log)
    n = sim.n
    dev = sim.tdev
    env = StandEnv(sim, mjm, seed=seed)
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, NU, horizon=32)
    steps_done, assist = 0, ASSIST0
    found = [] if start_over(p) else find_policies(ctx, "standup", "flat", "standup_checkpoint.pt")
    if start_over(p):
        ctx.log("Starting over (Training: Start over): a new policy that learns from nothing.")
    elif found:
        st = found[0]["st"]
        algo.load(st)
        steps_done = int(st.get("steps") or 0)
        assist = min(ASSIST0, float(st.get("assist") if st.get("assist") is not None else ASSIST0))
        ctx.log("Continuing from %s (stand-up score %.0f%%, %.0f million steps of experience)." % (found[0]["name"], float(st.get("score") or 0), steps_done / 1e6))
    else:
        ctx.log("No stand-up policy to carry on from yet: a new one learns from nothing.")
    left = ctx.time_left() if hasattr(ctx, "time_left") else None
    if left is not None and left / 60.0 - 3.0 < minutes:
        cut = max(1.0, left / 60.0 - 3.0)
        ctx.log("The run stops after %.0f minutes (Stop after): learning ends at %.0f minutes instead of %.0f, so the policy is kept. Raise Stop after to learn longer." % (ctx.max_minutes, cut, minutes))
        minutes = cut
    rounds = max(2, min(12, int(math.ceil(6000.0 / n))))
    env.build_bank(rounds, ctx.log)
    env.assist = assist
    ctx.log("The " + ROBOTS[ROBOT]["name"] + " learning to stand and to self-right on the flat floor, on %s: %d robots at once, for up to %.0f minutes. Each try starts lying where a fall left it (on its side, on its back), with %.0f s to roll over, get up and stay up." % (label, n, minutes, STAND_EP * CTRL_DT))
    if assist > 0:
        ctx.log("At first an upward help on its body (%.0f%% of its weight) makes it possible; the help goes down as it succeeds, to none." % (100 * assist))
    env.reset_all()
    a_obs, a_priv = env.obs()
    ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, "flat")})
    curve = open(ctx.output("curve", "standup.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "steps", "stand_score", "up_pct", "time_to_up_s", "assist_pct", "reward", "steps_per_s"])
    hist = collections.deque(maxlen=max(300, min(4000, n)))     # finished tries from the floor: (up at the end, seconds to get up, return)
    t_start, last_tel, last_replay, last_cut = time.time(), 0.0, -1e9, 0
    it, best, stop_reason = 0, -1.0, "time"
    out_path = os.path.join(os.getcwd(), "standup_checkpoint.pt")
    rec, rec_ep = [], 0

    def up_rate():
        return sum(h[0] for h in hist) / len(hist) if hist else 0.0

    def score():
        if len(hist) < 100:
            return 0.0
        return 100.0 * up_rate() * ((1.0 - env.assist / ASSIST0) if ASSIST0 > 0 else 1.0)

    def save(note):
        st = algo.state()
        st.update({"format": FORMAT, "task": "standup", "robot": ROBOT, "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": NU, "score": score(),
                   "note": note, "steps": steps_done, "device": label, "minutes": (time.time() - t_start) / 60.0, "assist": env.assist,
                   "up_pct": 100.0 * up_rate(), "action_scale": ACTION_SCALE, "default": env.default.cpu(), "ctrl_dt": CTRL_DT, "terrain": "flat"})
        torch.save(st, out_path)

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed > minutes * 60:
                break
            if any(isinstance(m.get("data") if isinstance(m, dict) else None, dict) and str(m["data"].get("control", "")).lower() in ("finish now", "finish")
                   for m in ctx.messages(ch_ctl)):
                stop_reason = "asked"
                break
            t0 = time.time()
            for t in range(algo.T):
                act = algo.act(a_obs, a_priv, t)
                rew, done, info = env.step(act)
                o2, p2 = env.obs()
                algo.record(t, rew, done, info["timeout"], p2)
                if int(env.steps[0]) % 2 == 0:
                    rec.append(frame_row(env, 0, float(env.steps[0]) * CTRL_DT))
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=False).flatten()
                    up = (env.up_steps[idx] >= 25) & ~info["fell"][idx]
                    for u_, tu, r_, s_ in zip(up.tolist(), env.t_up[idx].tolist(), env.ep_ret[idx].tolist(), env.start_standing[idx].tolist()):
                        if not s_:
                            hist.append((1.0 if u_ else 0.0, tu if tu >= 0 else None, r_))
                    if bool(done[0]):
                        if rec and time.time() - last_replay > replay_every and len(rec) > 25 and not bool(env.start_standing[0]):
                            last_replay = time.time()
                            rec_ep += 1
                            ok, tu = bool(up[0]), float(env.t_up[0])
                            res = {"fell": not ok, "seconds": round(float(env.steps[0]) * CTRL_DT, 1), "distance": 0.0}
                            lab = ("Try %d: up %sin %.1f s" % (rec_ep, "" if env.assist > 0.005 else "on its own ", tu) if ok and tu >= 0 else "Try %d: still down after %.0f s" % (rec_ep, res["seconds"])) + \
                                  (" (help %.0f%% of its weight)" % (100 * env.assist) if env.assist > 0.005 else "")
                            for k, part, last in chunks(rec):
                                ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "training", "ep": rec_ep, "seq": k, "last": last, "frames": part,
                                                       "result": res, "label": lab})
                        rec = []
                    env.reset(idx)
                    sim.refresh()
                    o2, p2 = env.obs()
                a_obs, a_priv = o2, p2
            stats = algo.update(a_priv)
            it += 1
            steps_done += algo.T * n
            sps = algo.T * n / max(1e-6, time.time() - t0)
            # less help as it succeeds: more than 60% of the tries from the floor up and staying up
            if env.assist > 0 and len(hist) >= min(300, max(100, n)) and it - last_cut >= 10 and up_rate() > 0.6:
                env.assist = max(0.0, env.assist - 0.05) if env.assist > 0.06 else 0.0
                last_cut = it
                hist.clear()
                ctx.log("It gets up in most tries: the help goes down to %.0f%% of its weight." % (100 * env.assist) if env.assist > 0 else
                        "It gets up with almost no help: from now on it gets up on its own, with no help at all.")
            sc = score()
            if sc > best and len(hist) >= 100 and it > 20:
                best = sc
                save("best so far")
            if time.time() - last_tel > 2.0 or it == 1:
                last_tel = time.time()
                ups = [h[1] for h in hist if h[0] and h[1] is not None]
                row = {"iteration": it, "minutes": round(elapsed / 60, 2), "steps_m": round(steps_done / 1e6, 2), "stand_score": round(sc, 1),
                       "up_pct": round(100.0 * up_rate(), 1), "time_to_up": round(sum(ups) / len(ups), 2) if ups else None,
                       "assist": round(100.0 * env.assist, 1), "reward": round(sum(h[2] for h in hist) / max(1, len(hist)), 2),
                       "steps_per_s": round(sps), "std": round(stats["std"], 3), "kl": round(stats["kl"], 4), "lr": stats["lr"]}
                ctx.telemetry(ch_train, row)
                cw.writerow([row["minutes"], it, steps_done, row["stand_score"], row["up_pct"], row["time_to_up"], row["assist"], row["reward"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "stand-up score %.0f%% · up %.0f%% · help %.0f%%" % (sc, row["up_pct"], row["assist"]))
    finally:
        curve.close()
    if best < 0 or score() >= best:
        save("final")
    final = max(0.0, best, score())
    name = "%s_standup_%s_%02d.pt" % (ROBOT, time.strftime("%Y%m%d_%H%M"), int(round(final)))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s (stand-up score %.0f%%: the share of falls it gets up from and stays up)." % (name, final))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ({"time": "time is up", "asked": "finished on request"}[stop_reason], it, steps_done / 1e6, (time.time() - t_start) / 60))
    return {"stand_score": round(final, 1), "help_pct": round(100 * env.assist, 1), "steps_m": round(steps_done / 1e6, 1)}






# ── learning to look around ─────────────────────────────────────────────────────────────────────────────────────────
def look_run(ctx):
    import collections
    import csv
    import os
    import shutil
    import time
    import mujoco
    import torch
    p = ctx.params
    n = int(p.get("robots") or 4096)
    minutes = float(p.get("minutes") or 60)
    replay_every = float(p.get("replay_every") or 20)
    seed = int(p.get("seed") or 1)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    torch.manual_seed(seed)
    mjm = load_model(mujoco, "flat")
    sim, label = make_sim(mjm, n, p.get("accelerator") or "auto", ctx.log)
    n = sim.n
    dev = sim.tdev
    env = LookEnv(sim, mjm, seed=seed)
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, NU, horizon=32)
    steps_done = 0
    found = [] if start_over(p) else find_policies(ctx, "look", "flat", "look_checkpoint.pt")
    if start_over(p):
        ctx.log("Starting over (Training: Start over): a new policy that learns from nothing.")
    elif found:
        st = found[0]["st"]
        algo.load(st)
        steps_done = int(st.get("steps") or 0)
        ctx.log("Continuing from %s (look score %.0f%%, %.0f million steps of experience)." % (found[0]["name"], float(st.get("score") or 0), steps_done / 1e6))
    else:
        ctx.log("No look-around policy to carry on from yet: a new one learns from nothing.")
    left = ctx.time_left() if hasattr(ctx, "time_left") else None
    if left is not None and left / 60.0 - 3.0 < minutes:
        minutes = max(1.0, left / 60.0 - 3.0)
        ctx.log("The run stops after %.0f minutes (Stop after): learning ends at %.0f minutes, so the policy is kept." % (ctx.max_minutes, minutes))
    R = LOOK_RANGE
    ctx.log("The " + ROBOTS[ROBOT]["name"] + " learning to look around, on %s: %d robots at once, for up to %.0f minutes. Standing, its feet "
            "where they are, it is asked to aim its front cameras: nose up or down by up to %.0f°, turned left or right by up to %.0f°, "
            "%.0f cm lower to %.0f cm higher; a new aim every 1.5 to 4 s. It does it with its legs alone, without slipping or stepping."
            % (label, n, minutes, math.degrees(R["pitch"]), math.degrees(R["yaw"]), -100 * R["low"], 100 * R["high"]))
    env.reset_all()
    a_obs, a_priv = env.obs()
    ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, "flat")})
    curve = open(ctx.output("curve", "look.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "steps", "look_score", "pitch_err_deg", "yaw_err_deg", "height_err_cm", "fell_pct", "steps_per_s"])
    hist = collections.deque(maxlen=max(300, min(4000, n)))     # finished tries: (tracking 0..1, fell, |pitch err|, |yaw err|, |height err|)
    t_start, last_tel, last_replay = time.time(), 0.0, -1e9
    it, best, stop_reason = 0, -1.0, "time"
    out_path = os.path.join(os.getcwd(), "look_checkpoint.pt")
    rec, rec_ep = [], 0

    def mean(k):
        return sum(h[k] for h in hist) / len(hist) if hist else 0.0

    def score():
        if len(hist) < 100:
            return 0.0
        return 100.0 * mean(0) * (1.0 - mean(1))

    def save(note):
        st = algo.state()
        st.update({"format": FORMAT, "task": "look", "robot": ROBOT, "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": NU, "score": score(),
                   "note": note, "steps": steps_done, "device": label, "minutes": (time.time() - t_start) / 60.0, "look_range": LOOK_RANGE,
                   "action_scale": ACTION_SCALE * 1.5, "default": env.default.cpu(), "ctrl_dt": CTRL_DT, "terrain": "flat"})
        torch.save(st, out_path)

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed > minutes * 60:
                break
            if any(isinstance(m.get("data") if isinstance(m, dict) else None, dict) and str(m["data"].get("control", "")).lower() in ("finish now", "finish")
                   for m in ctx.messages(ch_ctl)):
                stop_reason = "asked"
                break
            t0 = time.time()
            for t in range(algo.T):
                act = algo.act(a_obs, a_priv, t)
                rew, done, info = env.step(act)
                o2, p2 = env.obs()
                algo.record(t, rew, done, info["timeout"], p2)
                if int(env.steps[0]) % 2 == 0:
                    rec.append(frame_row(env, 0, float(env.steps[0]) * CTRL_DT))
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=False).flatten()
                    cnt = (env.steps[idx].float() - 50).clamp(min=1)
                    for tr, fl, e3 in zip((env.track_sum[idx] / cnt).tolist(), info["fell"][idx].tolist(), (env.err_sum[idx] / cnt.unsqueeze(-1)).tolist()):
                        hist.append((tr, 1.0 if fl else 0.0, e3[0], e3[1], e3[2]))
                    if bool(done[0]):
                        if rec and time.time() - last_replay > replay_every and len(rec) > 25:
                            last_replay = time.time()
                            rec_ep += 1
                            c0 = max(1.0, float(env.steps[0]) - 50)
                            e3 = (env.err_sum[0] / c0).tolist()
                            res = {"fell": bool(info["fell"][0]), "seconds": round(float(env.steps[0]) * CTRL_DT, 1), "distance": 0.0}
                            lab = "Look %d: %s" % (rec_ep, "it fell" if res["fell"] else "aim held within %.0f° up/down, %.0f° left/right, %.0f cm" % (
                                math.degrees(e3[0]), math.degrees(e3[1]), 100 * e3[2]))
                            for k, part, last in chunks(rec):
                                ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "training", "ep": rec_ep, "seq": k, "last": last, "frames": part,
                                                       "result": res, "label": lab})
                        rec = []
                    env.reset(idx)
                    sim.refresh()
                    o2, p2 = env.obs()
                a_obs, a_priv = o2, p2
            stats = algo.update(a_priv)
            it += 1
            steps_done += algo.T * n
            sps = algo.T * n / max(1e-6, time.time() - t0)
            sc = score()
            if sc > best and len(hist) >= 100 and it > 20:
                best = sc
                save("best so far")
            if time.time() - last_tel > 2.0 or it == 1:
                last_tel = time.time()
                row = {"iteration": it, "minutes": round(elapsed / 60, 2), "steps_m": round(steps_done / 1e6, 2), "look_score": round(sc, 1),
                       "pitch_err": round(math.degrees(mean(2)), 1), "yaw_err": round(math.degrees(mean(3)), 1), "height_err": round(100 * mean(4), 1),
                       "fell": round(100 * mean(1), 1), "steps_per_s": round(sps), "std": round(stats["std"], 3), "kl": round(stats["kl"], 4), "lr": stats["lr"]}
                ctx.telemetry(ch_train, row)
                cw.writerow([row["minutes"], it, steps_done, row["look_score"], row["pitch_err"], row["yaw_err"], row["height_err"], row["fell"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "look score %.0f%% · error %.0f° / %.0f° / %.0f cm" % (sc, row["pitch_err"], row["yaw_err"], row["height_err"]))
    finally:
        curve.close()
    if best < 0 or score() >= best:
        save("final")
    final = max(0.0, best, score())
    name = "%s_look_%s_%02d.pt" % (ROBOT, time.strftime("%Y%m%d_%H%M"), int(round(final)))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s (look score %.0f%%: how closely it holds the aims asked, less the tries it fell in)." % (name, final))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ({"time": "time is up", "asked": "finished on request"}[stop_reason], it, steps_done / 1e6, (time.time() - t_start) / 60))
    return {"look_score": round(final, 1), "steps_m": round(steps_done / 1e6, 1)}


LEG_PART = {"pelvis": "body", "fl_hip": "front-left hip", "fr_hip": "front-right hip", "hl_hip": "hind-left hip", "hr_hip": "hind-right hip",
            "fl_uleg": "front-left thigh", "fr_uleg": "front-right thigh", "hl_uleg": "hind-left thigh", "hr_uleg": "hind-right thigh",
            "fl_lleg": "front-left lower leg", "fr_lleg": "front-right lower leg", "hl_lleg": "hind-left lower leg", "hr_lleg": "hind-right lower leg"}


def zone_at(x, y, zones=None):
    """the part of the circuit (or of the playground) at x, y: (key, kind, what it is)"""
    for z in (zones or ZONES):
        if z[1] <= x <= z[2] and z[3] <= y <= z[4]:
            return z
    return None


def zone_text(z, climb):
    kind = z[5]
    if kind in ("slope", "stairs"):
        return z[6] % ("up" if climb > 0 else "down")
    return z[6]


def walk_run(ctx):
    import os
    import re
    import time
    import math
    import mujoco
    import torch
    p = ctx.params
    torch.set_num_threads(1)
    pols, ups, others = [], [], 0
    R = ROBOTS[ROBOT]
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                h = torch.load(f.path, map_location="cpu", weights_only=False)
                if h.get("format") == FORMAT and (h.get("robot") or "spot") != ROBOT:
                    others += 1
                    continue
                if h.get("format") == FORMAT and h.get("task") == "standup":
                    ups.append((policy_stamp(f.path, f.name), f))
                    continue
                if h.get("format") != FORMAT or h.get("task") != TASK:
                    ctx.log("Skipped %s: made for another task or version." % f.name)
                    continue
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                pols.append((m_.group(1) if m_ else "", os.path.getmtime(f.path), f))
            except Exception as e:     # noqa: BLE001
                ctx.log("Skipped %s: %s" % (f.name, e))
    if others:
        ctx.log("%d policies of other robots left aside: the %s walks with its own." % (others, R["name"]))
    if not pols:
        raise RuntimeError("No walking policy for the %s yet: run Learn to walk first (it writes %s_walk_*.pt to its dataset), and wire that dataset to Policies." % (R["name"], ROBOT))
    pols.sort(key=lambda x: (x[0], x[1]), reverse=True)
    want = str(p.get("policy_file") or "").strip()
    pick = [x for x in pols if want and want in str(x[2].name)]
    pol = (pick or pols)[0][2]
    act, st = load_policy(pol.path)
    sees = int(st.get("obs_dim") or OBS_BASE) > OBS_BASE
    walk_dim = int(st.get("obs_dim") or OBS_BASE)
    up_act, up_st = None, None
    if ups and str(p.get("after_fall") or "Get up on its own").lower().startswith("get up"):
        ups.sort(key=lambda x: x[0], reverse=True)
        up_act, up_st = load_policy(ups[0][1].path)
        ctx.log("Stand-up policy %s: stand-up score %.0f%%%s. After a fall it gets up on its own (put back on its feet if it is still down after 10 s)." % (
            ups[0][1].name, float(up_st.get("score") or 0), (", learned with help (%.0f%% of its weight) still on" % (100 * float(up_st.get("assist") or 0))) if float(up_st.get("assist") or 0) > 0.005 else ""))
    elif not ups:
        ctx.log("No stand-up policy among the policies: after a fall it is put back on its feet (Learn to stand up teaches it to get up on its own).")
    terrain = terrain_of(p, "circuit")
    ctx.log("Walking policy %s: walk score %.0f%% on %s, %.0f million steps of experience on %s; %s." % (
        pol.name, float(st.get("score") or 0), GROUND_NAME.get(st.get("terrain") or "flat", "?"), float(st.get("steps") or 0) / 1e6,
        st.get("device") or "?", "it sees the ground ahead (elevation map from its cameras)" if sees else "it is blind (it feels the ground only through its feet)"))
    if terrain in ("circuit", "playground") and (st.get("terrain") or "flat") != "circuit":
        ctx.log("This policy has not learned on the obstacle course: rough ground, slopes, stairs and ice are new to it (Learn the obstacles teaches them).")
    ch, ch_ev, ch_ctl = p.get("robot_channel") or "robot", p.get("events_channel") or "event", p.get("control_channel") or "control"
    mjm = load_model(mujoco, terrain)
    sim = CpuSim(mjm, 1, threads=1)
    env = WalkEnv(sim, mjm, seed=int(time.time()) % 100000, vmax=1.5, push=0.0, noise=0.0, train=False, terrain=terrain, scan=sees or up_act is not None)
    env.new_tiles(torch.zeros(1, dtype=torch.long), seed=20261112)       # (the same tiles every walk)
    ctx.telemetry(ch, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, terrain, sees)})
    circuit = terrain == "circuit"
    play = terrain == "playground"      # (driven by hand: the view's Drive pad or the keyboard sets where it walks)
    if play:
        sim.water = PLAYGROUND.get("water")
    drive = [0.0, 0.0, 0.0]               # forward, sideways, turn: -1 to 1 each
    drive_at = 0.0
    keep_yaw = None                       # the heading it holds while not told to turn (a policy left alone drifts round)
    up_yaw = None                         # its heading the last time it was up (to stand it up again facing the same way)
    speed = float(p.get("speed") if p.get("speed") not in (None, "") else 0.6)
    heading = math.radians(float(p.get("heading") or 0.0))
    minutes = float(p.get("minutes") or 30)
    dirn = 1 if math.cos(heading) >= 0 else -1          # on the circuit: counterclockwise, or (heading beyond 90°) the other way
    one = torch.zeros(1, dtype=torch.long)

    def set_line(h):
        q = sim.qpos[0]
        env.line_yaw[:] = h
        env.line_y[:] = -float(q[0]) * math.sin(h) + float(q[1]) * math.cos(h)

    def start(at=None, i=None):
        """on its feet: at the start line, or (at) where it fell, on its line (the playground: at its start, or where it fell)"""
        nonlocal keep_yaw
        keep_yaw = None
        if play:
            env.reset(one, at=at or list(PLAYGROUND["start"]))
            env.mode_line[:] = False
            env.cmd[:] = 0.0
            env.speed[:] = 0.0
            return
        if circuit:
            i = 0 if i is None else i
            px, py = float(env.P[i, 0]), float(env.P[i, 1])
            env.reset(one, at=[px, py, float(env.PH[i]) + (math.pi if dirn < 0 else 0.0)])
            env.put_on_path(one, i, dirn)
        else:
            env.reset(one, at=at)
            if at is None:
                set_line(heading)
            else:
                env.line_yaw[:] = heading
        env.mode_line[:] = True
        env.speed[:] = 0.0
    start()
    t_wall0 = time.time()
    sim_t, run, frames, last_send, fell_at = 0.0, 1, [], time.time(), None
    getting_up, up_n = False, 0

    def to_ground():
        """getting up: no command and no gait clock (as it learned it)"""
        env.mode_line[:] = False
        env.speed[:] = 0.0
        env.cmd[:] = 0.0
        env.phase_dt[:] = 0.0
        env.phase[:] = torch.tensor([0.0, math.pi])

    def back_to_walking():
        """up: walking again, from where it got up (on its line, or on the circuit from the point it reached)"""
        env.phase_dt[:] = 2 * math.pi * CTRL_DT * 1.8
        env.phase[:] = torch.tensor([0.0, math.pi])
        if play:
            env.mode_line[:] = False
            return
        if circuit:
            env.put_on_path(one, int(env.pidx[0]), dirn)        # (its point of the circuit, followed while it got up)
        else:
            ly, lyaw = float(env.line_y[0]), float(env.line_yaw[0])
            env.line_yaw[:] = lyaw
            env.line_y[:] = ly
        env.mode_line[:] = True
    zone, lap, lap_t0, falls_lap, s_last = None, 1, 0.0, 0, 0
    hold = 1.5        # s standing still before it sets off
    push_v = 0.8      # a push by hand: as much as a shove of 0.8 m/s to the whole body (Push strength changes it)
    mass = float(mjm.body_subtreemass[mjm.body("pelvis").id])

    def push_by_hand(v, dv):
        """a push on the part of the body clicked in the view: {body, at: the point in that body's frame, dir: the way (world)};
        a force for 0.1 s that gives the whole robot dv m/s, at that point (so it also turns it); what to say, or None"""
        if not isinstance(v, dict) or not v.get("body"):
            return None
        try:
            b = mjm.body(str(v["body"])).id
            loc = [float(x) for x in (v.get("at") or [0, 0, 0])][:3]
            dx, dy, dz = [float(x) for x in (v.get("dir") or [1, 0, 0])][:3]
        except (KeyError, TypeError, ValueError):
            return None
        n_ = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        dx, dy, dz = dx / n_, dy / n_, dz / n_
        F = mass * dv / 0.1
        sim.push_at(0, b, loc, [F * dx, F * dy, F * dz], 0.1)
        q = sim.qpos[0]
        yaw = math.atan2(2 * (float(q[3]) * float(q[6]) + float(q[4]) * float(q[5])), 1 - 2 * (float(q[5]) ** 2 + float(q[6]) ** 2))
        fwd, left = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
        frm = ("behind" if fwd > 0 else "the front") if abs(fwd) >= abs(left) else ("the right" if left > 0 else "the left")
        part = re.sub(r"_(link|col)$", "", str(v["body"]))
        return "A push on its %s, from %s (%.1f m/s)" % (LEG_PART.get(part, part), frm, dv)
    ctx.telemetry(ch_ev, {"text": ("The %s is on the obstacle course (%s): it sets off at %.1f m/s" % (R["short"], "counterclockwise" if dirn > 0 else "clockwise", speed)) if circuit
                          else ("The %s is in the playground: drive it with the arrows (or W A S D, Q E to step aside) or the buttons" % R["short"]) if play
                          else "The %s is on the floor: it sets off at %.1f m/s" % (R["short"], speed)})
    o, _ = env.obs(noisy=False)
    while time.time() - t_wall0 < minutes * 60:
        # the controls
        for msg in ctx.messages(ch_ctl):
            d = msg.get("data") if isinstance(msg, dict) else None
            if not isinstance(d, dict):
                continue
            c, v = str(d.get("control", "")).strip().lower(), d.get("value")
            if c == "speed":
                try:
                    speed = max(0.0, min(1.6, float(v)))
                except (TypeError, ValueError):
                    pass
            elif c in ("heading", "turn", "direction"):
                try:
                    heading = math.radians(float(v))
                except (TypeError, ValueError):
                    continue
                if circuit:
                    nd = 1 if math.cos(heading) >= 0 else -1
                    if nd != dirn:
                        dirn = nd
                        env.pdir[:] = float(dirn)
                        ctx.telemetry(ch_ev, {"text": "Turning round: now %s" % ("counterclockwise" if dirn > 0 else "clockwise")})
                else:
                    set_line(heading)
                    ctx.telemetry(ch_ev, {"text": "New heading: %.0f°" % math.degrees(heading)})
            elif c == "drive":            # {x: forward, y: to the left, turn: to the left}, each -1 to 1 (held keys or buttons)
                if isinstance(v, dict):
                    try:
                        drive = [max(-1.0, min(1.0, float(v.get(k_) or 0.0))) for k_ in ("x", "y", "turn")]
                        drive_at = time.time()
                    except (TypeError, ValueError):
                        pass
            elif c in ("push strength", "push_strength", "push speed"):
                try:
                    push_v = max(0.1, min(3.0, float(v)))
                except (TypeError, ValueError):
                    pass
            elif c == "push":
                hand = push_by_hand(v, push_v)
                if hand:
                    ctx.telemetry(ch_ev, {"text": hand})
                else:            # (a plain Push: a shove of the whole body, from a random side)
                    ang = env.U(1, 0.0, 2 * math.pi)
                    sim.kick(torch.stack([torch.cos(ang), torch.sin(ang)], -1) * push_v)
                    wh = float(env.path_heading()[0])
                    ctx.telemetry(ch_ev, {"text": "A push (%.1f m/s, from the %s)" % (push_v, "left" if math.sin(float(ang) - wh) > 0 else "right")})
            elif c in ("reset", "stand up", "restart", "back to the start"):
                getting_up, fell_at, keep_yaw = False, None, None
                start()
                sim_t += 0.5
                run += 1
                hold = 1.5
                zone, lap, lap_t0, falls_lap, s_last = None, 1, sim_t, 0, 0
                ctx.telemetry(ch_ev, {"text": "Back at the start line"})
        t_due = t_wall0 + sim_t
        now = time.time()
        if now < t_due:
            time.sleep(min(0.01, t_due - now))
            continue
        # one control step: 20 ms of the robot's time
        if getting_up:
            to_ground()
            a = up_act(o)
        else:
            env.speed[:] = 0.0 if hold > 0 else speed
            if play:                      # driven: forward up to Walking speed, back at half of it, 0.35 m/s aside, 0.7 rad/s round
                if time.time() - drive_at > 1.2:
                    drive = [0.0, 0.0, 0.0]          # (no word from the view for a while: it stops)
                fx = drive[0] * (speed if drive[0] > 0 else 0.5 * speed)
                env.mode_line[:] = False
                q_ = sim.qpos[0]
                yaw_ = math.atan2(2 * (float(q_[3]) * float(q_[6]) + float(q_[4]) * float(q_[5])), 1 - 2 * (float(q_[5]) ** 2 + float(q_[6]) ** 2))
                if float(q_[2]) - float(env.ground(q_[0:1].float(), q_[1:2].float())[0]) > 0.3:
                    up_yaw = yaw_
                if abs(drive[2]) > 1e-3 or keep_yaw is None or fell_at is not None:
                    keep_yaw, wz_ = yaw_, 0.7 * drive[2]      # (turning: the heading to keep is where it ends up)
                else:                                     # not told to turn: it keeps its heading (a gentle correction, nothing within 3°)
                    err = math.atan2(math.sin(keep_yaw - yaw_), math.cos(keep_yaw - yaw_))
                    wz_ = 0.0 if abs(err) < math.radians(3) else max(-0.5, min(0.5, 1.5 * err))
                tgt = (0.0, 0.0, 0.0) if hold > 0 else (fx, 0.35 * drive[1], wz_)
                for k_, rate in ((0, 2.0), (1, 1.5), (2, 3.5)):      # (eased in and out over a few steps, as a joystick would: no jolts; a release stops it within 0.2 s)
                    cur = float(env.cmd[0, k_])
                    env.cmd[0, k_] = cur + max(-rate * CTRL_DT, min(rate * CTRL_DT, tgt[k_] - cur))
            hold -= CTRL_DT
            a = act(o[:, :walk_dim])
        rew, done, info = env.step(a)
        sim_t += CTRL_DT
        o, _ = env.obs(noisy=False)
        if int(env.steps[0]) % 2 == 0:
            frames.append(frame_row(env, 0, sim_t))
        if bool(info["fell"][0]) and fell_at is None:
            fell_at = sim_t
            keep_yaw = None               # (up again: it keeps the heading it gets up with)
            falls_lap += 1
            where = (" on " + zone_text(zone, 0).split(":")[0].split(" (")[0].lower().replace(" down", "").replace(" up", "")) if zone else ""
            if up_act is not None:
                getting_up, up_n = True, 0
                to_ground()
                ctx.telemetry(ch_ev, {"text": "It fell%s: it gets up on its own" % where})
            else:
                ctx.telemetry(ch_ev, {"text": "It fell%s: back up in 3 s" % where})
        if getting_up:
            hq = float(env.height()[0])
            tz_ = float(env.S("upvector")[0, 2])
            up_n = up_n + 1 if (hq > 0.32 and tz_ > 0.85) else 0
            if up_n >= 25:           # up and steady for half a second: walking again
                ctx.telemetry(ch_ev, {"text": "Up on its own in %.1f s: walking on" % (sim_t - fell_at)})
                getting_up, fell_at = False, None
                back_to_walking()
                hold = 0.5
        if fell_at is not None and sim_t - fell_at > (10.0 if getting_up else 3.0):
            if getting_up:
                ctx.telemetry(ch_ev, {"text": "Still down after 10 s: put back on its feet"})
            getting_up = False
            back_to_walking()
            fell_at = None
            if circuit:      # back on its feet on the circuit's line, a step further on
                start(i=(int(env.pidx[0]) + 4 * dirn) % env.M)
            elif play:       # back on its feet where it fell, facing the way it was going
                q = sim.qpos[0]
                start([float(q[0]), float(q[1]), up_yaw if up_yaw is not None else 0.0])
            else:
                q = sim.qpos[0]
                ly, lyaw = float(env.line_y[0]), float(env.line_yaw[0])
                s_ = float(q[0]) * math.cos(lyaw) + float(q[1]) * math.sin(lyaw) + 0.4
                start([s_ * math.cos(lyaw) - ly * math.sin(lyaw), s_ * math.sin(lyaw) + ly * math.cos(lyaw), lyaw])
                env.line_y[:] = ly
            run += 1
            hold = 1.0
        # round the circuit: what it walks on now (said once as it gets there), a lap each time it crosses the start line
        if play and fell_at is None:      # the playground: what it walks on now, said once as it gets there
            q = sim.qpos[0]
            z_ = zone_at(float(q[0]), float(q[1]), ZONES_PLAY)
            if z_ is not zone:
                zone = z_
                if z_ is not None:
                    yw = math.atan2(2 * (float(q[3]) * float(q[6]) + float(q[4]) * float(q[5])), 1 - 2 * (float(q[5]) ** 2 + float(q[6]) ** 2))
                    climb = float(env.ground(q[0:1].float() + 0.6 * math.cos(yw), q[1:2].float() + 0.6 * math.sin(yw))[0] - env.ground(q[0:1].float(), q[1:2].float())[0])
                    ctx.telemetry(ch_ev, {"text": zone_text(z_, climb)})
        if circuit and fell_at is None:
            q = sim.qpos[0]
            x_, y_ = float(q[0]), float(q[1])
            z_ = zone_at(x_, y_)
            if z_ is not zone:
                zone = z_
                if z_ is not None:
                    wh = float(env.path_heading()[0])
                    climb = float(env.ground(q[0:1].float() + 0.6 * math.cos(wh), q[1:2].float() + 0.6 * math.sin(wh))[0] - env.ground(q[0:1].float(), q[1:2].float())[0])
                    ctx.telemetry(ch_ev, {"text": zone_text(z_, climb)})
            s_now = int(env.pidx[0])
            if hold <= 0 and abs(s_now - s_last) > env.M // 2:        # it went over the start line
                ctx.telemetry(ch_ev, {"text": "Lap %d in %.0f s, %s" % (lap, sim_t - lap_t0, "no falls" if not falls_lap else "%d fall%s" % (falls_lap, "s" if falls_lap > 1 else ""))})
                lap, lap_t0, falls_lap = lap + 1, sim_t, 0
            s_last = s_now
        if int(env.steps[0]) >= EP_STEPS - 1:
            env.steps[:] = 0          # (walking on: the episode clock only matters in training)
        if time.time() - last_send > (0.1 if play else 0.25) and frames:      # (driven by hand: sent more often, so the view keeps up)
            last_send = time.time()
            item = {"format": "esasim.humanoid/1", "kind": "live", "run": run, "t0": round(t_wall0, 3), "frames": frames, "label": "Walking: %s" % pol.name}
            if env.tiles_on:
                item["tiles"] = tiles_item(env, 0)
            if circuit:
                item["path"] = dirn
            elif play:
                item["drive"] = [round(x_, 2) for x_ in drive]       # (no line: driven by hand)
            else:
                item["line"] = [round(float(env.line_y[0]), 3), round(float(env.line_yaw[0]), 4)]
            ctx.telemetry(ch, item)
            frames = []
    return {"minutes": minutes}




# ── the patrol: a night of surveillance in the warehouse ─────────────────────────────────────────────────────────────
# The robot leaves its dock and walks a round through the areas asked for (A* on a 20 cm map of the floor, followed by
# pure pursuit: the speed and the turn it is told, as its walking policy learned them), stopping to look at each area's
# checkpoints. It listens (sounds reach it weaker with distance, and through racking) and looks (110° cameras, 14 m,
# line of sight). Something heard or seen sends it to look: it goes where the sound came from, finds what made it,
# looks at it and decides: a carton fallen, a door in the wind, a forklift driven by staff (a badge) are logged; a
# person without a badge raises the alarm, and it follows them, keeping its distance, until they are gone. At the end
# it goes back to its dock. What it reports: the incident report (what, where, when, what it did), the patrol log.
STATE_TEXT = {"dock": "LEAVING DOCK", "patrol": "PATROL", "scan": "CHECKPOINT", "investigate": "INVESTIGATING", "search": "SEARCHING",
              "inspect": "INSPECTING", "track": "ALARM · TRACKING", "hold": "ALARM · HOLDING", "return": "BACK TO ROUTE", "home": "RETURNING TO DOCK", "docked": "DOCKED"}
HEAR_DB = 42.0            # the quietest sound it turns to (the hall's background at night: about 35 dB)
SOUND_OF = {"carton": ("an impact", 0.91), "door": ("a metal rattle", 0.87), "forklift": ("an engine and a reversing beeper", 0.94),
            "intruder": ("a door forced, then footsteps", 0.82), "glass": ("glass breaking", 0.96)}
LOOK_AT = {"carton": 4.5, "door": 6.0, "forklift": 12.0, "intruder": 11.0, "glass": 10.0}     # how close it goes to tell what it is (m)


def pick_areas(txt):
    """the areas to patrol: all of them, some by name, or new ones (a name, then points: Cold room: 2 4.5; 10 4.5)"""
    import re
    t = str(txt or "").strip()
    if not t or t.lower() in ("all", "*", "everything", "all areas"):
        return [dict(a) for a in WAREHOUSE["areas"] if a["key"] != "stairs"], []
    out, bad = [], []
    for part in [x.strip() for x in re.split(r"[\n]+|,(?![^:]*\d\s*;)", t) if x.strip()]:
        if ":" in part:
            name, pts = part.split(":", 1)
            try:
                path = [[float(v) for v in re.split(r"[\s,]+", p_.strip())[:2]] for p_ in pts.split(";") if p_.strip()]
            except ValueError:
                bad.append(part)
                continue
            path = [[max(-5.5, min(35.5, x_)), max(-4.5, min(19.5, y_))] for x_, y_ in path if len([x_, y_]) == 2]
            if path:
                out.append({"key": "custom%d" % len(out), "name": name.strip() or "Area %d" % (len(out) + 1), "path": path, "look": [path[len(path) // 2]], "custom": True})
            continue
        low = part.lower()
        hit = [a for a in WAREHOUSE["areas"] if low == a["key"] or low in a["name"].lower()]
        if hit:
            out += [dict(a) for a in hit if a["key"] not in [o["key"] for o in out]]
        else:
            bad.append(part)
    return (out or [dict(a) for a in WAREHOUSE["areas"] if a["key"] != "stairs"]), bad


def build_route(fmap, areas):
    """the round: each area's way in turn, joined by the shortest ways between them, back to the first; with the area
    each point belongs to and the checkpoints (where it stops to look)"""
    pts, tags, looks = [], [], []
    cur = None
    for a in areas + [areas[0]]:
        way = a["path"] if a is not areas[0] or cur is None else a["path"][:1]
        for wp in way:
            leg = [tuple(wp)] if cur is None else (fmap.path(cur, wp) or [cur, tuple(wp)])
            for p_ in (leg if not pts else leg[1:]):
                pts.append((round(p_[0], 2), round(p_[1], 2)))
                tags.append(a["key"])
            cur = tuple(wp)
    for a in areas:
        for lk in a.get("look") or []:
            k = min(range(len(pts)), key=lambda i: (pts[i][0] - lk[0]) ** 2 + (pts[i][1] - lk[1]) ** 2 + (0 if tags[i] == a["key"] else 1e4))
            looks.append((k, a["key"]))
    return pts, tags, sorted(set(looks))


class KinBot:
    """the robot animated (no learned policy yet): it moves along the way it is told, its body at its standing height
    over the ground under its feet, its legs in a trot drawn from the gait clock: what the patrol will look like,
    clearly not the physics"""

    def __init__(self, H, x, y, yaw, default):
        self.H, self.x, self.y, self.yaw = H, x, y, yaw
        self.v, self.wz, self.phase = 0.0, 0.0, 0.0
        self.default = list(default)
        self.z = self.g(x, y) + H0
        self.pitch = 0.0
        self.j = list(default)
        self.contact = [1, 1, 1, 1]
        self.lp, self.ly, self.ldh = 0.0, 0.0, 0.0       # (looking: body pitch, yaw from the feet, height change)
        self.want_look = (0.0, 0.0, 0.0)

    def g(self, x, y):
        i = int(round((x - HF_X0) / HF_RES))
        j = int(round((y - HF_Y0) / HF_RES))
        return float(self.H[min(self.H.shape[0] - 1, max(0, j)), min(self.H.shape[1] - 1, max(0, i))])

    def step(self, v, wz, dt):
        self.v += max(-1.5 * dt, min(1.5 * dt, v - self.v))
        self.wz += max(-3.0 * dt, min(3.0 * dt, wz - self.wz))
        self.yaw = math.atan2(math.sin(self.yaw + self.wz * dt), math.cos(self.yaw + self.wz * dt))
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        gs = [self.g(self.x + dx * c - dy * s_, self.y + dx * s_ + dy * c) for dx, dy in ((0.33, 0.17), (0.33, -0.17), (-0.27, 0.17), (-0.27, -0.17))]
        front, back = max(gs[0], gs[1]), max(gs[2], gs[3])
        k_ = min(1.0, 4.0 * dt)
        self.lp += (self.want_look[0] - self.lp) * k_
        self.ly += (self.want_look[1] - self.ly) * k_
        self.ldh += (self.want_look[2] - self.ldh) * k_
        want_z = 0.5 * (front + back) + H0 + 0.03 * (abs(front - back) > 0.02) + self.ldh
        self.z += max(-0.8 * dt, min(0.8 * dt, want_z - self.z)) if abs(want_z - self.z) < 0.5 else (want_z - self.z)
        want_p = -math.atan2(front - back, 0.6)
        self.pitch += (want_p - self.pitch) * min(1.0, 6 * dt)
        moving = abs(self.v) > 0.03 or abs(self.wz) > 0.05
        if moving:
            self.phase = (self.phase + 2 * math.pi * 1.8 * dt) % (2 * math.pi)
        amp = min(0.3, 0.26 * abs(self.v) + 0.1 * abs(self.wz)) * (1 if self.v >= 0 else -1)
        j = list(self.default)
        for k in range(4):
            ph = self.phase + (0.0 if PAIR[k] == 0 else math.pi)
            sw = math.sin(ph)
            lift = max(0.0, sw) if moving else 0.0
            crouch = self.ldh + (-0.3 if k < 2 else 0.3) * math.sin(self.lp)          # (each leg shorter or longer for the body's attitude)
            j[3 * k + 1] = self.default[3 * k + 1] + (amp * math.cos(ph) if moving else 0.0) + 0.25 * lift - 1.1 * crouch
            j[3 * k + 2] = self.default[3 * k + 2] - 0.55 * lift + 2.2 * crouch
            self.contact[k] = 0 if (moving and sw > 0.05) else 1
        self.j = j

    def quat(self):
        yw, pt = self.yaw + self.ly, self.pitch + self.lp
        cy, sy, cp, sp = math.cos(yw / 2), math.sin(yw / 2), math.cos(pt / 2), math.sin(pt / 2)
        return [cy * cp, -sy * sp, cy * sp, sy * cp]

    def row(self, t, cmd):
        nf = max(1, sum(self.contact))
        return ([round(t, 3), round(self.x, 3), round(self.y, 3), round(self.z, 3)] + [round(v, 4) for v in self.quat()] + [round(v, 3) for v in self.j]
                + [round(cmd[0], 2), 0.0, round(cmd[1], 2), round(self.v, 2), 0.0, round(self.wz, 2)] + list(self.contact)
                + [round(c_ / nf, 2) for c_ in self.contact] + [round(self.g(self.x, self.y), 3), 0.0, 0.0, round(self.wz, 2), 0.0, 0.0, 9.8]
                + [0.0] * NU + [0.0])


def clock_of(start, s):
    h, m = [int(x) for x in (str(start or "22:00").split(":") + ["0"])[:2]]
    t = int(h * 3600 + m * 60 + s) % 86400
    return "%02d:%02d:%02d" % (t // 3600, (t // 60) % 60, t % 60)


def where_is(x, y, areas_all=None):
    """the name of the place x, y is in (for the report)"""
    W = WAREHOUSE
    if x > 27 and y > 12:
        return "the mezzanine office" if y > 15 and x > 29 else "the mezzanine"
    if y < 0.75:
        d = min(W["docks"], key=lambda d_: abs(d_["x"] - x))
        return "the dock aisle, by dock %d" % (W["docks"].index(d) + 1)
    if x > 25:
        return "the east floor"
    if y > 15.8:
        return "the staging area"
    for name, y0, y1 in (("aisle A", 3.0, 6.0), ("aisle B", 8.2, 11.2), ("aisle C", 13.4, 15.8)):
        if y0 <= y <= y1:
            return name + (" (cross aisle)" if 12.3 < x < 14.7 else "")
    for r in W["racks"]:
        if r["x0"] - 0.2 <= x <= r["x1"] + 0.2 and r["y0"] - 0.2 <= y <= r["y1"] + 0.2:
            return "rack %s" % r["name"]
    return "the west end" if x < 2 else "the cross aisle" if 12.3 < x < 14.7 else "the racking"


def patrol_run(ctx):
    import csv
    import json
    import random
    import time
    import mujoco
    import torch
    p = ctx.params
    torch.set_num_threads(1)
    R = ROBOTS[ROBOT]
    minutes = float(p.get("minutes") or 30)
    seed = int(float(p.get("seed") or 0)) or int(time.time()) % 100000
    speed = max(0.3, min(1.6, float(p.get("speed") or 1.0)))
    fast = "fast" in str(p.get("pace") or "").lower()
    demo = "demo" in str(p.get("incidents") or "Demo").lower()
    quiet = str(p.get("incidents") or "").lower().startswith("none")
    start_clock = str(p.get("start_time") or "22:00")
    ch, ch_ev, ch_ctl = p.get("robot_channel") or "robot", p.get("events_channel") or "event", p.get("control_channel") or "control"
    rng = random.Random(seed)
    # the walking policy (and the stand-up one), if any: else the patrol is animated, and says so
    pols, ups, looks = [], [], []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                h = torch.load(f.path, map_location="cpu", weights_only=False)
            except Exception as e:     # noqa: BLE001
                ctx.log("Skipped %s: %s" % (f.name, e))
                continue
            if h.get("format") != FORMAT or (h.get("robot") or "spot") != ROBOT:
                continue
            (ups if h.get("task") == "standup" else looks if h.get("task") == "look" else pols if h.get("task") == TASK else []).append(
                ((h.get("terrain") == "circuit" and float(h.get("score") or 0) >= 50, float(h.get("score") or 0)), policy_stamp(f.path, f.name), f))
    # (and those the training runs left on this machine: their policies, and the checkpoints of runs stopped early, when
    # the dataset does not have that kind)
    import glob
    import os

    class _F:
        def __init__(self, path, name):
            self.path, self.name = path, name
    jobs = os.path.dirname(os.path.dirname(os.getcwd()))
    if os.path.basename(jobs).lower() == "jobs" and not (pols and ups and looks):
        found_here = []
        cands = glob.glob(os.path.join(jobs, "*", "*", "outputs", "policy", "%s_*.pt" % ROBOT)) + \
            [x for k_ in ("policy", "standup", "look") for x in glob.glob(os.path.join(jobs, "*", "*", "%s_checkpoint.pt" % k_))]
        for pth in sorted(cands, key=os.path.getmtime, reverse=True)[:40]:
            try:
                h = torch.load(pth, map_location="cpu", weights_only=False)
            except Exception:     # noqa: BLE001
                continue
            if h.get("format") != FORMAT or (h.get("robot") or "spot") != ROBOT:
                continue
            parts_ = os.path.normpath(pth).split(os.sep)
            low_ = [x.lower() for x in parts_]
            run_ = parts_[low_.index("jobs") + 1] if "jobs" in low_ and low_.index("jobs") + 1 < len(parts_) else "?"
            f_ = _F(pth, "%s (run #%s, on this machine)" % (os.path.basename(pth), run_))
            k = h.get("task")
            tgt = ups if k == "standup" else looks if k == "look" else pols if k == TASK else None
            if tgt is None or any(not str(x[2].name).endswith("machine)") for x in tgt):
                continue                  # (the dataset has that kind: it comes first)
            tgt.append(((h.get("terrain") == "circuit" and float(h.get("score") or 0) >= 50, float(h.get("score") or 0)), os.path.getmtime(pth), f_))
            found_here.append(f_.name)
        if found_here:
            ctx.log("Policies the training runs left on this machine (not in the dataset): %s." % "; ".join(found_here[:6]))
    act = up_act = look_act = None
    if pols:
        # (the newest policy that learned the obstacle course well enough (a walk score of 50 % or more), else the newest
        # the best scoring (a flat-floor policy keeps the round to the flat floor); or the one asked by name)
        pols.sort(key=lambda x: (x[0], x[1]), reverse=True)
        pols = [(x[1], x[2]) for x in pols]
        want = str(p.get("policy_file") or "").strip()
        pol = ([x for x in pols if want and want in str(x[1].name)] or pols)[0][1]
        act, st = load_policy(pol.path)
        walk_dim = int(st.get("obs_dim") or OBS_BASE)
        ctx.log("Walking policy %s: walk score %.0f%% on %s, %.0f million steps of experience. The robot is simulated in MuJoCo: every step is the policy's, in the physics." % (
            pol.name, float(st.get("score") or 0), GROUND_NAME.get(st.get("terrain") or "flat", "?"), float(st.get("steps") or 0) / 1e6))
        if (st.get("terrain") or "flat") != "circuit":
            ctx.log("This policy has not learned on the obstacle course: the mezzanine's stairs and ramp may trip it (Learn the obstacles teaches them).")
        if ups:
            ups.sort(key=lambda x: x[1], reverse=True)
            ups = [(x[1], x[2]) for x in ups]
            up_act, _ = load_policy(ups[0][1].path)
            ctx.log("Stand-up policy %s: after a fall it gets up on its own." % ups[0][1].name)
        if looks:
            looks.sort(key=lambda x: x[1], reverse=True)
            looks = [(x[1], x[2]) for x in looks]
            look_act, _ = load_policy(looks[0][1].path)
            ctx.log("Look-around policy %s: at checkpoints and when it inspects, it aims its cameras with its legs (body up, down, left, right, lower)." % looks[0][1].name)
        else:
            ctx.log("No look-around policy wired: to look, it turns on the spot with its walking policy.")
    else:
        raise RuntimeError("The patrol runs in the physics and needs Spot's walking policy: wire the spot-policy dataset (written by "
                           "Learn to walk and Learn slopes, stairs and obstacles) to Policies. None of the files wired in is a Spot walking policy.")
    flat_policy = (st.get("terrain") or "flat") != "circuit"
    # the warehouse, its floor map, the round
    H = ground_grid("warehouse")
    fmap = FloorMap(H, HF_X0, HF_Y0, HF_RES, flat_only=flat_policy)
    areas, bad = pick_areas(p.get("areas"))
    if bad:
        ctx.log("Not understood as areas (left out): %s. Areas: %s; or a new one as Name: x y; x y …" % ("; ".join(bad), ", ".join(AREA_NAMES)))
    if flat_policy:      # (only the areas it can reach on the flat floor)
        st0 = (WAREHOUSE["start"][0], WAREHOUSE["start"][1])
        ok_, out_ = [], []
        for a in areas:
            cells = [fmap.cell(*wp) for wp in a["path"]]
            reach = all(fmap.free[c[1], c[0]] for c in cells) and fmap.path(st0, a["path"][0]) is not None and \
                all(fmap.path(a["path"][i], a["path"][i + 1]) is not None for i in range(len(a["path"]) - 1))
            (ok_ if reach else out_).append(a)
        if out_:
            ctx.log("This walking policy learned the flat floor only, so its round keeps to the flat floor and leaves out %s (stairs, ramps, "
                    "clutter): run Learn slopes, stairs and obstacles, and the patrol uses that policy and goes everywhere." % ", ".join(a["name"] for a in out_))
        areas = ok_ or areas
    loop, tags, looks = build_route(fmap, areas)
    loop_len = sum(math.hypot(loop[i + 1][0] - loop[i][0], loop[i + 1][1] - loop[i][1]) for i in range(len(loop) - 1))
    ctx.log("The round: %s; %.0f m, %d checkpoints (about %.1f minutes a lap at %.1f m/s)." % (
        " → ".join(a["name"] for a in areas), loop_len, len(looks), (loop_len / (0.8 * speed) + 4 * len(looks)) / 60, speed))
    events = [] if quiet else plan_incidents(minutes, seed, fmap, first=(35, 50) if demo else (100, 160), gap=(75, 110) if demo else (170, 290),
                                              threat_from=150 if demo else None)
    ctx.log("%d incidents will happen during the %.0f minutes (%s; seed %d): the robot does not know when or where." % (
        len(events), minutes, "demo pace: about one every 90 s" if demo else "realistic pace: about one every 4 minutes", seed))
    # the robot
    W = WAREHOUSE
    sx, sy, syaw = W["start"]
    physics = True
    if physics:
        mjm = load_model(mujoco, "warehouse")
        sim = CpuSim(mjm, 1, threads=1)
        env = WalkEnv(sim, mjm, seed=seed, vmax=1.5, push=0.0, noise=0.0, train=False, terrain="warehouse", scan=True)
        one = torch.zeros(1, dtype=torch.long)
        env.reset(one, at=[sx, sy, syaw])
        env.mode_line[:] = False
        o, _ = env.obs(noisy=False)
    else:
        mjm = mujoco.MjModel.from_xml_string(MODEL_IN_USE)
        kb = KinBot(H, sx, sy, syaw, [float(v) for v in mjm.key("knees_bent").qpos[7:]])
    info_robot = robot_info(mjm if physics else load_model(mujoco, "warehouse"), "warehouse", physics)
    info_robot["patrol"] = {"areas": [{"key": a["key"], "name": a["name"], "path": a["path"], "custom": bool(a.get("custom"))} for a in areas],
                            "route": [list(q_) for q_ in loop], "looks": [loop[k] for k, _ in looks], "animated": not physics,
                            "fov": 110, "range": 14.0, "hear_db": HEAR_DB}
    ctx.telemetry(ch, {"format": "esasim.humanoid/1", "kind": "robot", "robot": info_robot})

    push_v = 0.8
    mass = float(mjm.body_subtreemass[mjm.body("pelvis").id])

    def push_by_hand(v, dv):
        """a push on the part of the body clicked in the view ({body, at, dir}): a force for 0.1 s worth dv m/s to the whole robot"""
        if not isinstance(v, dict) or not v.get("body"):
            return None
        try:
            b = mjm.body(str(v["body"])).id
            loc = [float(x) for x in (v.get("at") or [0, 0, 0])][:3]
            dx, dy, dz = [float(x) for x in (v.get("dir") or [1, 0, 0])][:3]
        except (KeyError, TypeError, ValueError):
            return None
        n_ = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        F = mass * dv / 0.1
        sim.push_at(0, b, loc, [F * dx / n_, F * dy / n_, F * dz / n_], 0.1)
        return "A push on its %s (%.1f m/s)" % (LEG_PART.get(str(v["body"]), str(v["body"])), dv)

    def pose():
        if physics:
            q = sim.qpos[0]
            return float(q[0]), float(q[1]), float(yaw_of(q[3:7].float().unsqueeze(0))[0])
        return kb.x, kb.y, kb.yaw + kb.ly

    # the mission's state
    state, plan, pi = "dock", [], 0
    x0_, y0_, _ = pose()
    plan = fmap.path((x0_, y0_), loop[0]) or [(x0_, y0_), loop[0]]
    loop_i, lap, laps_done = 0, 1, 0
    target, timer, scan_key = None, 0.0, None
    alarm, alarm_ack_at = False, None
    actors, started = {}, set()
    current = None                              # the incident it is on
    queue = []                                  # incidents heard or seen, waiting
    checks = {a["key"]: [] for a in areas}
    looked = set()
    dist, battery = 0.0, 100.0
    t_state = {}
    sim_t, frames, last_send, last_tick, last_log = 0.0, [], 0.0, -1.0, -1.0
    t_wall0 = time.time()
    rings = []                                  # sounds for the view: [x, y, level at the source, t, heard]
    lost_t, last_seen = 0.0, None
    cmd_v, cmd_w = 0.0, 0.0
    fell_at, getting_up, up_n = None, False, 0
    look, in_look, look_yaw0 = [0.0, 0.0, 0.0], False, 0.0      # (where it aims its cameras: body pitch, yaw from its feet, height change)
    can_look = (look_act is not None) or not physics
    log_rows = open(ctx.output("report", "patrol_log.csv"), "w", newline="", encoding="utf-8")
    lw = csv.writer(log_rows)
    lw.writerow(["t_s", "clock", "x", "y", "yaw_deg", "state", "area", "speed_mps", "battery_pct", "alarm", "incident"])
    end_t = minutes * 60.0
    going_home = False
    feed = []

    def say(text, level="info", **kw):
        d_ = {"text": text, "level": level, "clock": clock_of(start_clock, sim_t)}
        d_.update(kw)
        ctx.telemetry(ch_ev, d_)
        feed.append([round(sim_t, 1), d_["clock"], text, level])
        del feed[:-8]
        ctx.log("[%s] %s" % (clock_of(start_clock, sim_t), text))

    def go(to, new_state):
        nonlocal plan, pi, state, timer
        x, y, _ = pose()
        plan = fmap.path((x, y), to) or [(x, y), tuple(to)]
        pi, state, timer = 0, new_state, 0.0

    def back_to_route():
        """the nearest point of the round ahead of where it left it"""
        nonlocal loop_i
        x, y, _ = pose()
        ahead = range(loop_i, len(loop)) if loop_i < len(loop) - 1 else range(len(loop))      # (where it left off: no area skipped)
        best = min(ahead, key=lambda i: math.hypot(loop[i][0] - x, loop[i][1] - y) + 0.15 * (i - loop_i))
        loop_i = best
        go(loop[best], "return")

    def area_name(key):
        return next((a["name"] for a in areas if a["key"] == key), key)

    def kind_of(ev):
        return ev["kind"]

    def start_incident(ev):
        nonlocal current, target
        current = ev
        ev["state"] = "investigating"
        ev["t_start"] = sim_t
        est = ev["heard_at"] if ev.get("heard_at") else (ev["x"], ev["y"])
        target = est
        go(est, "investigate")
        ev["log"].append((sim_t, "Went to look (%.0f m away)" % math.hypot(est[0] - pose()[0], est[1] - pose()[1])))

    def finish_incident(verdict, threat, action):
        nonlocal current, target
        ev = current
        ev["state"] = "closed"
        ev["verdict"], ev["threat_found"], ev["action"] = verdict, threat, action
        ev["t_closed"] = sim_t
        ev["log"].append((sim_t, verdict + ". " + action))
        current, target = None, None

    say("%s leaves its dock: %s patrol of %s (%.0f minutes)%s" % (R["short"], "night" if int(start_clock.split(":")[0]) >= 19 or int(start_clock.split(":")[0]) < 6 else "a",
                                                              ", ".join(a["name"] for a in areas), minutes, "" if physics else " · ANIMATED (no walking policy wired)"), "info")
    while True:
        # the controls: Acknowledge (the operator), Speed, End patrol
        for msg in ctx.messages(ch_ctl):
            d = msg.get("data") if isinstance(msg, dict) else None
            if not isinstance(d, dict):
                continue
            c, v = str(d.get("control", "")).strip().lower(), d.get("value")
            if c == "speed":
                try:
                    speed = max(0.3, min(1.6, float(v)))
                except (TypeError, ValueError):
                    pass
            elif c.startswith("acknowledge") or c.startswith("ack"):
                if alarm:
                    alarm, alarm_ack_at = False, sim_t
                    say("Alarm acknowledged by the operator", "ok")
            elif c in ("push strength", "push_strength", "push speed"):
                try:
                    push_v = max(0.1, min(3.0, float(v)))
                except (TypeError, ValueError):
                    pass
            elif c == "push":
                txt = push_by_hand(v, push_v)
                if not txt:              # (a plain Push: a shove of the whole body, from a random side)
                    ang = env.U(1, 0.0, 2 * math.pi)
                    sim.kick(torch.stack([torch.cos(ang), torch.sin(ang)], -1) * push_v)
                    txt = "A push (%.1f m/s)" % push_v
                say(txt, "info")
            elif c in ("end patrol", "end", "go home", "dock"):
                end_t = min(end_t, sim_t)
            elif c in ("intruder", "intruder drill", "drill", "simulate intruder", "test intruder"):        # (a drill: an intruder now, at a dock)
                d0 = W["docks"][rng.randrange(len(W["docks"]))]
                hide = rng.choice([[13.5, 4.5], [7.0, 9.7], [18.0, 14.9], [13.5, 9.7], [4.0, 17.2]])
                way = fmap.path((d0["x"], -4.3), hide) or [(d0["x"], -4.3), hide]
                ev = {"id": 100 + len(events), "kind": "intruder", "t0": round(sim_t + 1, 1), "way": [list(q_) for q_ in way], "exit": [d0["x"], -4.3],
                      "x": d0["x"], "y": -4.3, "speed": 1.1, "drill": True, **KINDS["intruder"]}
                events.append(ev)
                say("Drill: an intruder will come in by dock %d" % (W["docks"].index(d0) + 1), "info")
        if not fast:
            t_due = t_wall0 + sim_t
            now = time.time()
            if now < t_due:
                time.sleep(min(0.01, t_due - now))
                continue
        x, y, yaw = pose()
        # ── the mission, ten times a second ──
        if sim_t - last_tick >= 0.1 - 1e-9:
            last_tick = sim_t
            # the incidents that start now
            for ev in events:
                if ev["id"] in started or sim_t < ev["t0"]:
                    continue
                started.add(ev["id"])
                ev["log"], ev["state"], ev["clock"] = [], "unnoticed", clock_of(start_clock, ev["t0"])
                actors[ev["id"]] = Actor(ev)
            # what it hears and sees
            emit = []
            for ev in events:
                if ev["id"] not in started:
                    continue
                a_ = actors[ev["id"]]
                age = sim_t - ev["t0"]
                k = ev["kind"]
                if k == "carton" and age < 0.4:
                    emit.append((ev, ev["x"], ev["y"], ev["db"]))
                elif k == "door" and age < ev["dur"] and (age % 1.3) < 0.35:
                    emit.append((ev, ev["x"], ev["y"], ev["db"]))
                elif k == "forklift" and a_.moving and not a_.gone and age < ev["dur"]:
                    emit.append((ev, a_.x, a_.y, ev["db"]))
                elif k == "intruder" and age < 1.5:
                    emit.append((ev, ev["exit"][0], ev["exit"][1], 76.0))
                elif k == "glass" and age < 0.6:
                    emit.append((ev, ev["x"], ev["y"], ev["db"]))
                elif k in ("intruder", "glass") and a_.moving and not a_.gone and (age % 0.6) < 0.12:
                    emit.append((ev, a_.x, a_.y, 50.0 if a_.fleeing else 46.0))
            for ev, ex, ey, db in emit:
                lvl = heard((ex, ey), db, (x, y), fmap)
                rings.append([round(ex, 2), round(ey, 2), round(db), round(sim_t, 2), lvl >= HEAR_DB])
                if lvl >= HEAR_DB and ev["state"] == "unnoticed":
                    d_ = math.hypot(ex - x, ey - y)
                    err = 0.12 * d_ + 0.3
                    hx, hy = ex + rng.gauss(0, err), ey + rng.gauss(0, err)
                    c_ = fmap.xy(fmap.nearest_free(fmap.cell(hx, hy)))
                    ev["heard_at"] = c_
                    ev["state"] = "noticed"
                    ev["noticed_t"] = sim_t
                    bearing = (90 - math.degrees(math.atan2(ey - y, ex - x))) % 360
                    what, conf = SOUND_OF[ev["kind"]]
                    if ev["kind"] == "intruder":
                        what, conf = "a door forced", 0.78
                    ev["log"].append((sim_t, "Heard %s (%.0f dB at the robot, %.0f m away, sound classifier %.2f)" % (what, lvl, d_, conf)))
                    queue.append(ev)
                    say("Heard %s · %.0f dB · bearing %03.0f° · about %.0f m" % (what, lvl, bearing, d_), "warn" if KINDS[ev["kind"]]["threat"] else "info",
                        sound={"x": round(ex, 1), "y": round(ey, 1), "est": [round(c_[0], 1), round(c_[1], 1)], "db": round(lvl)})
            rings[:] = [r_ for r_ in rings if sim_t - r_[3] < 1.6][-24:]
            seen_now = None
            for ev in events:
                if ev["id"] not in started:
                    continue
                a_ = actors[ev["id"]]
                if a_.gone or ev["kind"] == "door":
                    continue
                rng_ = {"carton": 6.0, "forklift": 14.0}.get(ev["kind"], 14.0)
                if seen((a_.x, a_.y), (x, y), yaw, fmap, rng=rng_):
                    if ev["state"] == "unnoticed":
                        ev["state"] = "noticed"
                        ev["noticed_t"] = sim_t
                        ev["heard_at"] = (a_.x, a_.y)
                        what = {"carton": "something on the floor", "forklift": "a vehicle moving"}.get(ev["kind"], "a person")
                        ev["log"].append((sim_t, "Saw %s at %.0f m (camera)" % (what, math.hypot(a_.x - x, a_.y - y))))
                        queue.append(ev)
                        say("Saw %s · %.0f m" % (what, math.hypot(a_.x - x, a_.y - y)), "warn" if KINDS[ev["kind"]]["threat"] else "info")
                    if current is ev or seen_now is None:
                        seen_now = ev
            for a_ in actors.values():
                a_.step(0.1, (x, y), fmap)
            # what next: the most urgent incident first (a person over a carton)
            if queue and state in ("patrol", "scan", "return", "dock", "investigate", "search", "inspect") and not going_home:
                queue.sort(key=lambda e_: (not KINDS[e_["kind"]]["threat"], e_.get("noticed_t", 0)))
                nxt = queue[0]
                if current is None or (KINDS[nxt["kind"]]["threat"] and not KINDS[current["kind"]]["threat"]):
                    queue.pop(0)
                    if current is not None:
                        current["state"] = "noticed"
                        current["log"].append((sim_t, "Left for something more urgent"))
                        queue.append(current)
                        current = None
                    start_incident(nxt)
                    say("Investigating: %s" % where_is(*target), "warn" if KINDS[nxt["kind"]]["threat"] else "info", target=[round(target[0], 1), round(target[1], 1)])
            # the incident it is on
            if current is not None:
                ev = current
                a_ = actors[ev["id"]]
                d_t = math.hypot(a_.x - x, a_.y - y)
                vis = seen_now is ev
                if state in ("investigate", "search"):
                    if vis and (d_t <= LOOK_AT[ev["kind"]] or ev["kind"] in ("intruder", "glass")) and (ev["kind"] != "door"):
                        state, timer = "inspect", 0.0
                        ev["log"].append((sim_t, "In view at %.1f m: looking" % d_t))
                        say("Inspecting · %s" % {"carton": "object on the floor", "forklift": "vehicle", "intruder": "person", "glass": "person"}[ev["kind"]], "warn" if KINDS[ev["kind"]]["threat"] else "info")
                    elif ev["kind"] == "door" and math.hypot(ev["x"] - x, ev["y"] - y) < LOOK_AT["door"] and fmap.visible((x, y), (ev["x"], ev["y"])):
                        state, timer = "inspect", 0.0
                        say("Inspecting · dock door %d" % (min(range(4), key=lambda i: abs(W["docks"][i]["x"] - ev["x"])) + 1))
                    elif vis and ev["kind"] in ("intruder", "glass", "forklift") and int(sim_t * 10) % 10 == 0:
                        target = (a_.x, a_.y)
                        go(target, state)
                elif state == "inspect":
                    timer += 0.1
                    if timer >= (1.2 if ev["kind"] in ("intruder", "glass", "forklift") else 2.5):
                        k = ev["kind"]
                        place = where_is(ev["x"], ev["y"])
                        if k == "carton":
                            finish_incident("A carton fallen from rack %s in %s; nobody around" % (min(W["racks"], key=lambda r_: abs((r_["y0"] + r_["y1"]) / 2 - ev["y"]) + (0 if r_["x0"] <= ev["x"] <= r_["x1"] else 9))["name"], place),
                                            False, "Logged, photo attached; a ticket for the morning shift")
                            say("Carton on the floor · no threat · logged", "ok")
                            back_to_route()
                        elif k == "door":
                            finish_incident("Dock door %d rattling in the wind; closed and locked" % (min(range(4), key=lambda i: abs(W["docks"][i]["x"] - ev["x"])) + 1), False, "Logged")
                            say("Dock door in the wind · locked · logged", "ok")
                            back_to_route()
                        elif k == "forklift":
                            finish_incident("A forklift driven by staff in %s; badge recognised" % place, False, "Logged (authorised)")
                            say("Forklift · operator badge recognised · authorised", "ok")
                            back_to_route()
                        else:
                            alarm = True
                            ev["threat_found"] = True
                            ev["alarm_t"] = sim_t
                            ev["log"].append((sim_t, "ALARM: a person without a badge in %s; operator alerted with snapshot, siren on, recording" % where_is(a_.x, a_.y)))
                            say("ALARM · person without a badge in %s · operator alerted · siren on" % where_is(a_.x, a_.y), "alarm", alarm=True)
                            state, timer, last_seen, lost_t = "track", 0.0, (a_.x, a_.y), 0.0
                elif state in ("track", "hold"):
                    if vis:
                        last_seen, lost_t = (a_.x, a_.y), 0.0
                    else:
                        lost_t += 0.1
                    if a_.gone:
                        exit_ = where_is(*ev["exit"])
                        finish_incident("A person without a badge (%s); followed until they left by %s" % ("broke the office window" if ev["kind"] == "glass" else "came in by the dock door", exit_ if ev["kind"] != "glass" else "the broken window"),
                                        True, "Alarm raised at %s; tracked; evidence clip saved; police report recommended" % clock_of(start_clock, ev.get("alarm_t", sim_t)))
                        say("The intruder left (%s) · alarm stays on until the operator acknowledges" % ("by the window" if ev["kind"] == "glass" else exit_), "alarm")
                        alarm_ack_at = sim_t + 25.0
                        back_to_route()
                    elif lost_t > 25.0:
                        finish_incident("A person without a badge; lost from view near %s" % where_is(*last_seen), True, "Alarm raised; last position sent to the operator")
                        say("Lost sight of the intruder near %s · last position sent" % where_is(*last_seen), "alarm")
                        alarm_ack_at = sim_t + 25.0
                        back_to_route()
                    else:
                        d_l = math.hypot(last_seen[0] - x, last_seen[1] - y)
                        if d_l > 3.5 and int(sim_t * 10) % 5 == 0:
                            target = last_seen
                            go(last_seen, "track")
                        elif d_l <= 3.5 and state == "track":
                            state = "hold"
                        elif d_l > 4.5 and state == "hold":
                            state = "track"
            if alarm and alarm_ack_at is not None and sim_t >= alarm_ack_at:
                alarm = False
                say("Alarm acknowledged by the operator (auto, demo)", "ok")
                alarm_ack_at = None
            # the session's end: back to the dock
            if sim_t >= end_t and not going_home and state not in ("track", "hold", "inspect"):
                going_home = True
                current = None
                queue.clear()
                go((sx + 0.6, sy), "home")
                say("End of the patrol: back to the dock", "info")
            # the log, once a second
            if sim_t - last_log >= 1.0 - 1e-9:
                last_log = sim_t
                lw.writerow([round(sim_t, 1), clock_of(start_clock, sim_t), round(x, 2), round(y, 2), round(math.degrees(yaw)), state, where_is(x, y),
                             round(cmd_v, 2), round(battery, 1), int(alarm), current["id"] if current else ""])
                if int(sim_t) % 2 == 0:       # (the numbers for the charts and gauges: every 2 s)
                    ctx.telemetry("patrol", {"minutes": round(sim_t / 60.0, 2), "clock": clock_of(start_clock, sim_t), "battery": round(battery, 1), "distance": round(dist),
                                             "incidents": sum(1 for e_ in events if e_.get("noticed_t") is not None), "alarms": sum(1 for e_ in events if e_.get("alarm_t") is not None),
                                             "resolved": sum(1 for e_ in events if e_.get("state") == "closed"), "speed": round(cmd_v, 2), "alarm": int(alarm), "laps": laps_done, "state": STATE_TEXT.get(state, state)})
        # ── how it moves: along its plan (pure pursuit), turned to look, or still ──
        v, w = 0.0, 0.0
        if state in ("dock", "patrol", "return", "investigate", "track", "home"):
            v, w, pi, d_end = follow(plan, pi, x, y, yaw, speed if state != "track" else min(1.6, speed * 1.3))
            if pi >= len(plan) - 1 and d_end < (0.45 if state in ("dock", "home") else 0.8):
                if state == "dock":
                    state, plan, pi, loop_i = "patrol", loop, 0, 0
                    say("On its round: %s" % area_name(tags[0]))
                elif state == "return":
                    plan = loop[loop_i:] if loop_i < len(loop) - 1 else loop
                    pi, state = 0, "patrol"
                    say("Back on its round")
                elif state == "patrol":
                    lap += 1
                    laps_done += 1
                    plan, pi, loop_i = loop, 0, 0
                    looked.clear()
                    say("Lap %d done · on to lap %d" % (laps_done, lap))
                elif state == "investigate":
                    state, timer = "search", 0.0
                    say("At the spot: looking round")
                elif state == "track":
                    state = "hold"
                elif state == "home":
                    state = "docked"
                    say("Docked · charging · patrol over", "ok")
            # (where on the round it is, and the checkpoints)
            if state == "patrol":
                base = len(loop) - len(plan)
                loop_i = max(loop_i, base + pi)
                for k, key in looks:
                    if (k, lap) not in looked and abs(k - loop_i) <= 3 and math.hypot(loop[k][0] - x, loop[k][1] - y) < 1.0:
                        looked.add((k, lap))
                        state, timer, scan_key = "scan", 0.0, key
                        checks[key].append(sim_t)
                        say("Checkpoint · %s: %s" % (area_name(key), ("looking along the racks, up to the top levels" if key.startswith("aisle") or key == "staging" else "looking round, up and down") if can_look else "looking round"))
                        break
        want_look = None
        if state == "scan":
            timer += CTRL_DT
            if can_look:          # (it looks along the racks, left and right, up to the top levels and down to the floor)
                want_look = (-0.38 if (timer % 2.5) < 1.4 else 0.15, 0.45 * math.sin(timer * 2 * math.pi / 5.0), -0.04)
            else:
                w = 0.9 * math.sin(timer * 2 * math.pi / 3.0)
            if timer >= (5.0 if can_look else 3.0):
                state = "patrol"
                say("%s: all clear" % area_name(scan_key), "ok")
        elif state == "search":
            timer += CTRL_DT
            w = 0.9
            if timer > 7.0 and current is not None:
                ev = current
                if KINDS[ev["kind"]]["threat"]:
                    # (a door forced, glass broken, and nobody in sight: the operator is told, and it keeps watching for them)
                    ev["log"].append((sim_t, "Nobody in sight at %s: operator alerted; it keeps watching" % where_is(x, y)))
                    ev["state"], ev["heard_at"] = "unnoticed", None
                    current, target = None, None
                    say("%s at %s · nobody in sight · operator alerted, watching" % ("Door forced" if ev["kind"] == "intruder" else "Glass broken", where_is(ev["x"], ev["y"]) if ev["kind"] == "glass" else where_is(*ev["exit"])), "warn")
                else:
                    finish_incident("Nothing found where %s came from (%s)" % (SOUND_OF[ev["kind"]][0], where_is(x, y)), False, "Logged as unresolved; the area gets an extra pass")
                    say("Nothing found · logged as unresolved", "info")
                back_to_route()
            elif timer > 7.0:
                back_to_route()
        elif state in ("inspect", "hold") and current is not None:
            a_ = actors[current["id"]]
            tx, ty = (a_.x, a_.y) if current["kind"] != "door" else (current["x"], current["y"])
            bearing = math.atan2(ty - y, tx - x)
            err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
            v = 0.0
            base = look_yaw0 if in_look else yaw
            rel = math.atan2(math.sin(bearing - base), math.cos(bearing - base))
            if can_look and abs(rel) < 0.48:      # (near enough: it aims with its body; else it turns to face first)
                th = {"carton": 0.15, "forklift": 1.3, "door": 1.6}.get(current["kind"], 1.55)
                d_ = max(0.5, math.hypot(tx - x, ty - y))
                want_look = (max(-0.45, min(0.45, -math.atan2(th - 0.5, d_))), rel, -0.12 if current["kind"] == "carton" else 0.0)
                w = 0.0
            else:
                w = max(-0.9, min(0.9, 2.0 * err))
        elif state == "docked":
            v, w = 0.0, 0.0
        # (slower on stairs and ramps)
        if v > 0.5:
            c_, s_ = math.cos(yaw), math.sin(yaw)
            g0 = H[min(H.shape[0] - 1, max(0, int(round((y - HF_Y0) / HF_RES)))), min(H.shape[1] - 1, max(0, int(round((x - HF_X0) / HF_RES))))]
            g1 = H[min(H.shape[0] - 1, max(0, int(round((y + 0.8 * s_ - HF_Y0) / HF_RES)))), min(H.shape[1] - 1, max(0, int(round((x + 0.8 * c_ - HF_X0) / HF_RES))))]
            if abs(float(g1) - float(g0)) > 0.04:
                v = min(v, 0.5)
        cmd_v += max(-2.0 * CTRL_DT, min(2.0 * CTRL_DT, v - cmd_v))
        cmd_w += max(-4.0 * CTRL_DT, min(4.0 * CTRL_DT, w - cmd_w))
        # one control step: 20 ms
        if physics:
            looking = want_look is not None and look_act is not None and abs(cmd_v) < 0.05 and not getting_up
            if looking and not in_look:
                in_look, look_yaw0 = True, yaw
                env.phase_dt[:] = 0.0
                env.phase[:] = torch.tensor([0.0, math.pi])
            elif not looking and in_look:
                in_look = False
                look = [0.0, 0.0, 0.0]
                env.phase_dt[:] = 2 * math.pi * CTRL_DT * 1.8
            if in_look:
                for i_ in range(3):
                    look[i_] += (want_look[i_] - look[i_]) * min(1.0, 3.0 * CTRL_DT)
                env.cmd[0, 0], env.cmd[0, 1], env.cmd[0, 2] = look[0], look[1], look[2]
                env.speed[:] = 0.0
                a = look_act(o[:, :walk_dim]) * 1.5
            elif getting_up:
                env.cmd[:] = 0.0
                env.phase_dt[:] = 0.0
                a = up_act(o)
            else:
                env.cmd[0, 0], env.cmd[0, 1], env.cmd[0, 2] = cmd_v, 0.0, cmd_w
                env.speed[:] = 0.0
                a = act(o[:, :walk_dim])
            _, _, info = env.step(a)
            o, _ = env.obs(noisy=False)
            if int(env.steps[0]) >= EP_STEPS - 1:
                env.steps[:] = 0
            if bool(info["fell"][0]) and fell_at is None:
                fell_at = sim_t
                say("It fell in %s%s" % (where_is(x, y), ": getting up" if up_act is not None else ": put back on its feet in 3 s"), "warn")
                getting_up = up_act is not None
                up_n = 0
            if getting_up:
                up_n = up_n + 1 if (float(env.height()[0]) > 0.32 and float(env.S("upvector")[0, 2]) > 0.85) else 0
                if up_n >= 25:
                    getting_up, fell_at = False, None
                    env.phase_dt[:] = 2 * math.pi * CTRL_DT * 1.8
                    say("Up on its own · carrying on", "ok")
            if fell_at is not None and sim_t - fell_at > (10.0 if getting_up else 3.0):
                getting_up, fell_at = False, None
                env.reset(one, at=[x, y, yaw])
                env.mode_line[:] = False
                env.phase_dt[:] = 2 * math.pi * CTRL_DT * 1.8
                o, _ = env.obs(noisy=False)
            if int(env.steps[0]) % 2 == 0:
                frames.append(frame_row(env, 0, sim_t))
        else:
            if want_look is not None and not in_look:
                in_look, look_yaw0 = True, kb.yaw
            elif want_look is None and in_look:
                in_look = False
            kb.want_look = want_look if want_look is not None else (0.0, 0.0, 0.0)
            kb.step(cmd_v, cmd_w, CTRL_DT)
            if int(round(sim_t / CTRL_DT)) % 2 == 0:
                frames.append(kb.row(sim_t, (cmd_v, cmd_w)))
        x2, y2, _ = pose()
        dist += math.hypot(x2 - x, y2 - y)
        battery -= CTRL_DT * (0.019 if abs(cmd_v) > 0.05 or abs(cmd_w) > 0.1 else 0.008)
        t_state[state] = t_state.get(state, 0.0) + CTRL_DT
        sim_t += CTRL_DT
        # the view: frames and the mission, four times a second
        if (sim_t - last_send >= 0.25 or state == "docked") and frames:
            last_send = sim_t
            item = {"format": "esasim.humanoid/1", "kind": "live", "run": 1, "t0": round(t_wall0, 3), "frames": frames,
                    "label": "Patrol%s" % (" (animated: no walking policy)" if not physics else ""),
                    "mission": {"state": state, "text": STATE_TEXT.get(state, state.upper()), "t": round(sim_t, 1), "clock": clock_of(start_clock, sim_t),
                                "left": round(max(0.0, end_t - sim_t)), "battery": round(battery, 1), "lap": lap, "dist": round(dist), "alarm": alarm,
                                "area": where_is(x2, y2), "plan": [[round(a_[0], 2), round(a_[1], 2)] for a_ in plan[pi:pi + 120]],
                                "target": [round(target[0], 2), round(target[1], 2)] if target else None,
                                "actors": [a_.item() for a_ in actors.values() if not a_.gone and (a_.kind != "door" or sim_t - a_.ev["t0"] < a_.ev["dur"])],
                                "rings": rings[-12:], "seen": None if current is None else current["id"],
                                "counts": {"incidents": sum(1 for e_ in events if e_["id"] in started), "noticed": sum(1 for e_ in events if e_.get("noticed_t") is not None),
                                           "alarms": sum(1 for e_ in events if e_.get("alarm_t") is not None), "closed": sum(1 for e_ in events if e_.get("state") == "closed")},
                                "feed": feed[-6:], "animated": not physics,
                                "aim": ([round(-math.degrees(look[0] if physics else kb.lp)), round(math.degrees(look[1] if physics else kb.ly)), round(100 * (look[2] if physics else kb.ldh))] if in_look else None)}}
            ctx.telemetry(ch, item)
            frames = []
            ctx.progress(min(99, 100 * sim_t / max(1.0, end_t)), "%s · %s · %.0f m · %d incidents" % (STATE_TEXT.get(state, state), clock_of(start_clock, sim_t), dist, item["mission"]["counts"]["incidents"]))
        if state == "docked" or sim_t > end_t + 240:
            break
    log_rows.close()
    # ── the report ──
    rows = []
    for ev in events:
        if ev["id"] not in started:
            continue
        rows.append({"id": ev["id"], "kind": ev["kind"], "what_it_was": ev["what"], "clock": ev["clock"], "t_s": ev["t0"], "where": where_is(ev["x"], ev["y"]),
                     "noticed": ev.get("noticed_t") is not None, "noticed_after_s": round(ev["noticed_t"] - ev["t0"], 1) if ev.get("noticed_t") is not None else None,
                     "verdict": ev.get("verdict") or ("not noticed" if ev.get("noticed_t") is None else "not closed when the patrol ended"),
                     "threat": KINDS[ev["kind"]]["threat"], "alarm": ev.get("alarm_t") is not None,
                     "alarm_after_s": round(ev["alarm_t"] - ev["t0"], 1) if ev.get("alarm_t") is not None else None,
                     "action": ev.get("action") or "", "drill": bool(ev.get("drill")),
                     "log": [{"clock": clock_of(start_clock, t_), "text": tx_} for t_, tx_ in ev["log"]]})
    threats = [r_ for r_ in rows if r_["threat"]]
    caught = [r_ for r_ in threats if r_["alarm"]]
    false_alarms = [r_ for r_ in rows if r_["alarm"] and not r_["threat"]]
    summary = {"robot": R["name"], "mode": "physics, learned policy" if physics else "animated (no walking policy)", "minutes": minutes, "seed": seed,
               "start": start_clock, "end": clock_of(start_clock, sim_t), "areas": [a["name"] for a in areas], "laps": laps_done,
               "distance_m": round(dist), "battery_left_pct": round(battery, 1), "incidents": len(rows), "noticed": sum(1 for r_ in rows if r_["noticed"]),
               "threats": len(threats), "threats_alarmed": len(caught), "false_alarms": len(false_alarms),
               "mean_alarm_after_s": round(sum(r_["alarm_after_s"] for r_ in caught) / len(caught), 1) if caught else None,
               "checkpoints": {area_name(k): len(v) for k, v in checks.items()},
               "time_in_state_s": {STATE_TEXT.get(k, k): round(v) for k, v in sorted(t_state.items(), key=lambda kv: -kv[1])}}
    with open(ctx.output("report", "incidents.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "incidents": rows}, f, indent=1)
    md = ["# Patrol report: %s" % W.get("name", "Warehouse"), "",
          "%s, %s patrol from %s to %s (%s). Seed %d." % (R["name"], "%.0f-minute" % minutes, start_clock, summary["end"], summary["mode"], seed), "",
          "| | |", "|---|---|",
          "| Areas | %s |" % ", ".join(summary["areas"]), "| Laps | %d (%.0f m walked) |" % (laps_done, dist),
          "| Battery left | %.0f %% |" % battery, "| Incidents | %d (%d noticed) |" % (len(rows), summary["noticed"]),
          "| Intrusions | %d, alarm raised for %d%s |" % (len(threats), len(caught), (", on average %.0f s after they began" % summary["mean_alarm_after_s"]) if caught else ""),
          "| False alarms | %d |" % len(false_alarms), "",
          "## Incidents", ""]
    for r_ in rows:
        md.append("### %s · %s · %s%s" % (r_["clock"], KINDS[r_["kind"]]["label"], r_["where"], " (drill)" if r_["drill"] else ""))
        md.append("")
        md.append("**%s.** %s" % (r_["verdict"], r_["action"]))
        md.append("")
        for l_ in r_["log"]:
            md.append("- %s %s" % (l_["clock"], l_["text"]))
        md.append("")
    md += ["## Checkpoints", ""] + ["- %s: %d" % (k, v) for k, v in summary["checkpoints"].items()] + ["",
           "## Time", ""] + ["- %s: %d s" % (k, v) for k, v in summary["time_in_state_s"].items()] + ["",
           "_Simulated in EsaSim: the warehouse, the incidents and the sensors are models (hearing: sound level falling with "
           "distance and through racking; sight: field of view, range, line of sight). %s_" % (
               "The robot walked with its learned policy, in MuJoCo." if physics else "The robot was animated: no walking policy was wired.")]
    with open(ctx.output("report", "incident_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    ctx.log("Patrol over: %d incidents, %d intrusions (alarm for %d), %d false alarms, %.0f m, %d laps. Report: incident_report.md, incidents.json, patrol_log.csv." % (
        len(rows), len(threats), len(caught), len(false_alarms), dist, laps_done))
    return summary


def run(ctx):
    import mujoco
    what = str(ctx.params.get("mode") or "Learn to walk").lower()
    use_robot(robot_id(ctx.params), mujoco)
    if "patrol" in what or "surveil" in what or "warehouse" in what:
        return patrol_run(ctx)
    if what.startswith("walk"):
        return walk_run(ctx)
    if "look" in what:
        return look_run(ctx)
    if "stand" in what:
        return stand_run(ctx)
    return learn_run(ctx)
