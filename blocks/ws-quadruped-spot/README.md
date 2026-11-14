# Quadruped · Spot

Boston Dynamics Spot in MuJoCo: learns to stand, walk and climb (PPO, GPU or CPU), then patrols a warehouse, hearing and seeing incidents and raising alarms

Boston Dynamics Spot (MuJoCo Menagerie, BSD-3: 12 joints, joint limits; masses scaled to Spot's real 32.5 kg, hip and knee torques from Boston Dynamics' joint data) in MuJoCo, the physics engine robotics research uses. It learns by trial and error (PPO), many robots at once on an NVIDIA GPU (MuJoCo Warp) or on the CPU, then patrols a warehouse.

What to do:
- Learn to stand: it starts lying where a fall left it (on its side, on its back) and learns to roll over, get up and stay up.
- Learn to look around: its feet planted, it aims its front cameras with its legs: body up, down, left, right, lower.
- Learn to walk: a trot at the speed and turn asked. On the Obstacle course (slopes, stairs up and down, curbs, rough ground, moving tiles, ice) it also gets an elevation map of the ground ahead from its cameras and learns to use it.
- Walk: the newest policy drives it live in the 3D view: speed, way round, push it, reset.
- Patrol the warehouse: a session of surveillance (30 minutes by default) in a 42 x 25 m warehouse: four loading docks, three aisles of pallet racking, staging, an east floor, a mezzanine office reached by 7 steps and a 15 % ramp, a conveyor crossed by a step-over bridge, a dispatch platform with steps and a 12.5 % ramp, aisles cluttered with pallets and cartons to step over. It walks a round through the areas asked for (A* on a map of the floor, followed with its walking policy), stops at checkpoints to look (with the look-around policy: up to the top racks, down to the floor), hears (sound falling with distance and through racking) and sees (110° cameras, 14 m, line of sight). A sound or a movement sends it to look: a carton fallen, a door in the wind, a forklift with a badged driver are logged; a person without a badge raises the alarm (siren, operator alerted) and it follows them, keeping its distance, until they leave. At the end it goes back to its dock. It writes incident_report.md, incidents.json and patrol_log.csv.

Areas: all, or names (Loading docks, Aisle A, Aisle B, Aisle C, Staging area, East floor, Dispatch platform, Mezzanine office), or your own: Name: x y; x y (metres; x 0 at dock 1, along the south wall; y 0 at the dock aisle, toward the north wall).

The patrol is always in the physics: every step is the walking policy's, in MuJoCo, and you can push the robot (Push, then click on it). It uses the policies wired to Policies, or else those the training runs left on the machine; with a policy that learned the flat floor only, its round keeps to the flat floor. Good walking needs GPU training: about 1 to 2 hours per stage on one NVIDIA GPU (4096 robots); a few minutes on the CPU only shows it learning.

Controls in the 3D view: Patrol: Acknowledge alarm, Intruder drill, End patrol, Push, Push strength, Speed. Walk: Speed, Push, Reset.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-quadruped-spot` · version 9 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policies | Files (pt) | Spot's policies (.pt): walking, stand-up, look-around. Learn: carries on from the newest. Walk, Patrol: the newest of each Several connections. Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | spot_walk_[course_]…, spot_standup_… or spot_look_<date>_<NN>.pt: the best policy (network, scaling, optimizer, score) |
| `curve` Learning curve | Table (csv) | training.csv (Learn to walk), standup.csv (Learn to stand) or look.csv (Learn to look around) |
| `report` Patrol report | Files (md, json, csv) | Patrol: incident_report.md, incidents.json, patrol_log.csv (once a second: where, what it was doing, battery) |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to walk"` | Learn to stand: get up. Learn to look around: aim its cameras with its legs. Learn to walk: trot. Walk: live. Patrol the warehouse: the surveillance session |
| `robot` Robot | select | `"Boston Dynamics Spot"` | Spot: 1.1 m long, 32.5 kg, 12 joints (MuJoCo Menagerie; masses and torque limits set to Boston Dynamics' figures) |
| `accelerator` Run on | select | `"auto"` | auto: an NVIDIA GPU machine (MuJoCo Warp; PyTorch with CUDA installed on its own if missing). CPU: any machine, fewer robots |
| `terrain` Ground | select | `"Flat floor"` | Learn to walk, Walk. Obstacle course: slopes, stairs up and down, curbs, rough ground, moving tiles, ice |
| `robots` Robots learning at once | number | `4096` | Learn: at most 512 on the CPU |
| `minutes` Minutes | number | `90` | How long it learns, walks or patrols (Patrol: 30 is a night round) |
| `max_speed` Top speed asked (m/s) | number | `1.2` |  |
| `push` Pushes up to (m/s) | number | `1` | Learn to walk: a shove every 5 to 10 s; 0: none |
| `replay_every` Replay every (s) | number | `20` | Learn: how often an episode is replayed in the 3D view |
| `seed` Seed | number | `1` | Patrol: which night (where and when incidents happen); 0: a new one each run |
| `training` Training | select | `"Continue"` | Continue: from the newest policy wired in (or a stopped run's checkpoint on this machine). Start over: from nothing |
| `speed` Walking speed (m/s) | number | `0.8` | Walk, Patrol: the speed it walks at (the view's slider changes it) |
| `heading` Heading (degrees) | number | `0` | Walk: on the obstacle course beyond ±90° it goes round clockwise; on the flat floor, the way it walks |
| `policy_file` Walking policy | text | `""` | Walk, Patrol: which walking policy wired in to use (empty: the newest) |
| `after_fall` After a fall | select | `"Get up on its own"` | Walk, Patrol: with a stand-up policy wired in it gets up on its own; else it is set on its feet 3 s later |
| `areas` Areas to patrol | textarea | `"all"` | Patrol: all, or names (Aisle A, Loading docks…), or your own as Name: x y; x y (see the help) |
| `incidents` Incidents | select | `"Demo (about one every 90 s)"` | Patrol: how often something happens (cartons, doors, forklifts, intruders) |
| `start_time` Patrol starts at | text | `"22:00"` | Patrol: the clock time shown and reported (hh:mm) |
| `pace` Pace | select | `"Real time"` | Patrol: As fast as it can writes the report in seconds (nothing to watch live) |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
