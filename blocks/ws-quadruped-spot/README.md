# Quadruped · Spot

Boston Dynamics Spot in MuJoCo: learns to stand, walk and climb (PPO, GPU or CPU), then patrols a warehouse, hearing and seeing incidents and raising alarms

Boston Dynamics Spot (MuJoCo Menagerie, BSD-3: 12 joints; masses scaled to Spot's 32.5 kg, torques from Boston Dynamics' data) in MuJoCo. It learns by trial and error (PPO), many robots at once on an NVIDIA GPU (MuJoCo Warp) or on the CPU, then patrols a warehouse.

What to do:
- Learn to stand: it starts lying where a fall left it (on its side, on its back) and learns to roll over, get up and stay up, in lessons: first from its side with a help (an upward force and a torque that rolls it upright), then from its back too, the help going down to none. The stand-up score is how often it gets up on its own from any fall.
- Learn to look around: its feet planted, it aims its front cameras with its legs: body up, down, left, right, lower.
- Learn to walk: a trot at the speed and turn asked, and standing still, four feet down, when told to stop. On the Obstacle course (slopes, stairs, curbs, rough ground, moving tiles, ice) or in the Warehouse (the patrol's round, its stairs, pallets, cartons) it also gets an elevation map of the ground ahead from its cameras to use.
- Learn to take pushes: the walking policy goes on learning under shoves from any side (stronger as it copes), walking or standing.
- Walk: the newest policy drives it live in the 3D view: speed, way round, push it, reset.
- Patrol the warehouse: a session of surveillance (30 minutes by default) in a 42 x 28 m warehouse: four loading docks, three 4 m aisles of pallet racking and a row along the west wall, staging, a conveyor along the walls with loading bays and a recycling bin, two forklifts parked, a mezzanine office (7 steps; its ramp is for trolleys) where its charging dock is. It walks a round through the areas asked for (A* on a map of the floor, round what lies on it), stops at checkpoints to look (with the look-around policy), hears and sees (110° cameras, 14 m). A sound or a movement sends it to look: a carton fallen, a door in the wind are logged; an intruder (one a session) raises the alarm and it follows them until they leave. At the end it backs onto its charging dock. It writes incident_report.md, incidents.json and patrol_log.csv.

Areas: all, or names (Loading docks, Aisle A, Aisle B, Aisle C, Staging area, East floor, West racking, Mezzanine office), or your own: Name: x y; x y (metres; x along the south wall, dock 1 at x 3; y toward the north wall, 0 at the dock aisle).

The patrol is always in the physics: every step is the walking policy's, in MuJoCo, and you can push the robot (Push, then click on it), and knock the loose cartons about. It uses the policies wired to Policies, or else those left on the machine; a flat-floor policy keeps its round to the flat floor. Good walking needs GPU training: 1 to 2 hours per stage on one NVIDIA GPU.

3D view controls: Patrol: Acknowledge alarm, Intruder drill, End patrol, Push, Push strength, Speed; Log shows the mission panel. Walk: Speed, Push, Reset. Person, forklift: Next Stop Willoughby, Kamran Mughal (CC BY 4.0).

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-quadruped-spot` · version 22 |
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
| `mode` What to do | select | `"Learn to walk"` | Learn to stand: get up. Learn to look around: aim its cameras with its legs. Learn to walk: trot. Learn to take pushes: the walk under shoves. Walk: live. Patrol the warehouse: the surveillance session |
| `robot` Robot | select | `"Boston Dynamics Spot"` | Spot: 1.1 m long, 32.5 kg, 12 joints (MuJoCo Menagerie; masses and torque limits set to Boston Dynamics' figures) |
| `accelerator` Run on | select | `"auto"` | auto: an NVIDIA GPU machine (MuJoCo Warp; PyTorch with CUDA installed on its own if missing). CPU: any machine, fewer robots |
| `terrain` Ground | select | `"Flat floor"` | Learn to walk, Walk. Obstacle course: slopes, stairs up and down, curbs, rough ground, moving tiles, ice. Warehouse: the patrol's own round, up and down its stairs, over floor pallets and cartons (carry on from an obstacle course policy) |
| `robots` Robots learning at once | number | `4096` | Learn: at most 512 on the CPU |
| `minutes` Minutes | number | `90` | How long it learns, walks or patrols (Patrol: 30 is a night round) |
| `max_speed` Top speed asked (m/s) | number | `1.2` |  |
| `push` Pushes up to (m/s) | number | `1` | Learn to walk: a shove every 5 to 10 s; 0: none. Learn to take pushes: up to this (at least 2.5), every 2 to 5 s |
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
