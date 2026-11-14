# Spot · Autonomous Warehouse Security

A Boston Dynamics Spot quadruped learns, in a physics simulation, to stand up, to walk, and to handle slopes, stairs and obstacles. It then patrols a warehouse at night: it walks its round, hears and sees what happens, investigates, and raises the alarm when there is an intruder.

## The robot

Spot comes from **MuJoCo Menagerie** (Google DeepMind; the model is by Clearpath Robotics, BSD-3 licence). It keeps the real robot's 12 joints, its masses, inertias and joint limits, and its 32.9 kg. The 3D view draws Spot from the Menagerie meshes, in its yellow and black.

The physics run in **MuJoCo**, the engine robotics research uses. On an NVIDIA GPU, MuJoCo Warp simulates thousands of robots at once; the CPU also works, more slowly.

Each motor is a position loop (stiffness 300 N·m/rad) with Spot's torque limits. The feet touch the ground as spheres with friction.

## The four flows

| Flow | What it does | Writes |
|---|---|---|
| **1 · Learn to stand and self-right** | Each attempt starts with the robot lying where a fall left it: on its side or on its back. It learns by trial and error (PPO) to roll over, get up and stay up. | `spot_standup_<date>_<score>.pt` |
| **2 · Learn to walk** | It learns a trot at whatever speed and turn it is told, on a flat floor, while being shoved now and then. | `spot_walk_<date>_<score>.pt` |
| **3 · Slopes, stairs and obstacles** | It carries on from flow 2 on the obstacle course: bumps, blocks, a 12.5 % slope up to a plateau and down, ten 12 cm steps up and ten down, practice stairs, curbs, moving tiles and ice. It also gets an elevation map of the ground ahead, as its cameras would give it, and learns to use it. | `spot_walk_course_<date>_<score>.pt` |
| **4 · Warehouse surveillance · 30 minutes** | The patrol, live in 3D (details below). | `incident_report.md`, `incidents.json`, `patrol_log.csv` |

Flows 1 to 3 save their policies to the **spot-policy** dataset. Each run replays one attempt in the 3D view every 20 s and charts the learning curve.

## The patrol (flow 4)

**The warehouse.** A 42 × 25 m hall with:

- four loading docks with roller doors;
- three aisles of 8 m pallet racking, full of goods, with a cross aisle;
- a staging area of stretch-wrapped pallets;
- an east floor with pallets and a pallet truck;
- a mezzanine office with glass walls, reached by 7 steps of 12 cm or a 15 % ramp;
- the robot's charging dock.

The walls, racking, stacks and office are real obstacles in the physics. The stairs and ramp are real heights.

**Its round.** It patrols the areas listed in *Areas to patrol*. Use `all`, list area names (Loading docks, Aisle A, Aisle B, Aisle C, Staging area, East floor, Mezzanine office), or add your own as `Name: x y; x y`. It plans its way with A* on a 20 cm map of the floor. It stops at each area's checkpoint to look around.

**Its senses.**

- **Hearing:** a sound gets 6 dB quieter each time the distance doubles, and another 6 dB through each block of racking or wall. Below 42 dB the robot does not notice it.
- **Sight:** cameras with a 110° field of view and 14 m range, which need a clear line of sight.

**What happens.** Incidents come at random times and places that the robot does not know in advance:

- a carton falls from a shelf;
- a dock door rattles in the wind;
- a forklift driven by staff (wearing a badge);
- an intruder forces a dock door;
- glass breaks in the office and someone climbs in.

The robot goes to where the sound came from, finds what made it, and decides what it is. A carton, a door or staff on a forklift are **logged**. A person without a badge **raises the alarm**: the siren sounds, the operator is alerted, and the robot follows the person at a distance until they leave. When the session ends, it walks back to its dock.

**In the 3D view:**

- the mission panel: time, state, area, battery, distance, incidents and alarms;
- the event feed and a map of the hall;
- its route, where it is heading, and its field of view;
- sound rings, people and forklifts;
- alarm beacons and the siren.

**Buttons:** *Acknowledge alarm*, *Intruder drill* (an intruder enters now), *End patrol*, and a *Speed* slider.

**Settings:**

- *Incidents:* Demo (about one every 90 s), Realistic (about one every 4 minutes), or None.
- *Seed:* chooses which night you get; the same seed gives the same night.
- *Pace → As fast as it can:* writes the full report in seconds, without watching.

## What is real, and what is not

- **The locomotion is learned, not scripted.** Each step the robot takes in flows 1 to 3, and in the patrol once a walking policy is wired in, comes from the neural network acting in the physics.
- **Training needs a GPU to give good results.** On one NVIDIA GPU (4096 robots), plan on about 1 hour for flow 1, about 1.5 hours for flow 2, and about 2 hours for flow 3. A few minutes on a CPU only shows the robot beginning to learn.
- **The patrol can run before training.** If no walking policy is wired in, it runs **animated** and says so on screen: the body follows its path and the legs are drawn from a trot, without physics. Everything else is the same: the route, the senses, the decisions and the report. Once the **spot-policy** dataset exists (after flow 2 or 3), wire it to the patrol's *Policies* input (see *Running it*).
- **The incidents and senses are simulated.** The sound and vision models above are simplified on purpose. Alarms are not connected to a real security system.
- **This is a simulation, not a validated digital twin.** It is not a Boston Dynamics product. Real Spot hardware has its own controller and Autowalk missions.

## Running it

1. Run flows **1**, **2** and **3** on the machine with the NVIDIA GPU. *Run on: auto* chooses it.
2. Run flow **4**. It works straight away, animated. Once flow 2 or 3 has made the **spot-policy** dataset, add a *Dataset* node for it and wire it to the patrol block's *Policies* input, so the learned policy walks the round in the physics. The patrol runs in real time for 30 minutes on any machine; the CPU is enough for one robot.
3. Watch the 3D view. Use *Intruder drill* to show an intrusion on demand. Read the report when the session ends.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to stand and self-right](blueprints/1-learn-to-stand-and-self-right.json) · 8 blocks · version 5
- [2 · Learn to walk](blueprints/2-learn-to-walk.json) · 9 blocks · version 4
- [3 · Learn to look around](blueprints/3-learn-to-look-around.json) · 0 blocks
- [3 · Slopes, stairs and obstacles](blueprints/3-slopes-stairs-and-obstacles.json) · 9 blocks · version 4
- [4 · Warehouse surveillance · 30-minute session](blueprints/4-warehouse-surveillance-30-minute-session.json) · 10 blocks · version 5

### Workspace blocks it uses

- [Quadruped · Spot](blocks/ws-quadruped-spot) · version 3

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
