# Spot · Autonomous Warehouse Security

A Boston Dynamics Spot quadruped learns, in a physics simulation, to stand up, to aim its cameras by moving its body, to walk, and to handle slopes, stairs and obstacles. It then patrols a warehouse at night, still in the physics: it walks its round, hears and sees what happens, investigates, and raises the alarm when there is an intruder.

## The robot

Spot comes from **MuJoCo Menagerie** (Google DeepMind; the model is by Clearpath Robotics, BSD-3 licence). It keeps the real robot's 12 joints, joint limits and mass distribution, and weighs **32.5 kg**, Boston Dynamics' figure for Spot (the model as published came to 50 kg, so every part's mass and inertia is scaled down in the same proportion). The 3D view draws Spot from the Menagerie meshes, in its yellow and black.

The physics run in **MuJoCo**, the engine robotics research uses. On an NVIDIA GPU, MuJoCo Warp simulates thousands of robots at once; the CPU also works, more slowly.

Each motor is a position loop (stiffness 300 N·m/rad) limited to Spot's published torques: **45 N·m** at the hips (a 0.88 N·m motor through a 51:1 gear) and up to **113 N·m** at the knees. Boston Dynamics gives the knee 30.6 to 113.2 N·m depending on its angle; the simulation allows the peak at every angle, so a deeply bent knee is somewhat stronger than the real one. The feet touch the ground as spheres with friction.

Like the real Spot, it can roll over from its back and stand up: Boston Dynamics ships self-righting as a feature of the robot, and flow 1 learns it within these limits.

## The five flows

| Flow | What it does | Writes |
|---|---|---|
| **1 · Learn to stand and self-right** | Each attempt starts with the robot lying where a fall left it: on its side or on its back. It learns by trial and error (PPO) to roll over, get up and stay up. | `spot_standup_<date>_<score>.pt` |
| **2 · Learn to walk** | It learns a trot at whatever speed and turn it is told, on a flat floor, while being shoved now and then. | `spot_walk_<date>_<score>.pt` |
| **3 · Learn to look around** | Spot's cameras look forward, so to look up, down or to the side it moves its body, as the real Spot does. Standing with its feet planted, it is asked for a body attitude: nose up or down by up to 26°, turned left or right by up to 29°, and between 15 cm lower and 5 cm higher. A new aim comes every 1.5 to 4 s. It learns to reach and hold each aim with its legs alone, without slipping or stepping. | `spot_look_<date>_<score>.pt` |
| **4 · Slopes, stairs and obstacles** | It carries on from flow 2 on the obstacle course: bumps, blocks, a 12.5 % slope up to a plateau and down, ten 12 cm steps up and ten down, practice stairs of 6 to 15 cm, curbs, moving tiles and ice. It also gets an elevation map of the ground ahead, as its cameras would give it, and learns to use it. It learns with a curriculum: each robot starts on flat stretches, moves up a level when it walks a whole try without falling, and moves down one when it falls early. The levels are: flat; 6 cm stairs, ice and slopes; 9 cm stairs, moving tiles and rough ground; 12 cm stairs; then 15 cm stairs and anywhere. The chart *Curriculum: average level* shows its progress. | `spot_walk_course_<date>_<score>.pt` |
| **5 · Warehouse surveillance · 30 minutes** | The patrol, live in 3D and in the physics (details below). | `incident_report.md`, `incidents.json`, `patrol_log.csv` |

Flows 1 to 4 save their policies to the **spot-policy** dataset. Each run replays one attempt in the 3D view every 20 s and charts the learning curve.

## The patrol (flow 5)

**The warehouse.** A 42 × 25 m hall with:

- four loading docks with roller doors;
- three aisles of 8 m pallet racking, full of goods, with a cross aisle;
- a staging area of stretch-wrapped pallets;
- an east floor with pallets and a pallet truck;
- a mezzanine office with glass walls, reached by 7 steps of 12 cm or a 15 % ramp;
- a roller conveyor across the east floor, crossed by a steel step-over bridge (7 steps up, 7 down);
- a dispatch platform at the west end, 48 cm high, reached by 4 steps and left by a 12.5 % ramp;
- aisles B and C cluttered: pallets left on the floor (14 cm) and fallen cartons to step over or onto;
- the robot's charging dock.

The walls, racking, stacks, office, conveyor and clutter are real obstacles in the physics, and the stairs, ramps and platforms are real heights. These are the same challenges the robot met on the obstacle course in flow 4, so the patrol tests what it learned there.

The 3D view draws the warehouse with shadows, lamp light and surface relief on its materials: sealed concrete, steel panels, perforated racking, cardboard, pallet wood, stretch wrap, steel grating and checker plate.

**Always in the physics.** Every step Spot takes in the warehouse comes from its walking policy acting in MuJoCo. You can push it by hand, as with the humanoids: click *Push*, then click the part of the robot to push; *Push strength* sets how hard. It uses the policies in **spot-policy** (wired to its *Policies* input), or else those the training runs left on the machine. For walking it prefers the policy from flow 4. With a policy that has learned only the flat floor, its round keeps to the flat floor: it leaves out the areas behind stairs, ramps or clutter and says so. Without a walking policy the patrol stops at the start and says what to wire.

**Its round.** It patrols the areas listed in *Areas to patrol*. Use `all`, list area names (Loading docks, Aisle A, Aisle B, Aisle C, Staging area, East floor, Dispatch platform, Mezzanine office), or add your own as `Name: x y; x y`. It plans its way with A* on a 20 cm map of the floor. It stops at each area's checkpoint to look around. With the look-around policy, it aims its cameras with its body: along the racks up to the top levels, then down to the floor. When it inspects something, it raises its nose toward a person's face, lowers it toward a carton on the floor, and turns its body toward the target. The mission panel shows where the cameras point. After a fall it gets up with the stand-up policy.

**Its senses.**

- **Hearing:** a sound gets 6 dB quieter each time the distance doubles, and another 6 dB through each block of racking or wall. Below 42 dB the robot does not notice it.
- **Sight:** cameras with a 110° field of view and 14 m range, which need a clear line of sight.

**What happens.** Incidents come at random times and places that the robot does not know in advance:

- a carton falls from a shelf;
- a dock door rattles in the wind;
- a forklift driven by staff (wearing a badge);
- an intruder forces a dock door;
- glass breaks in the office and someone climbs in.

The robot goes to where the sound came from, finds what made it, and decides what it is. A carton, a door or staff on a forklift are **logged**. A person without a badge **raises the alarm**: the siren sounds, the operator is alerted, and the robot follows the person at a distance until they leave. After an incident it picks up its round where it left off, so no area is skipped. When the session ends, it walks back to its dock.

**In the 3D view:**

- the mission panel: time, state, area, battery, distance, incidents, alarms, and where the cameras point;
- the event feed and a map of the hall;
- its route, where it is heading, and its field of view;
- sound rings, people and forklifts;
- alarm beacons and the siren.

**Buttons:** *Push* (with *Push strength*), *Acknowledge alarm*, *Intruder drill* (an intruder enters now), *End patrol*, and a *Speed* slider.

**Settings:**

- *Incidents:* Demo (about one every 90 s), Realistic (about one every 4 minutes), or None.
- *Seed:* chooses which night you get; the same seed gives the same night.
- *Pace → As fast as it can:* writes the full report in seconds, without watching.

## What is real, and what is not

- **The locomotion is learned, not scripted.** Each step the robot takes, in training and in the patrol, comes from the neural networks acting in the physics.
- **The body is Spot's.** Mass (32.5 kg) and motor torques follow Boston Dynamics' published figures; the one simplification is the knee limit, held at its peak at every angle.
- **How well it walks depends on its training.** On one NVIDIA GPU (4096 robots), plan on about 1 hour for flow 1, about 1.5 hours for flow 2, about 45 minutes for flow 3, and about 2 hours for flow 4. On a CPU (512 robots) it learns far more slowly. A policy still learning the course will stumble and fall in the warehouse, in the physics, as it would anywhere.
- **The incidents and senses are simulated.** The sound and vision models above are simplified on purpose. Alarms are not connected to a real security system.
- **This is a simulation, not a validated digital twin.** It is not a Boston Dynamics product. Real Spot hardware has its own controller and Autowalk missions.

## Running it

1. Run flows **1** to **4**. With *Run on: auto* they go to the machine with the NVIDIA GPU (Gianluca De Novi PC, RTX 3070), and wait for it if it is off. If PyTorch on that machine cannot use the GPU, the block installs the CUDA build of PyTorch that the driver supports, by itself, before it trains. *Run on: CPU* trains on any machine, slowly. Policies trained before block version 7 learned on a heavier, stronger Spot (50 kg): train flows 1 to 4 again.
2. Run flow **5**. The **spot-policy** dataset is wired to the patrol's *Policies* input: it walks with the learned walking policy, looks around with the look-around policy, and gets up with the stand-up policy. The patrol runs in real time for 30 minutes; the CPU is enough for one robot.
3. Watch the 3D view. Push the robot, and use *Intruder drill* to show an intrusion on demand. Read the report when the session ends.

## Sources for the robot's figures

- Spot's weight (32.5 kg): [Boston Dynamics, About Spot](https://dev.bostondynamics.com/docs/concepts/about_spot)
- Hip motors and gearing: [Boston Dynamics, joint control supplemental data](https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data)
- Knee torque limits: [Boston Dynamics, knee torque limits](https://dev.bostondynamics.com/docs/concepts/joint_control/knee_torque_limits)

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to stand and self-right](blueprints/1-learn-to-stand-and-self-right.json) · 9 blocks · version 6
- [2 · Learn to walk](blueprints/2-learn-to-walk.json) · 9 blocks · version 5
- [3 · Learn to look around](blueprints/3-learn-to-look-around.json) · 9 blocks
- [4 · Slopes, stairs and obstacles](blueprints/4-slopes-stairs-and-obstacles.json) · 10 blocks · version 5
- [5 · Warehouse surveillance · 30-minute session](blueprints/5-warehouse-surveillance-30-minute-session.json) · 10 blocks · version 8
- [6 · Take pushes](blueprints/6-take-pushes.json) · 8 blocks · version 1

### Workspace blocks it uses

- [Quadruped · Spot](blocks/ws-quadruped-spot) · version 19

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
