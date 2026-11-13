# Spot · Warehouse Security

Four flows: **1 Stand / self-right**, **2 Walk**, **3 Obstacles, ramps and stairs**, **4 Warehouse surveillance (30 minutes)**.

## Model and licensing
Use the **Boston Dynamics Spot** model from Google DeepMind's MuJoCo Menagerie (BSD-3-Clause), not an official Spot Mini release. Robot model source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot . Its physical approximation, gains and inertial values must not be represented as a validated digital twin of production Spot hardware. Official reference: https://support.bostondynamics.com/s/article/spot-specifications .

## Demonstration integrity
The stand/walk/terrain flows run dynamics in MuJoCo and tune a small gait controller, with exported scores and parameter checkpoints. This is *not* a trained commercial-grade navigation stack; self-righting and stair traversal are experimental and are assessed rather than presumed successful. The patrol flow runs a scripted incident sensor/navigation simulation; alarms are synthetic, not connected to a real security system. Warehouse 3D graphics are procedurally rendered, not a photogrammetric scan. No real-world intrusion surveillance is performed.

## Operation
Run flows in order. Controls/settings are on each flow's simulator block; use the live 3D control panel and telemetry. The robot asset is fetched from GitHub on first run and cached on the executing machine. Network access is required for the first fetch. Run on a CPU machine if GPU or MuJoCo Warp is unavailable.

Outputs include CSV telemetry, incident logs, simulation assessments and JSON policy/configuration. A 30-minute simulated patrol is separate from wall-clock execution; review event records before investor presentations.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to stand and self-right](blueprints/1-learn-to-stand-and-self-right.json) · 5 blocks · version 2
- [2 · Learn to walk](blueprints/2-learn-to-walk.json) · 5 blocks · version 1
- [3 · Slopes, stairs and obstacles](blueprints/3-slopes-stairs-and-obstacles.json) · 5 blocks · version 1
- [4 · Warehouse surveillance · 30-minute session](blueprints/4-warehouse-surveillance-30-minute-session.json) · 5 blocks · version 4

### Workspace blocks it uses

- [Spot · MuJoCo and warehouse patrol](blocks/ws-spot-mujoco-and-warehouse-patrol) · version 4

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
