# Spot · MuJoCo and warehouse patrol

Spot Menagerie physics experiments and 30-minute synthetic warehouse surveillance

Modes stand/walk/terrain perform actual MuJoCo physics and simple randomized gait-controller tuning; patrol produces simulated navigation/sensor alarms. All results labeled honestly.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-spot-mujoco-and-warehouse-patrol` · version 4 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Controller / waypoints | Files (json) |  |
| `table` Trials or patrol telemetry | Table (csv) |  |
| `report` Audit and incident report | Files (md, json) |  |
| `poses` 3D path and physical pose | Table (csv) | Physical MuJoCo base trajectory on training flows; synthetic route on patrol |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` Flow | select | `"stand"` |  |
| `minutes` Simulated patrol minutes | number | `30` |  |
| `trials` Physics optimization trials | number | `8` |  |
| `seed` Scenario seed | number | `17` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
