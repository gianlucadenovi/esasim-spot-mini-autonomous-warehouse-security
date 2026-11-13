# Spot · realistic warehouse 3D concept

Procedural warehouse shelves, pallets, lighting, Spot-like quadruped and surveillance route in 3D

3D procedural preview of training terrain or warehouse and labeled proxy robot. Uses recorded pose CSV after run; proxy not the physical robot articulated mesh.

An [EsaSim](https://esacode.com/) blueprint block: a view that draws in the browser, in JavaScript (`frontend.js`, `function render(ctx)`).

| | |
|---|---|
| Block | `ws-spot-realistic-warehouse-3d-concept` · version 2 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |

## Inputs

| Port | Type | Description |
|---|---|---|
| `data` Patrol telemetry | Table (csv) | Optional. |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` Scene | select | `"patrol"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
