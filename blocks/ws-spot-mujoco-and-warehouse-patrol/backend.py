# ── EsaSim block ──────────────────────────────────────────────────────
# Spot · MuJoCo and warehouse patrol (ws-spot-mujoco-and-warehouse-patrol) · version 3 · machine block, Python: def run(ctx)
# Spot Menagerie physics experiments and 30-minute synthetic warehouse surveillance
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-13
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   (none)
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (json)              Controller / waypoints
#   table       Table (csv)               Trials or patrol telemetry
#   report      Files (md, json)          Audit and incident report
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    Flow = "stand" of stand | walk | terrain | patrol
#   minutes     number    Simulated patrol minutes = 30
#   trials      number    Physics optimization trials = 8
#   seed        number    Scenario seed = 17
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages mujoco>=3.3,<4 (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Spot warehouse security prototype: physical MuJoCo dynamics; explicit evaluation outputs.
The published Menagerie model is retrieved on worker only; no claims of official Spot Mini parity.
"""
import os, math, time, json, csv, random, zipfile, urllib.request, tempfile, pathlib
from xml.etree import ElementTree as ET

BASE="https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
def load_spot(mujoco, ctx):
    root=pathlib.Path.home()/".esasim"/"robot_assets"/"menagerie_spot"
    scene=root/"boston_dynamics_spot"/"scene.xml"
    if not scene.exists():
        root.mkdir(parents=True,exist_ok=True)
        ctx.log("Downloading BSD-3-Clause Spot asset from Google DeepMind MuJoCo Menagerie")
        zfile=root/"menagerie.zip"
        urllib.request.urlretrieve(BASE,zfile)
        with zipfile.ZipFile(zfile) as z:
            for n in z.namelist():
                if "/boston_dynamics_spot/" in n and not n.endswith("/"):
                    short=n.split("/boston_dynamics_spot/",1)[1]
                    dest=root/"boston_dynamics_spot"/short
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(n) as source_stream:
                        with open(dest,"wb") as target_stream:
                            import shutil
                            shutil.copyfileobj(source_stream,target_stream)
        zfile.unlink(missing_ok=True)
    if not scene.exists(): raise RuntimeError("Spot model unavailable after download")
    ctx.log("Source model: "+str(scene))
    return scene

def warehouse_geometry(xml, terrain=False):
    # X and Y metres. Physical collision geometry, incl racks/pallets, ramps and steps.
    world=xml.find("worldbody")
    def box(name,x,y,z,hx,hy,hz,rgba="0.40 0.46 0.52 1"):
        ET.SubElement(world,"geom",name=name,type="box",pos=f"{x} {y} {z}",size=f"{hx} {hy} {hz}",rgba=rgba,contype="1",conaffinity="1",friction="0.95 0.005 0.0001")
    if terrain:
        # real rigid collision bodies: 8-step staircase and slope, accessible for testing
        for i in range(8):
            box("step_"+str(i),2.0+i*.42,3, (i+1)*.09/2,.21,1.1,(i+1)*.09/2,"0.45 0.48 0.49 1")
        for i in range(8):
            box("down_step_"+str(i),9+i*.42,3,(8-i)*.09/2,.21,1.1,(8-i)*.09/2)
        box("ramp",6.8,-3,.22,2,1.2,.10,"0.47 0.51 0.56 1")
    else:
        box("north_wall",0,15,2.5,21,.14,2.5)
        box("south_wall",0,-15,2.5,21,.14,2.5)
        box("west_wall",-21,0,2.5,.14,15,2.5)
        box("east_wall",21,0,2.5,.14,15,2.5)
        for aisle in range(4):
            y=-9+aisle*6
            for side in (-1,1):
                for k in range(5):
                    x=-12+k*5.5
                    box(f"rack_{aisle}_{side}_{k}",x,y+side*1.05,1.1,1.9,.42,1.1,"0.15 0.29 0.38 1")
                    for j in range(3):
                        box(f"crate_{aisle}_{side}_{k}_{j}",x-.8+j*.8,y+side*1.05,.5+j*.55,.32,.30,.22,"0.55 0.35 0.18 1")
    return xml

def model_scene(mujoco,ctx,terrain=False):
    p=load_spot(mujoco,ctx)
    xml=ET.parse(p)
    warehouse_geometry(xml,terrain)
    # ground plane from source; ensure it is wide enough
    modified=p.parent/("scene_esasim_terrain.xml" if terrain else "scene_esasim_warehouse.xml")
    xml.write(modified,encoding="unicode")
    try:
        m=mujoco.MjModel.from_xml_path(str(modified))
    except Exception as ex:
        raise RuntimeError(f"Menagerie asset compilation failed: {ex}")
    return m

def assets_at(folder):
    a={}
    for root,_,files in os.walk(folder):
        for f in files:
            path=os.path.join(root,f)
            rel=os.path.relpath(path,folder).replace("\\","/")
            if f.lower().endswith((".obj",".stl",".png",".jpg",".mtl",".dae")):
                with open(path,"rb") as inp:a[rel]=inp.read()
    return a

def simulate(mujoco,m, mode, sec, gains, seed, callback=None):
    rng=random.Random(seed)
    d=mujoco.MjData(m)
    if m.nkey: mujoco.mj_resetDataKeyframe(m,d,0)
    else: mujoco.mj_resetData(m,d)
    nq=m.nu
    if nq==0:raise RuntimeError("Asset has no actuators")
    home=list(d.ctrl)
    if not any(abs(v)>1e-6 for v in home) and m.nkey:
        home=list(m.key_ctrl[0])
    # If ctrl range not known, use default fixed base joint actuator positions
    dt=m.opt.timestep
    step_control=max(1,round(.02/dt))
    if mode=="stand":
        # Initial roll: the controller is evaluated for self-righting, not assumed to succeed
        if m.nq>=7: d.qpos[3:7]=[.63,.77,0,0]; d.qpos[2]=.28
    mujoco.mj_forward(m,d)
    root0=(float(d.qpos[0]),float(d.qpos[1])) if m.nq>=7 else (0,0)
    score=0;upright=0;fell=0;energy=0;travel=0
    px=py=yaw=0.0
    nsteps=min(int(sec/dt),int(3600/dt))
    for i in range(nsteps):
        t=i*dt
        if i%step_control==0:
            phase=t*gains["frequency"]*math.tau
            for j in range(nq):
                # Very simple periodic PD target perturbation, optimized per morphology.
                nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,j) or ""
                offset=0
                if mode!="stand":
                    off=(j%3)
                    leg=(j//3)%4
                    side=(leg%2)*math.pi
                    fore=(leg//2)*math.pi
                    offset=gains["amplitude"]*math.sin(phase+side+fore+(off*.65))
                    if off==2:offset=abs(offset)*.6
                home_j=home[j] if j<len(home) else 0
                target=home_j+offset
                lo,hi=m.actuator_ctrlrange[j]
                if m.actuator_ctrllimited[j]:target=max(lo,min(hi,target))
                d.ctrl[j]=target
        mujoco.mj_step(m,d)
        if i%step_control==0 and m.nq>=7:
            quat=d.qpos[3:7]
            # z alignment of base; quaternion convention w x y z
            up=1-2*(quat[1]**2+quat[2]**2)
            z=float(d.qpos[2])
            good=up>.70 and z>.25
            upright+=int(good)
            fell+=int(not good)
            energy+=sum(abs(float(v)) for v in d.actuator_force)*dt
            px=float(d.qpos[0]);py=float(d.qpos[1])
            yaw=math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
            if callback and i%(step_control*5)==0:callback(t,px,py,z,yaw,good,up)
    total=max(1,(nsteps+step_control-1)//step_control)
    travel=math.hypot(px-root0[0],py-root0[1])
    stable=upright/total
    reward=(stable*3 + min(1.0,travel/max(1,sec))*(.7 if mode!="stand" else 0) - (1-stable)*2)
    return dict(score=round(reward,4),stable_pct=round(100*stable,2),distance_m=round(travel,2),falls=int(fell),energy_proxy=round(energy,2),end=[px,py],upright_final=bool(good))

def save_csv(ctx,name,rows):
    if not rows: rows=[{"status":"no data"}]
    p=ctx.output("table",name)
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def run(ctx):
    import mujoco
    par=ctx.params
    mode=str(par.get("mode","stand")).lower()
    duration=float(par.get("minutes",30))
    seed=int(par.get("seed",17))
    ctx.progress(2,"Fetching and compiling Spot physics model")
    m=model_scene(mujoco,ctx,mode=="terrain")
    ctx.log("Model compiled with %d DOF, %d actuators, %d collision geoms"%(m.nv,m.nu,m.ngeom))
    ctx.write_json("report","robot_description.json",{"model":"Boston Dynamics Spot · MuJoCo Menagerie","license":"BSD-3-Clause","repository":"https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot","simulator":"MuJoCo","dof":m.nv,"actuators":m.nu,"geometries":m.ngeom,"limits":"Unverified model/control fidelity. Not a production digital twin."})
    if mode!="patrol":
        trials=max(2,min(24,int(par.get("trials",8))))
        rows=[];best=None
        ctx.log("Optimizing a small periodic joint controller by random-search policy selection; not PPO")
        for n in range(trials):
            rng=random.Random(seed+n)
            g={"frequency":rng.uniform(.7,2.2),"amplitude":rng.uniform(.06,.40)}
            res=simulate(mujoco,m,mode,6 if mode!="terrain" else 12,g,seed+n)
            rows.append({"trial":n+1,**g,**res})
            if best is None or res["score"]>best["score"]:best={"gains":g,**res}
            ctx.telemetry("train",{"trial":n+1,"score":res["score"],"stable_pct":res["stable_pct"],"distance_m":res["distance_m"]})
            ctx.progress(5+85*(n+1)/trials,f"Physical trial {n+1}/{trials}")
        ctx.write_json("policy","spot_"+mode+"_controller.json",{"mode":mode,"algorithm":"random-search over periodic joint controller","best":best,"warning":"Experimental; stability and actual task success must be validated from metrics"})
        save_csv(ctx,"training_"+mode+".csv",rows)
        ctx.write_text("report","assessment.md",f"# {mode.title()} training assessment\n\nModel: MuJoCo Menagerie Boston Dynamics Spot.\n\nBest score: {best['score']}\n\nUpright ratio: {best['stable_pct']}%\n\nDistance: {best['distance_m']} m.\n\nPolicy: simple periodic joint control tuned by random search; no claim of robust learned stand-up, walking or stair traversal.\n")
        ctx.progress(100,"Controller candidate and assessment saved")
        return {"best":best}
    # Warehouse patrol: 30 simulation minutes, waypoint and synthetic intrusion events.
    # Navigation is a high-level surveillance planner, not a claim of low-level learned locomotion.
    sim_minutes=max(1,min(180,duration))
    zones=[(-17,-11),(-6,-11),(6,-11),(17,-11),(17,0),(6,0),(-6,0),(-17,0),(-17,11),(-6,11),(6,11),(17,11)]
    rng=random.Random(seed);step_s=1.0;now=[-18.,-12.];target=0
    events=[];rows=[];state="PATROL";active=None
    scripted=[(5*60+25,"noise",9,10),(12*60+10,"motion",-13,5),(22*60+5,"motion",16,-8)]
    handled=set()
    for sec in range(int(sim_minutes*60)):
        for k,(evtime,etype,ex,ey) in enumerate(scripted):
            if sec>=evtime and k not in handled:
                handled.add(k);active={"type":etype,"x":ex,"y":ey,"when":sec};state="INVESTIGATE"
                events.append({"time_s":sec,"kind":"sensor_trigger","sensor":etype,"location":[ex,ey],"confidence":.84,"synthetic":True})
                ctx.telemetry("event",{"text":f"Simulated {etype} detected in zone {ex},{ey}"})
        dest=(active["x"],active["y"]) if active else zones[target]
        dx=dest[0]-now[0];dy=dest[1]-now[1];dist=math.hypot(dx,dy)
        if dist<.8:
            if active:
                alarm=active["type"]=="motion"
                events.append({"time_s":sec,"kind":"inspection","location":[round(now[0],2),round(now[1],2)],"classification":"simulated person" if alarm else "ambient machinery","alarm":alarm,"synthetic":True})
                if alarm:
                    ctx.telemetry("event",{"text":"SIMULATED ALARM · visual motion classification (no external alarm fired)"})
                active=None;state="PATROL"
            else:target=(target+1)%len(zones)
        else:
            velocity=min(.8,dist);now[0]+=dx/dist*velocity;now[1]+=dy/dist*velocity
        row={"t":sec,"x":round(now[0],3),"y":round(now[1],3),"z":.61,"yaw":round(math.atan2(dy,dx),3),"mode":state,"zone":target,"alarms":sum(1 for e in events if e.get("alarm"))}
        rows.append(row)
        if sec%5==0:
            ctx.telemetry("patrol",row)
            ctx.progress(round(100*(sec+1)/(sim_minutes*60),1),f"Simulated surveillance {sec//60:02d}:{sec%60:02d} / {sim_minutes:02.0f}:00")
    save_csv(ctx,"patrol_telemetry.csv",rows)
    ctx.write_json("report","incident_log.json",events)
    ctx.write_json("policy","warehouse_waypoints.json",{"zones":zones,"synthetic_events":scripted,"note":"Editable waypoints by modifying simulator params / project; kinematic planner simulation"})
    ctx.write_text("report","surveillance_report.md",f"# Warehouse patrol — {sim_minutes:.0f} simulated minutes\n\n- Waypoints: {len(zones)}\n- Synthetic events: {sum(e['kind']=='sensor_trigger' for e in events)}\n- Synthetic alarms: {sum(bool(e.get('alarm')) for e in events)}\n- Security equipment integration: **none**\n- Patrol positions: kinematic mission planner, **not physical locomotion simulation**.\n- Locomotion training assessed separately in flows 1–3.\n")
    ctx.progress(100,"Patrol evidence and incident report saved")
    return {"events":len(events)}
