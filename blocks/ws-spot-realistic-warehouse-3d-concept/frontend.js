// ── EsaSim block ──────────────────────────────────────────────────────
// Spot · realistic warehouse 3D concept (ws-spot-realistic-warehouse-3d-concept) · version 1 · view, JavaScript: function render(ctx)
// Procedural warehouse shelves, pallets, lighting, Spot-like quadruped and surveillance route in 3D
// Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
// Saved: 2026-11-13
//
// Inputs   ctx.input("id") gives { files, stream }; ctx.rows("id"), ctx.json("id"), ctx.text("id")
//   data        Table (csv)               Patrol telemetry · optional
// Settings ctx.params.key
//   (none)
// EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
// ── end of EsaSim block header ────────────────────────────────────────

function render(ctx) {
  const el=ctx.el; el.innerHTML="";
  const box=document.createElement("div");box.style="height:560px;position:relative;background:#101923;border-radius:12px;overflow:hidden";el.appendChild(box);
  const label=document.createElement("div");label.style="position:absolute;top:12px;left:12px;color:#d5f5f8;background:#071523cc;padding:10px;border-radius:6px;font:13px monospace;z-index:2";label.textContent="WAREHOUSE SECURITY / PROCEDURAL VISUAL DEMO";box.appendChild(label);
  const THREE=ctx.THREE;
  if(!THREE){label.textContent="3D engine unavailable";return}
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x101923);
  const camera=new THREE.PerspectiveCamera(57,1,.1,200);camera.position.set(24,30,31);camera.lookAt(0,0,0);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));box.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xb8e3ff,0x303b4b,2.4));
  const lamp=new THREE.DirectionalLight(0xffffff,2.4);lamp.position.set(15,25,12);scene.add(lamp);
  const mat=(c,r=.75)=>new THREE.MeshStandardMaterial({color:c,roughness:r});
  const floor=mat(0x6c7779), steel=mat(0x214158), pallet=mat(0x946b3e), crate=mat(0xa37a4a), white=mat(0xc9d6df),yellow=mat(0xd7a52b);
  function block(w,h,d,x,y,z,m){const a=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);a.position.set(x,y,z);scene.add(a);return a}
  block(43,.22,31,0,-.12,0,floor);
  for(let k=0;k<7;k++)block(.04,.008,30,-18+k*6,.006,0,mat(0x99a8a8));
  for(let i=0;i<4;i++)for(let s of [-1,1])for(let k=0;k<5;k++){
    const z=-9+i*6+s*1.15, x=-12+k*5.5;
    for(let lev=0;lev<3;lev++){
      block(4,.10,.92,x,.60+lev*.78,z,steel);
      for(let p=-1;p<=1;p++)block(.82,.40,.68,x+p*1.06,.86+lev*.78,z,lev%2?crate:pallet)
    }
    for(let xx of [-1.94,1.94])block(.12,2.95,.12,x+xx,1.48,z,steel);
  }
  for(let x=-18;x<=18;x+=12)for(let z of [-13,13]){
    const light=new THREE.PointLight(0xb8e9ff,15,15);light.position.set(x,5,z);scene.add(light);
    block(1,.06,.32,x,4.7,z,white)
  }
  block(43,5,.16,0,2.5,-15,mat(0x333f4b));
  block(43,5,.16,0,2.5,15,mat(0x333f4b));
  block(.16,5,30,-21,2.5,0,mat(0x333f4b));
  block(.16,5,30,21,2.5,0,mat(0x333f4b));
  const dog=new THREE.Group();scene.add(dog);
  function member(w,h,d,x,y,z,m){const q=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);q.position.set(x,y,z);dog.add(q);return q}
  const gold=mat(0xe0c44b),dark=mat(0x27333c),cyan=mat(0x22d9db);
  member(.9,.35,.42,0,.65,0,gold);
  member(.24,.26,.27,.48,.73,0,dark);
  member(.1,.07,.10,.61,.77,0,cyan);
  let legs=[];
  for(let sx of [-.32,.32])for(let sz of [-.16,.16]){const upper=member(.12,.25,.12,sx,.43,sz,dark);const low=member(.11,.25,.11,sx,.17,sz,gold);legs.push([upper,low])}
  const pathpoints=[[-17,-11],[-6,-11],[6,-11],[17,-11],[17,0],[6,0],[-6,0],[-17,0],[-17,11],[-6,11],[6,11],[17,11]];
  const verts=[];for(let a of pathpoints)verts.push(new THREE.Vector3(a[0],.04,a[1]));
  const route=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(verts),new THREE.LineBasicMaterial({color:0x16cdcc}));scene.add(route);
  let theta=0,phase=0;let patrol=pathpoints,idx=0,latest="";
  function update(){
    try{
      const rs=ctx.rows("data")||[];
      if(rs.length){patrol=rs.map(r=>[+r.x,+r.y,+r.yaw]);idx=0}
    }catch(e){}
  }
  ctx.on("update",update);update();
  let frames=0;
  function animate(){
    if(!box.isConnected)return;requestAnimationFrame(animate);frames++;
    phase+=.055;
    const q=patrol.length>0?patrol[idx%patrol.length]:[0,0];
    const next=patrol[(idx+1)%patrol.length]||q;
    if(!Array.isArray(q))return;
    if(patrol.length>1 && idx<patrol.length-1){idx=(idx+1)%patrol.length}
    else if(patrol.length===12&&frames%75===0)idx=(idx+1)%12;
    const a=patrol[Math.min(idx,patrol.length-1)]||[0,0,0];
    dog.position.set(+a[0]||0,.04+(Math.sin(phase*2)*.008),+a[1]||0);
    dog.rotation.y=-(a[2]||0);
    legs.forEach((pair,i)=>{pair[0].rotation.z=.16*Math.sin(phase+i*Math.PI);pair[1].rotation.z=.2*Math.sin(phase+i*Math.PI+1)});
    if(frames%1000===0&&patrol.length===12)idx=(idx+1)%12;
    renderer.render(scene,camera)
  }
  function resize(){const w=box.clientWidth||700;const h=box.clientHeight||560;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}
  ctx.on("resize",resize);resize();animate();
}
