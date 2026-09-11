import json
import multiprocessing
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lightrag.api.lightrag_server import global_args,main as run_server

# 全局子进程列表（各个lightrag‑server后端）
backend_processes: List[multiprocessing.Process] = []
# 后端实例元信息：instance_name -> base_url
backend_map: Dict[str, str] = {}

# ---------------------- 后端LightRAG启动逻辑 ----------------------
def start_lightrag_instance(instance_conf: dict):
    name = instance_conf["instance_name"]
    port = instance_conf["port"]
    workspace = instance_conf["workspace"]
    env_file = Path(instance_conf["env_file"])
    print(f"[BACKEND {name}] start port={port}, workspace={workspace}, env={env_file}")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()
    try:
        global_args.host = "0.0.0.0"
        global_args.port = port
        global_args.workspace = workspace
        global_args.auto_reload = False
        run_server()
        #run_server(host="0.0.0.0", port=port, workspace=workspace, auto_reload=False)
    except Exception as e:
        print(f"[BACKEND {name}] exit with error: {e}", file=sys.stderr)

def stop_all_backends(signum, frame):
    print("\nShutting down all lightrag‑server backends...")
    for p in backend_processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    sys.exit(0)

# ---------------------- FastAPI网关定义 ----------------------
app = FastAPI(title="LightRAG Multi‑Workspace Gateway")

class SingleQueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "hybrid"
    top_k: Optional[int] = 5

class MultiQueryRequest(BaseModel):
    query: str
    instances: Optional[List[str]] = None  # 指定要检索的实例列表；None=全部实例
    mode: Optional[str] = "hybrid"
    top_k_per_backend: int = 5

@app.post("/api/multi_query", summary="多知识库并行检索（网关新增接口）")
async def multi_query(req: MultiQueryRequest):
    target_instances: List[str]
    if req.instances is None:
        target_instances = list(backend_map.keys())
    else:
        unknown = [x for x in req.instances if x not in backend_map]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown instances: {unknown}")
        target_instances = req.instances

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for inst in target_instances:
            url = f"{backend_map[inst]}/api/query"
            payload = {
                "query": req.query,
                "mode": req.mode,
                "top_k": req.top_k_per_backend
            }
            tasks.append(client.post(url, json=payload))
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    aggregated = []
    for inst, resp in zip(target_instances, responses):
        if isinstance(resp, Exception):
            aggregated.append({"instance": inst, "error": str(resp), "result": None})
            continue
        if resp.status_code != 200:
            aggregated.append({
                "instance": inst,
                "error": f"status {resp.status_code}",
                "result": None
            })
            continue
        data = resp.json()
        data["instance_name"] = inst
        aggregated.append(data)

    return {
        "query": req.query,
        "mode": req.mode,
        "per_backend_topk": req.top_k_per_backend,
        "results": aggregated
    }

# ---------------------- 网关主入口 ----------------------
def gateway_main(gateway_port: int, conf: List[Dict[str, Any]]):
    import uvicorn
    for item in conf:
        name = item["instance_name"]
        port = item["port"]
        backend_map[name] = f"http://127.0.0.1:{port}"
    uvicorn.run(app, host="0.0.0.0", port=gateway_port)

def main():
    config_path = Path("instances.json")
    if not config_path.exists():
        print(f"Config {config_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        instances_cfg = json.load(f)

    signal.signal(signal.SIGINT, stop_all_backends)
    signal.signal(signal.SIGTERM, stop_all_backends)

    # 启动所有lightrag‑server后端子进程
    for cfg in instances_cfg:
        p = multiprocessing.Process(
            target=start_lightrag_instance,
            args=(cfg,),
            name=cfg["instance_name"],
            daemon=False
        )
        p.start()
        backend_processes.append(p)
        print(f"Started backend {cfg['instance_name']}, pid={p.pid}, port={cfg['port']}")

    # 在主进程启动网关服务（也可以把网关放入单独进程）
    gateway_port = 9620
    gateway_main(gateway_port, instances_cfg)

    # 等待后端进程（网关退出后等待子进程）
    for p in backend_processes:
        p.join()

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn")
    main()
