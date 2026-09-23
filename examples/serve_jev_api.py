"""Serve a forged checkpoint over the same HTTP protocol as TypeSafe's Jev (`POST /v1/systemone`).

    pip install "laya[serve]"
    python examples/serve_jev_api.py runs/agentguard/model      # listens on 127.0.0.1:8000

`laya-serve` only knows Laya's published checkpoints, so this builds its app around a router whose
default model is the forged one. Clients written for Jev work unchanged; point them at this URL.
The forge's thresholds are not applied here: the server returns probabilities, as Jev does.
"""

import sys

import uvicorn
from laya.router import Router
from laya.serve import create_app

model = sys.argv[1] if len(sys.argv) > 1 else "runs/agentguard/model"
router = Router(models={"english": model})
router.preload(["english"])  # the default preload would also fetch Laya's other two checkpoints
app = create_app(router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
