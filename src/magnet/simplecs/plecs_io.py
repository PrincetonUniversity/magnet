
import xmlrpc.client
import numpy as np
import os

PLECS_RPC_URL = "http://localhost:1080/RPC2"

MODELS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models'
))


def get_plecs_server():
    return xmlrpc.client.ServerProxy(PLECS_RPC_URL)


def load_model(server, model_name):
    model_path = os.path.normpath(os.path.join(MODELS_DIR, model_name))
    server.plecs.load(model_path)


def close_model(server, model_name):
    try:
        server.plecs.close(model_name)
    except Exception:
        pass


def extract_full_cycle_by_rising_zero(x, t, y, cycle_index=8):
    s = np.sign(x)
    idx = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    if len(idx) <= cycle_index:
        raise RuntimeError(
            f"Not enough zero crossings: have {len(idx)}, need > {cycle_index}"
        )
    a, b = idx[cycle_index - 1] + 1, idx[cycle_index] + 1
    return (t[a:b] - t[a]), x[a:b], y[a:b]


def parse_plecs_outports(res):
    t = np.asarray(res["Time"], dtype=float)
    vals = res["Values"]

    if isinstance(vals, (list, tuple)) and len(vals) >= 2:
        H = np.asarray(vals[0], dtype=float)
        B = np.asarray(vals[1], dtype=float)
    else:
        arr = np.asarray(vals, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise RuntimeError(f"Unexpected Values shape: {arr.shape}")
        H = arr[:, 0]
        B = arr[:, 1]

    m = min(len(t), len(H), len(B))
    return t[:m], H[:m], B[:m]


def simulate_bh_cycle(server, model_name, model_vars, stop_time, cycle_index=8):
    opts = {
        "ModelVars": model_vars,
        "SolverOpts": {
            "StartTime": 0.0,
            "StopTime": float(stop_time),
        },
    }

    res = server.plecs.simulate(model_name, opts)
    t, H, B = parse_plecs_outports(res)

    t1, H1, B1 = extract_full_cycle_by_rising_zero(
        H, t, B, cycle_index=cycle_index
    )

    B1 = B1 - np.mean(B1)
    return t1, H1, B1


def run_plecs_bh_cycle(model_name, model_vars, stop_time, cycle_index=8):
    server = get_plecs_server()
    load_model(server, model_name)
    try:
        return simulate_bh_cycle(server, model_name, model_vars, stop_time, cycle_index)
    finally:
        close_model(server, model_name)
