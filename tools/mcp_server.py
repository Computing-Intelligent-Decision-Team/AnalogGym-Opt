#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) server exposing AnalogGym-Opt to LLM agents.

Tools:
  list_circuits        - available circuit instances and their categories
  describe_circuit     - design variables, ranges, init point, performance targets
  evaluate_design      - run ngspice on one design point, return all metrics + reward
  run_instance_sims    - smoke-run an instance's testbenches (sensors/refs too)
  validate_configs     - consistency check over all configs
  start_training       - launch a GRPO run in the background
  training_status      - poll a running/finished GRPO run
  stop_training        - kill a GRPO run

Setup:
  pip install mcp
  # Claude Code picks it up automatically via the repo's .mcp.json; for other
  # clients register:  python tools/mcp_server.py   (stdio transport)

All heavyweight work (ngspice, torch) runs in subprocesses of the repo's own
CLI tools, so this server stays lightweight and import-safe.
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
PY = sys.executable
ENV = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE')
LOG_DIR = os.path.join(REPO_ROOT, 'training_logs')

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.stderr.write('The "mcp" package is required:  pip install mcp\n')
    sys.exit(1)

mcp = FastMCP('analoggym-opt')


def _run(cmd, timeout):
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=ENV, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def _load_config(circuit: str):
    from circuit_config_loader import CircuitConfigLoader
    return CircuitConfigLoader(
        config_dir=os.path.join(REPO_ROOT, 'circuit_configs')).get_circuit_config(circuit)


@mcp.tool()
def list_circuits() -> str:
    """List every circuit instance: GRPO-integrated ones (amp_*/ldo_* with a YAML
    config) and simulation-only ones (sensors, voltage references)."""
    from circuit_config_loader import CircuitConfigLoader
    loader = CircuitConfigLoader(config_dir=os.path.join(REPO_ROOT, 'circuit_configs'))
    configured = sorted(loader.get_available_circuits())
    sim_root = os.path.join(REPO_ROOT, 'simulation_files')
    sim_only = sorted(d for d in os.listdir(sim_root)
                      if os.path.isdir(os.path.join(sim_root, d))
                      and d != 'sky130_pdk'
                      and d not in configured)
    return json.dumps({'grpo_integrated': configured, 'simulation_only': sim_only})


@mcp.tool()
def describe_circuit(circuit: str) -> str:
    """Describe one GRPO-integrated circuit: design variables with ranges/steps and
    the initial point, performance targets, action dimension, and category."""
    cfg = _load_config(circuit)
    return json.dumps({
        'circuit': circuit,
        'category': cfg.get('category'),
        'action_dim': cfg.get('action_dim'),
        'device': cfg.get('device'),
        'performance_targets': cfg.get('performance'),
        'sim_dir': cfg.get('base_dir'),
    }, default=str)


@mcp.tool()
def evaluate_design(circuit: str, params: str = '') -> str:
    """Simulate ONE design point with ngspice and return every measured metric,
    per-metric scores and the total reward. `params` is a JSON object of design
    variable overrides (e.g. {"W_M0": 2, "M_M11": 64}); variables not given use
    the config's init values. Takes roughly 0.5-5 minutes depending on the circuit."""
    cmd = [PY, 'tools/evaluate_design.py', '--circuit', circuit, '--json']
    tmp = None
    if params.strip():
        tmp = os.path.join(REPO_ROOT, 'simulation_output', '_mcp_point_%d.json' % os.getpid())
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, 'w') as f:
            json.dump(json.loads(params), f)
        cmd += ['--vars-file', tmp]
    try:
        rc, out, err = _run(cmd, timeout=1800)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    if rc != 0:
        return json.dumps({'error': 'evaluation failed', 'stderr': err[-2000:]})
    return out


@mcp.tool()
def run_instance_sims(instance: str) -> str:
    """Run every ngspice testbench of one instance dir (works for the
    simulation-only sensors/voltage references as well). Returns pass/fail per
    testbench; the wrdata output files stay in simulation_files/<instance>/."""
    rc, out, err = _run([PY, 'tools/run_instance_sims.py', instance], timeout=1800)
    return json.dumps({'ok': rc == 0, 'output': out[-3000:], 'stderr': err[-1000:]})


@mcp.tool()
def validate_configs() -> str:
    """Consistency-check every circuit config against its instance files
    (paths, action_dim, graph shape, vars.spice <-> device section)."""
    rc, out, err = _run([PY, 'tools/validate_configs.py'], timeout=600)
    return json.dumps({'ok': rc == 0, 'report': out[-4000:], 'stderr': err[-1000:]})


@mcp.tool()
def start_training(circuit: str, steps: int = 300, pvt_mode: str = 'proxy') -> str:
    """Launch a GRPO sizing run in the background. pvt_mode: 'tt' (nominal corner
    only), 'proxy' (TT training + VAE proxy with selective PVT verification,
    default), or 'full' (all PVT corners every step, slow). Returns the pid and
    log file to poll with training_status. Do not start two runs in the same repo
    concurrently (they share simulation_output/)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, '%s_%s.log' % (circuit, time.strftime('%Y%m%d_%H%M%S')))
    log = open(log_path, 'w')
    proc = subprocess.Popen(
        [PY, 'main_AMP_grpo.py', '--circuit', circuit, '--steps', str(steps),
         '--pvt-mode', pvt_mode],
        cwd=REPO_ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    return json.dumps({'pid': proc.pid, 'log': log_path,
                       'note': 'poll with training_status(pid, log)'})


def _pid_alive(pid: int) -> bool:
    if os.name == 'nt':
        out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@mcp.tool()
def training_status(pid: int, log: str, tail_lines: int = 40) -> str:
    """Check whether a GRPO run is still alive and return the tail of its log."""
    tail = ''
    if os.path.exists(log):
        with open(log, 'r', errors='replace') as f:
            tail = ''.join(f.readlines()[-tail_lines:])
    return json.dumps({'running': _pid_alive(pid), 'log_tail': tail})


@mcp.tool()
def stop_training(pid: int) -> str:
    """Terminate a background GRPO run by pid."""
    if os.name == 'nt':
        rc = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                            capture_output=True, text=True)
        return json.dumps({'ok': rc.returncode == 0, 'detail': rc.stdout.strip()})
    os.kill(pid, 15)
    return json.dumps({'ok': True})


if __name__ == '__main__':
    mcp.run()
