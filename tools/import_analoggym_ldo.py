#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Import the AnalogGym Basic_LDO (RGNN_RL/simulations/LDO_TB*) into this repo's
circuit-instance format, as `ldo_basic`.

Produces:
  simulation_files/ldo_basic/LDO_BASIC_netlist.txt      renamed netlist (W_Mx/L_Mx/M_Mx params)
  simulation_files/ldo_basic/LDO_BASIC_ACDC.cir         ACDC testbench (Vdrop/LNR/LR/PSRR/loop/OP)
  simulation_files/ldo_basic/LDO_BASIC_Tran.cir         load-step transient testbench
  simulation_files/ldo_basic/LDO_BASIC_vars.spice       initial design variables
  simulation_files/ldo_basic/LDO_BASIC_dev_params.spice .OP parameter extraction
  circuit_configs/ldo_basic.yaml                        config (device/performance/hierarchy/graph)

The evaluation environment for this instance is LdoEnv (see LdoEnv.py); the circuit
is selected like any other by its config name `ldo_basic`.

Usage:  python tools/import_analoggym_ldo.py --upstream <analoggym_root> [--force]
"""
import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from import_analoggym_amp import (read_joined_lines, parse_num, fmt, write_file,
                                  CAP_UNIT, MOS_PARAM_RE)

PREFIX = 'LDO_BASIC'
SHORT = 'ldo_basic'
SUBCKT = 'Basic_LDO'


def parse_ldo_netlist(path):
    """Parse the upstream Basic_LDO netlist (7 pins, Ib already exposed)."""
    mos, res, caps = [], [], []
    for ln in read_joined_lines(path):
        low = ln.lower()
        if low.startswith('.subckt'):
            pins = [t.lower() for t in ln.split()[2:]]
            if pins != ['gnda', 'vdda', 'vinn', 'vout', 'vfb', 'vinp', 'ib']:
                raise ValueError('unexpected LDO pins: %s' % pins)
        elif low.startswith('.ends'):
            break
        elif low.startswith('xm'):
            toks = ln.split()
            idx = int(re.match(r'^xm(\d+)', low).group(1))
            nets = [t.lower() for t in toks[1:5]]
            model = toks[5]
            params = {}
            for t in toks[6:]:
                k, v = t.split('=', 1)
                params[k.lower()] = v.strip("'")
            group, exprs = None, {}
            for key in ('l', 'w', 'm'):
                expr = params[key]
                m = MOS_PARAM_RE.search(expr)
                if not m:
                    raise ValueError('no group param in %s of XM%d' % (expr, idx))
                g = 'M%d' % int(m.group(1))
                group = group or g
                if group != g:
                    raise ValueError('XM%d mixes groups' % idx)
                exprs[key] = expr.replace(m.group(0), '{P}')
            mos.append({'idx': idx, 'nets': nets, 'model': model,
                        'group': group, 'exprs': exprs})
        elif re.match(r'^r\d+', low):
            toks = ln.split()
            res.append({'name': toks[0].upper(), 'nets': [toks[1].lower(), toks[2].lower()],
                        'value': toks[3]})
        elif low.startswith('xc'):
            toks = ln.split()
            caps.append({'name': toks[0][1:].upper(), 'nets': [toks[1].lower(), toks[2].lower()]})
        else:
            raise ValueError('unhandled LDO netlist line: %s' % ln)
    return mos, res, caps


def parse_ldo_vars(path):
    """Upstream LDO_TB_vars.spice: long mosfet_* params, current_0_bias, M_C0, M_CL."""
    vals = {}
    for ln in read_joined_lines(path):
        ln = re.sub(r'^\.param\s*', '', ln, flags=re.IGNORECASE)
        for tok in ln.split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                vals[k.lower()] = parse_num(v)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', required=True, help='AnalogGym repo checkout root')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    src_dir = os.path.join(args.upstream, 'RGNN_RL', 'simulations')
    sim_dir = os.path.join(REPO_ROOT, 'simulation_files', SHORT)
    cfg_path = os.path.join(REPO_ROOT, 'circuit_configs', '%s.yaml' % SHORT)
    if not args.force and (os.path.exists(sim_dir) or os.path.exists(cfg_path)):
        raise SystemExit('instance %s already exists (use --force to overwrite)' % SHORT)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    mos, res, caps = parse_ldo_netlist(os.path.join(src_dir, 'LDO_TB.txt'))
    upvars = parse_ldo_vars(os.path.join(src_dir, 'LDO_TB_vars.spice'))

    mos_sorted = sorted(mos, key=lambda d: d['idx'])
    groups = {}
    for d in mos:
        groups.setdefault(d['group'], []).append(d)
    group_order = sorted(groups, key=lambda g: int(g[1:]))

    # ------- initial values (from upstream vars, long names -> short) -------
    init = {}
    for g in group_order:
        base = int(g[1:])
        found = {}
        for k, v in upvars.items():
            m = re.match(r'^mosfet_%d_\d+_([wlm])_' % base, k)
            if m:
                found[m.group(1).upper()] = v
        init[g] = {'W': round(found['W'], 4), 'L': round(found['L'], 4),
                   'M': max(1, int(round(found['M'])))}
    ib_init = upvars['current_0_bias']
    # upstream XC0 was a 10x10 MiM (unit ~0.215pF); the repo convention is 30x30
    # (unit ~1.82pF), so rescale the multiplier to keep roughly the same capacitance
    c0_unit_old = 10.0 * 10.0 * 2e-15 + (10.0 + 10.0) * 0.38e-15
    c0_init = max(1, int(round(upvars['m_c0'] * c0_unit_old / CAP_UNIT)))
    cl_init = max(1, int(round(upvars['m_cl'])))

    # ------- netlist -------
    nl = ['.subckt %s gnda vdda vinn vout vfb vinp Ib' % SUBCKT]
    for d in mos:
        g = d['group']
        l_expr = d['exprs']['l'].format(P='L_%s' % g)
        w_expr = d['exprs']['w'].format(P='W_%s' % g)
        m_expr = d['exprs']['m'].format(P='M_%s' % g)
        l_s = "l='%s'" % l_expr if '*' in l_expr else 'l=%s' % l_expr
        w_s = "w='%s'" % w_expr
        m_s = "m='%s'" % m_expr if '*' in m_expr else 'm=%s' % m_expr
        nl.append('XM%d %s %s %s' % (d['idx'], ' '.join(d['nets']), d['model'],
                                     ' '.join([l_s, w_s, m_s])))
    for r in res:
        nl.append('%s %s %s' % (r['name'].lower(), ' '.join(r['nets']), r['value']))
    for c in caps:
        nl.append('X%s %s sky130_fd_pr__cap_mim_m3_1 W=30 L=30 MF=M_%s m=M_%s'
                  % (c['name'], ' '.join(c['nets']), c['name'], c['name']))
    nl.append('.ends %s' % SUBCKT)
    netlist_file = '%s_netlist.txt' % PREFIX
    write_file(os.path.join(sim_dir, netlist_file), '\n'.join(nl) + '\n')

    # ------- vars -------
    vl = []
    for g in group_order:
        vl.append('.param W_%s=%s' % (g, fmt(init[g]['W'])))
        vl.append('.param L_%s=%s' % (g, fmt(init[g]['L'])))
        vl.append('.param M_%s=%d' % (g, init[g]['M']))
    vl.append('.param I_Ib=%s' % fmt(ib_init))
    vl.append('.param M_C0=%d' % c0_init)
    vl.append('.param M_CL=%d' % cl_init)
    write_file(os.path.join(sim_dir, '%s_vars.spice' % PREFIX), '\n'.join(vl) + '\n')

    # ------- testbenches -------
    write_file(os.path.join(sim_dir, '%s_ACDC.cir' % PREFIX),
               ACDC_TB.format(PREFIX=PREFIX, SUBCKT=SUBCKT, NETLIST=netlist_file))
    write_file(os.path.join(sim_dir, '%s_Tran.cir' % PREFIX),
               TRAN_TB.format(PREFIX=PREFIX, SUBCKT=SUBCKT, NETLIST=netlist_file))

    # ------- hierarchy + dev_params -------
    hierarchy = []
    for d in mos_sorted:
        hierarchy.append(['M%d' % d['idx'], 'x1.XM%d' % d['idx'],
                          d['model'].replace('sky130_fd_pr__', ''), 'm'])
    hierarchy.append(['Ib', '', 'Ib', 'i'])
    hierarchy.append(['C0', 'x1.XC0', 'cap_mim_m3_1', 'c'])
    hierarchy.append(['CL', 'XCL', 'cap_mim_m3_1', 'c'])
    from dev_params import DeviceParams
    dp = DeviceParams(tuple(tuple(x) for x in hierarchy)).gen_dev_params(
        file_name='%s_op' % PREFIX)
    write_file(os.path.join(sim_dir, '%s_dev_params.spice' % PREFIX), '\n'.join(dp) + '\n')

    # ------- graph -------
    node_names = ['M%d' % d['idx'] for d in mos_sorted] + ['Ib', 'VDD', 'GND', 'C0', 'CL']
    node_of = {n: i for i, n in enumerate(node_names)}
    edges = set()

    def add_edge(a, b):
        if a != b:
            edges.add((min(a, b), max(a, b)))

    sig = {}
    for d in mos_sorted:
        sig['M%d' % d['idx']] = set(n for n in d['nets'][:3] if n not in ('vdda', 'gnda'))
    names_m = ['M%d' % d['idx'] for d in mos_sorted]
    for i, a in enumerate(names_m):
        for b in names_m[i + 1:]:
            if sig[a] & sig[b]:
                add_edge(node_of[a], node_of[b])
    for d in mos_sorted:
        nm = 'M%d' % d['idx']
        dgs = d['nets'][:3]
        if 'vdda' in dgs:
            add_edge(node_of[nm], node_of['VDD'])
        if 'gnda' in dgs:
            add_edge(node_of[nm], node_of['GND'])
        if 'ib' in dgs:
            add_edge(node_of[nm], node_of['Ib'])
    add_edge(node_of['Ib'], node_of['GND'])
    # C0 terminals from the netlist; CL sits on the amplifier output in every testbench
    cap_nets = {'C0': caps[0]['nets'], 'CL': ['vout', 'gnda']}
    for cname, nets in cap_nets.items():
        for net in nets:
            if net == 'vdda':
                add_edge(node_of[cname], node_of['VDD'])
            elif net == 'gnda':
                add_edge(node_of[cname], node_of['GND'])
            elif net == 'ib':
                add_edge(node_of[cname], node_of['Ib'])
            else:
                for nm in names_m:
                    if net in sig[nm]:
                        add_edge(node_of[cname], node_of[nm])

    special = {node_of['Ib'], node_of['VDD'], node_of['GND']}
    edge_index, edge_type = [], []
    for a, b in sorted(edges):
        t = 1 if (a in special or b in special) else 0
        edge_index.extend([[a, b], [b, a]])
        edge_type.extend([t, t])

    # ------- yaml -------
    y = []
    y.append('# Auto-generated by tools/import_analoggym_ldo.py from AnalogGym Basic_LDO')
    y.append('# Evaluated by LdoEnv (LDO metrics mapped onto the repo MOO objective keys).')
    y.append('train_device: "cpu"')
    y.append('')
    y.append('base_dir: ../simulation_files/%s' % SHORT)
    y.append('file_paths:')
    y.append('  ACDC_cir_path: "%s_ACDC.cir"' % PREFIX)
    y.append('  Tran_cir_path: "%s_Tran.cir"' % PREFIX)
    y.append('  netlist_path: "%s"' % netlist_file)
    y.append('  vars_path: "%s_vars.spice"' % PREFIX)
    y.append('  dev_params_path: "%s_dev_params.spice"' % PREFIX)
    y.append('  op_mean_std_path: "%s_op_mean_std.json"' % PREFIX)
    y.append('')
    y.append('  op_results_path: "%s_op"' % PREFIX)
    y.append('  Vdrop_maxload_path: "%s_Vdrop_maxload"' % PREFIX)
    y.append('  Vdrop_minload_path: "%s_Vdrop_minload"' % PREFIX)
    y.append('  LNR_maxload_path: "%s_ACDC_LNR_maxload"' % PREFIX)
    y.append('  LNR_minload_path: "%s_ACDC_LNR_minload"' % PREFIX)
    y.append('  LR_Power_vos_path: "%s_ACDC_LR_Power_vos"' % PREFIX)
    y.append('  PSRR_dcgain_maxload_path: "%s_ACDC_PSRR_dcgain_maxload"' % PREFIX)
    y.append('  PSRR_dcgain_minload_path: "%s_ACDC_PSRR_dcgain_minload"' % PREFIX)
    y.append('  GBW_PM_maxload_path: "%s_ACDC_GBW_PM_maxload"' % PREFIX)
    y.append('  GBW_PM_minload_path: "%s_ACDC_GBW_PM_minload"' % PREFIX)
    y.append('  tran_meas_path: "%s_Tran_meas"' % PREFIX)
    y.append('')
    y.append('device:')
    for g in group_order:
        m_hi = 50 if init[g]['M'] <= 50 else max(500, 2 * init[g]['M'])
        y.append('  %s: {range: {W: [0.5, 10], L: [0.5, 5], M: [1, %d]}, '
                 'step: {W: 0.1, L: 0.1, M: 1}, init: {W: %s, L: %s, M: %d}, num: %d}'
                 % (g, m_hi, fmt(init[g]['W']), fmt(init[g]['L']), init[g]['M'],
                    len(groups[g])))
    y.append('  Ib: {range: {I: [1e-6, %s]}, init: {I: %s}, num: 1}'
             % (fmt(max(30e-6, 2 * ib_init)), fmt(ib_init)))
    y.append('  C0: {range: {M: [1, %d]}, step: {M: 1}, init: {M: %d}, num: 1}'
             % (max(30, 2 * c0_init), c0_init))
    y.append('  CL: {range: {M: [50, %d]}, step: {M: 1}, init: {M: %d}, num: 1}'
             % (max(600, 2 * cl_init), cl_init))
    y.append('action_dim: %d' % (3 * len(group_order) + 1 + 2))
    y.append('')
    y.append('# LDO performance targets. Constraints: phase_margin/dcgain/PSRR/LNR/LR/vos.')
    y.append('# Objectives (mapped onto the repo MOO keys by LdoEnv):')
    y.append('#   FOML  <- transient ripple undershoot+overshoot [V] (smaller is better)')
    y.append('#   FOMS  <- GBW*CL/Power [MHz*pF/mW] (larger is better)')
    y.append('performance:')
    y.append('  phase_margin: {target: 60} #unit:deg (worst case over min/max load)')
    y.append('  dcgain: {target: 60, objective: max, baseline: 40} #unit:dB loop gain')
    y.append('  PSRR: {target: -40, objective: min, baseline: -20} #unit:dB at DC')
    y.append('  LNR: {target: 0.05, objective: min, baseline: 0.5} #line regulation, /V')
    y.append('  LR: {target: 0.05, objective: min, baseline: 0.5} #load regulation, /A-range')
    y.append('  vos: {target: 0.02, objective: min, baseline: 0.1} #unit:V, |vout-1.6|')
    y.append('  vdrop: {target: 0.3, objective: min, baseline: 0.6} #unit:V dropout')
    y.append('  undershoot: {target: 0.15, objective: min, baseline: 0.5} #unit:V')
    y.append('  overshoot: {target: 0.15, objective: min, baseline: 0.5} #unit:V')
    y.append('  FOML: {target: 0.3, objective: min, baseline: 1.0} #V ripple (under+over)')
    y.append('  FOMS: {target: 100, objective: max, baseline: 20} #MHz*pF/mW')
    y.append('  Active_Area: {target: 400, objective: min, baseline: 800}')
    y.append('  Power: {target: 10, objective: min, baseline: 20} #unit:mW quiescent+minload')
    y.append('  GBW: {target: 1e6, objective: max, baseline: 0.5e6} #unit:Hz')
    y.append('')
    y.append('# nominal load capacitance in pF at the CL init point (M_CL * 1.8228pF)')
    y.append('PARAM_CLOAD: %s' % fmt(round(cl_init * CAP_UNIT / 1e-12, 1)))
    y.append('')
    y.append('ckt_hierarchy:')
    for h in hierarchy:
        y.append('  - [%s, %s, %s, %s]' % (h[0], ('"%s"' % h[1]) if h[1] == '' else h[1],
                                           h[2], h[3]))
    y.append('')
    y.append('graph:')
    for i in range(0, len(node_names), 6):
        y.append('# ' + ' , '.join('node %d : %s' % (i + j, n)
                                   for j, n in enumerate(node_names[i:i + 6])))
    y.append('  num_relations: 2')
    y.append('  num_nodes: %d' % len(node_names))
    y.append('  num_node_features: 12')
    y.append('  edge_index: [')
    for i in range(0, len(edge_index), 8):
        chunk = ', '.join('[%d,%d]' % (a, b) for a, b in edge_index[i:i + 8])
        y.append('    %s%s' % (chunk, ',' if i + 8 < len(edge_index) else ''))
    y.append('  ]')
    y.append('  edge_type: [')
    for i in range(0, len(edge_type), 16):
        chunk = ','.join(str(t) for t in edge_type[i:i + 16])
        y.append('    %s%s' % (chunk, ',' if i + 16 < len(edge_type) else ''))
    y.append('  ]')
    y.append('  observation_matrix:')
    for d in mos_sorted:
        y.append('      M%d: [0, 0, 0, 0, 0, "id", "gm", "gds", "vth", "vdsat", "vds", "vgs"]'
                 % d['idx'])
    y.append('      Ib: [0, 0, "current", 0, 0, 0, 0, 0, 0, 0, 0, 0]')
    y.append("      VDD: ['v', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]")
    y.append('      GND: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]')
    y.append('      C0: [0, 0, 0, "c", 0, 0, 0, 0, 0, 0, 0, 0]')
    y.append('      CL: [0, 0, 0, 0, "c", 0, 0, 0, 0, 0, 0, 0]')
    y.append('')
    write_file(cfg_path, '\n'.join(y) + '\n')

    print('  [ok] %s: %d transistors in %d groups, action_dim=%d, %d graph nodes'
          % (SHORT, len(mos), len(group_order), 3 * len(group_order) + 3,
             len(node_names)))


ACDC_TB = """Test LDO ACDC

*.OPTIONS RELTOL=.0001
***************************************
* Step 1: Replace circuit netlist here.
***************************************
.include ./{NETLIST}

.param mc_mm_switch=0
.param mc_pr_switch=0
*.temp
.include ../sky130_pdk/libs.tech/ngspice/corners/tt.spice
.include ../sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical.spice
.include ../sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical__lin.spice
.include ../sky130_pdk/libs.tech/ngspice/corners/tt/specialized_cells.spice

***************************************
* Step 2: Replace circuit param.  here.
***************************************
.PARAM supply_voltage = 1.8
.PARAM Vref = 0.4
.PARAM PARAM_ILOAD =55m
.include ./{PREFIX}_vars.spice

V1 vdd 0 'supply_voltage'
V2 vss 0 0

Vindc vref_in 0 'Vref'
Vin signal_in 0 dc 'Vref' ac 1 sin('Vref' 100m 500)

* XLDO gnda vdda vinn vout vfb vinp Ib
*        |  |     |     |   |    |   |
*        |  |     |     |   |    |   bias current
*        |  |     |     |   |    Non-inverting input
*        |  |     |     |   Feedback voltage
*        |  |     |     Output
*        |  |     Inverting Input
*        |  Positive Supply
*        Negative Supply

*    ADM TB (loop broken via Lfb/Cfb for loop-gain measurement)
x1 vss vdd vref_in vout1 vfb1 vinp1 Ib {SUBCKT}
Ib Ib 0 DC='I_Ib'
Lfb vinp1 vfb1 1T
Cfb vinp1 signal_in 1T
XCL vout1 0 sky130_fd_pr__cap_mim_m3_1 W=30 L=30 MF=M_CL m=M_CL
Iload1 vout1 0 'PARAM_ILOAD'

* PSRR   TB
VVDDApsrr vddpsrr 0 'supply_voltage'  AC=1
Ib2 Ib2 0 DC='I_Ib'
XCL2 ppsr1 0 sky130_fd_pr__cap_mim_m3_1 W=30 L=30 MF=M_CL m=M_CL
x2 vss vddpsrr vref_in ppsr1 vfb2 vfb2 Ib2 {SUBCKT}
Iload2 ppsr1 0 'PARAM_ILOAD'

* DC ALL  TB
VVDDdc VDDdc 0 'supply_voltage'
Ib3 Ib3 0 DC='I_Ib'
XCL3 vout6 0 sky130_fd_pr__cap_mim_m3_1 W=30 L=30 MF=M_CL m=M_CL
x3 vss vdddc vref_in vout6 vfb3 vfb3 Ib3 {SUBCKT}
Iload3 vout6 0 'PARAM_ILOAD'

.control
save all
.options savecurrents
set filetype=ascii
set units=degrees

* DC sweep at maxload
alter Iload3 dc=55m
dc VVDDdc 1 3 0.01
plot  v(vout6)
wrdata {PREFIX}_Vdrop_maxload v(vout6)

* DC sweep at minload
alter Iload3 dc=5m
dc VVDDdc 1 3 0.01
plot  v(vout6)
wrdata {PREFIX}_Vdrop_minload v(vout6)

* LNR at maxload
alter Iload3 dc=55m
dc VVDDdc 1.62 1.98 0.01
meas dc avgval1 AVG V(vout6) from=1.62 to=1.98
meas dc ppavl1  PP V(vout6) from=1.62 to=1.98
let LNR1 = ppavl1/avgval1/0.36
print LNR1
wrdata {PREFIX}_ACDC_LNR_maxload LNR1

* LNR at minload
alter Iload3 dc=5m
dc VVDDdc 1.62 1.98 0.01
meas dc avgval2 AVG V(vout6) from=1.62 to=1.98
meas dc ppavl2  PP V(vout6) from=1.62 to=1.98
let LNR2 = ppavl2/avgval2/0.36
print LNR2
wrdata {PREFIX}_ACDC_LNR_minload LNR2

dc Iload3 5m 55m 0.1m
* LR meas
meas dc avgval AVG V(vout6) from=5m to=55m
meas dc ppavl  PP V(vout6) from=5m to=55m
let LR = ppavl/avgval/50m
print LR
* Power meas at maxload
meas dc Ivdd1 FIND I(VVDDDC) AT=55m
let Power1 = -1*Ivdd1*1.8
print Power1
* Power meas at minload
meas dc Ivdd2 FIND I(VVDDDC) AT=5m
let Power2 = -1*Ivdd2*1.8
print Power2
*   Vos.meas at maxload
meas dc vout_x FIND V(vout6) AT=55m
let vos1 = vout_x-4*0.4
print vos1
*   Vos.meas at minload
meas dc vout_y FIND V(vout6) AT=5m
let vos2 = vout_y-4*0.4
print vos2
wrdata {PREFIX}_ACDC_LR_Power_vos LR Power1 Power2 vos1 vos2

* Loop test at maxload
alter Iload1 dc=55m
ac dec 10 0.1 1G
meas ac DCPSRp1 find vdb(ppsr1) at = 0.1
meas ac dcgain1 find vdb(vout1) at = 0.1
meas ac gain_bandwidth_product1 when vdb(vout1)=0
meas ac phase_margin1 find vp(vout1) when vdb(vout1)=0
wrdata {PREFIX}_ACDC_PSRR_dcgain_maxload DCPSRp1 dcgain1
wrdata {PREFIX}_ACDC_GBW_PM_maxload gain_bandwidth_product1 phase_margin1

* Loop test at minload
alter Iload1 dc=5m
ac dec 10 0.1 1G
meas ac DCPSRp2 find vdb(ppsr1) at = 0.1
meas ac dcgain2 find vdb(vout1) at = 0.1
meas ac gain_bandwidth_product2 when vdb(vout1)=0
meas ac phase_margin2 find vp(vout1) when vdb(vout1)=0
wrdata {PREFIX}_ACDC_PSRR_dcgain_minload DCPSRp2 dcgain2
wrdata {PREFIX}_ACDC_GBW_PM_minload gain_bandwidth_product2 phase_margin2

* OP
op
.include ./{PREFIX}_dev_params.spice
.endc

.end
"""

TRAN_TB = """Test LDO Tran

*.OPTIONS RELTOL=.0001
***************************************
* Step 1: Replace circuit netlist here.
***************************************
.include ./{NETLIST}

.param mc_mm_switch=0
.param mc_pr_switch=0
*.temp
.include ../sky130_pdk/libs.tech/ngspice/corners/tt.spice
.include ../sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical.spice
.include ../sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical__lin.spice
.include ../sky130_pdk/libs.tech/ngspice/corners/tt/specialized_cells.spice

***************************************
* Step 2: Replace circuit param.  here.
***************************************
.PARAM supply_voltage = 1.8
.PARAM Vref = 0.4
.PARAM PARAM_ILOAD = 55m
.PARAM val0 = 5m
.PARAM val1 = 55m
.PARAM GBW_ideal = 5e4
.PARAM STEP_TIME = '10/GBW_ideal'
.include ./{PREFIX}_vars.spice

V1 vdd 0 'supply_voltage'
V2 vss 0 0

Vindc vref_in 0 'Vref'

* XLDO gnda vdda vinn vout vfb vinp Ib

*   Tran TB (load current step 5m -> 55m -> 5m)
x1 vss vdd vref_in vout1 vfb1 vfb1 Ib {SUBCKT}
Ib Ib 0 DC='I_Ib'
XCL vout1 0 sky130_fd_pr__cap_mim_m3_1 W=30 L=30 MF=M_CL m=M_CL
Iload1 vout1 0 pulse('val0' 'val1' 1u 1p 1p '0.25*STEP_TIME' 1)

.control
save all
.options savecurrents
set filetype=ascii
set units=degrees

tran 10n 100u
meas tran v_min MIN v(vout1) from=0 to= 50u
meas tran v_max MAX v(vout1) from=50u to= 100u
let v_undershoot = 4*0.4 - v_min
let v_overshoot = v_max - 4*0.4
print v_undershoot
print v_overshoot
wrdata {PREFIX}_Tran_meas v_undershoot v_overshoot

* OP
op
.include ./{PREFIX}_dev_params.spice
.endc

.end
"""


if __name__ == '__main__':
    main()
