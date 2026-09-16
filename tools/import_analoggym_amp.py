#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Import an AnalogGym amplifier (Pin_3 family) into this repo's circuit-instance format.

Given the upstream AnalogGym repo checkout, this script converts one amplifier:

  AnalogGym/Amplifier/spice_netlist/<Upstream_Name>     (sky130 subckt, long param names)
  AnalogGym/Amplifier/design_variables/<Upstream_Name>  (.PARAM block with initial sizing)

into a complete, runnable instance under this repo:

  simulation_files/amp_<short>/AMP_<SHORT>_Pin_3_HSPICE_130.txt   renamed netlist (W_Mx/L_Mx/M_Mx, Ib pin)
  simulation_files/amp_<short>/AMP_<SHORT>_ACDC.cir               ACDC testbench (DC/AC/PSRR/CMRR/OP)
  simulation_files/amp_<short>/AMP_<SHORT>_Tran.cir               transient testbench
  simulation_files/amp_<short>/AMP_<SHORT>_vars.spice             initial design variables
  simulation_files/amp_<short>/AMP_<SHORT>_dev_params.spice       .OP parameter extraction (via DeviceParams)
  circuit_configs/amp_<short>.yaml                                device/performance/hierarchy/graph config

Conversion conventions (matching the 5 hand-converted instances already in the repo):
  * subckt pins become "gnda vdda vinn vinp vout Ib": the internal bias current source
    is removed and its bias net is exposed as the Ib pin (driven by the testbench).
  * MOSFET_<a>_<b>_{W,L,M}_<role> parameter groups are renamed to {W,L,M}_M<a>.
  * ideal capacitors become sky130 MiM caps (W=30 L=30, M_Ck parallel units of ~1.82pF).
  * ideal resistors become sky130_fd_pr__res_high_po_0p35 (W=0.35), with the length chosen
    so the upstream resistance is reached at multiplier M_Rk=4 (R = Rsheet*L/W/M).

Usage:
  python tools/import_analoggym_amp.py --upstream <analoggym_root> --name Fan_SMC_Pin_3 [--force]
  python tools/import_analoggym_amp.py --upstream <analoggym_root> --all
"""
import argparse
import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Fixed passive-device geometry conventions (must match AmpEnv._get_obs / _calculate_area)
CAP_W = 30.0
CAP_L = 30.0
CAP_UNIT = CAP_W * CAP_L * 2e-15 + (CAP_W + CAP_L) * 0.38e-15   # ~1.8228 pF per unit
RES_RSHEET = 1112.4
RES_W = 0.35
RES_INIT_MULT = 4          # choose resistor length so upstream R is hit at M_R = 4
SKY130_MIN_W = 0.42        # smallest fet width covered by the sky130 tt bins (um)
SKY130_MIN_L = 0.15

# Mapping upstream circuit name -> (short dir suffix, subckt name used in the converted netlist)
KNOWN_CIRCUITS = {
    'Leung_NMCNR_Pin_3': 'nmcnr',
    'Leung_NMCF_Pin_3': 'nmcf',
    'Leung_DFCFC1_Pin_3': 'dfcfc1',
    'Leung_DFCFC2_Pin_3': 'dfcfc2',
    'Ramos_PFC_Pin_3': 'pfc',
    'Alfio_RAFFC_Pin_3': 'raffc',
    'Fan_SMC_Pin_3': 'smc',
    'HoiLee_AFFC_Pin_3': 'affc',
    'Peng_ACBC_Pin_3': 'acbc',
    'Peng_IAC_Pin_3': 'iac',
    'Peng_TCFC_Pin_3': 'tcfc',
    'Qu2017_AZC_Pin_3': 'azc',
    'Sau_CFCC_Pin_3': 'cfcc',
    'Song_DACFC_Pin_3': 'dacfc',
    'Tan_CLIA_Pin_3': 'clia',
    'Yan_AZ_Pin_3': 'az',
}

SI_SUFFIX = {'t': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3, 'm': 1e-3,
             'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15}


def parse_num(text):
    """Parse a spice number with optional SI suffix (20u, 35K, 810f, 0.35)."""
    t = text.strip().strip("'").strip('"').lower()
    m = re.match(r'^([0-9+\-.e]+)\s*(meg|t|g|k|m|u|n|p|f)?', t)
    if not m:
        raise ValueError('cannot parse number: %r' % text)
    val = float(m.group(1))
    if m.group(2):
        val *= SI_SUFFIX[m.group(2)]
    return val


def read_joined_lines(path):
    """Read a spice file, join '+' continuation lines, drop comments/blank lines."""
    with open(path, 'r') as f:
        raw = [ln.rstrip('\r\n') for ln in f]
    joined = []
    for ln in raw:
        s = ln.strip()
        if not s or s.startswith('*'):
            continue
        if s.startswith('+') and joined:
            joined[-1] += ' ' + s[1:].strip()
        else:
            joined.append(s)
    return joined


MOS_PARAM_RE = re.compile(r'MOSFET_(\d+)_(\d+)_([WLM])_([A-Za-z0-9]+)_([PN])MOS', re.IGNORECASE)


class UpstreamAmp(object):
    def __init__(self, netlist_path, design_vars_path):
        self.mos = []          # dicts: name, idx, nets(d,g,s,b), model, group, m_expr_template
        self.caps = []         # dicts: name, idx, nets, val_param
        self.res = []          # dicts: name, idx, nets, val_param
        self.bias = None       # dict: nets, val_param
        self.subckt_pins = None
        self.design_vars = {}  # upstream param name (upper) -> float
        self._parse_netlist(netlist_path)
        self._parse_design_vars(design_vars_path)

    def _parse_netlist(self, path):
        lines = read_joined_lines(path)
        for ln in lines:
            low = ln.lower()
            if low.startswith('.subckt'):
                toks = ln.split()
                self.subckt_pins = [t.lower() for t in toks[2:]]
                if self.subckt_pins != ['gnda', 'vdda', 'vinn', 'vinp', 'vout']:
                    raise ValueError('unexpected subckt pins: %s' % self.subckt_pins)
            elif low.startswith('.ends'):
                break
            elif low.startswith('xm'):
                self._parse_mos(ln)
            elif re.match(r'^c\d+', low):
                toks = ln.split()
                self.caps.append({'name': toks[0].upper(),
                                  'idx': int(re.match(r'^c(\d+)', low).group(1)),
                                  'nets': [toks[1].lower(), toks[2].lower()],
                                  'val_param': toks[3].strip("'").upper()})
            elif re.match(r'^r\d+', low):
                toks = ln.split()
                self.res.append({'name': toks[0].upper(),
                                 'idx': int(re.match(r'^r(\d+)', low).group(1)),
                                 'nets': [toks[1].lower(), toks[2].lower()],
                                 'val_param': toks[3].strip("'").upper()})
            elif re.match(r'^i\d+', low):
                toks = ln.split()
                if self.bias is not None:
                    raise ValueError('more than one bias current source found')
                self.bias = {'nets': [toks[1].lower(), toks[2].lower()],
                             'val_param': toks[3].strip("'").upper()}
            else:
                raise ValueError('unhandled netlist line: %s' % ln)
        if self.bias is None:
            raise ValueError('no internal bias current source found')
        # the bias net is the terminal that is not a supply
        nets = [n for n in self.bias['nets'] if n not in ('gnda', 'vdda')]
        if len(nets) != 1:
            raise ValueError('cannot identify bias net from %s' % self.bias['nets'])
        self.bias_net = nets[0]
        if self.bias['nets'].index(self.bias_net) != 0 or self.bias['nets'][1] != 'gnda':
            raise ValueError('unexpected bias source orientation %s (expected <net> gnda)'
                             % self.bias['nets'])

    def _parse_mos(self, ln):
        toks = ln.split()
        name = toks[0].upper()          # XM11
        idx = int(re.match(r'^xm(\d+)', toks[0].lower()).group(1))
        nets = [t.lower() for t in toks[1:5]]
        model = toks[5]
        if not model.startswith('sky130_fd_pr__'):
            raise ValueError('unexpected mos model: %s' % model)
        params = {}
        for t in toks[6:]:
            k, v = t.split('=', 1)
            params[k.lower()] = v.strip("'")
        group = None
        exprs = {}
        for key in ('l', 'w', 'm'):
            expr = params[key]
            m = MOS_PARAM_RE.search(expr)
            if not m:
                raise ValueError('no MOSFET_* param in %s=%s of %s' % (key, expr, name))
            g = 'M%d' % int(m.group(1))
            if group is None:
                group = g
            elif group != g:
                raise ValueError('%s references multiple groups (%s vs %s)' % (name, group, g))
            if m.group(3).upper() != key.upper():
                raise ValueError('%s: %s= uses a _%s_ param' % (name, key, m.group(3)))
            # keep any numeric multiplier around the group parameter (e.g. '4*<p>' or '<p>*8')
            exprs[key] = expr.replace(m.group(0), '{P}')
        self.mos.append({'name': name, 'idx': idx, 'nets': nets, 'model': model,
                         'group': group, 'exprs': exprs})

    def _parse_design_vars(self, path):
        text = ' '.join(read_joined_lines(path))
        text = re.sub(r'^\.param\s*', '', text, flags=re.IGNORECASE)
        for tok in text.split():
            if '=' not in tok:
                continue
            k, v = tok.split('=', 1)
            self.design_vars[k.upper()] = parse_num(v)


def build_groups(amp):
    """Collect transistor groups in ascending base-index order."""
    groups = {}
    for d in amp.mos:
        groups.setdefault(d['group'], []).append(d)
    order = sorted(groups.keys(), key=lambda g: int(g[1:]))
    return groups, order


def group_init(amp, group, kind):
    """Initial W/L/M for a transistor group from the upstream design variables."""
    pat = re.compile(r'^MOSFET_%s_\d+_%s_' % (group[1:], kind), re.IGNORECASE)
    for k, v in amp.design_vars.items():
        if pat.match(k):
            return v
    raise ValueError('no %s design variable found for group %s' % (kind, group))


def convert(amp, upstream_name, short, force=False):
    prefix = 'AMP_%s' % short.upper()
    subckt_name = upstream_name  # keep the upstream (capitalized) subckt name
    sim_dir = os.path.join(REPO_ROOT, 'simulation_files', 'amp_%s' % short)
    cfg_path = os.path.join(REPO_ROOT, 'circuit_configs', 'amp_%s.yaml' % short)
    if not force and (os.path.exists(sim_dir) or os.path.exists(cfg_path)):
        print('  [skip] amp_%s already exists (use --force to overwrite)' % short)
        return None, None
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    groups, group_order = build_groups(amp)
    mos_sorted = sorted(amp.mos, key=lambda d: d['idx'])
    caps_sorted = sorted(amp.caps, key=lambda d: d['idx'])
    res_sorted = sorted(amp.res, key=lambda d: d['idx'])

    # ---------------- initial values ----------------
    init = {}
    warnings = []
    for g in group_order:
        w = group_init(amp, g, 'W')
        l = group_init(amp, g, 'L')
        m = int(round(group_init(amp, g, 'M')))
        if w < SKY130_MIN_W:
            warnings.append('group %s: W init %.3g below sky130 min bin, clamped to %.2f'
                            % (g, w, SKY130_MIN_W))
            w = SKY130_MIN_W
        if l < SKY130_MIN_L:
            warnings.append('group %s: L init %.3g below sky130 min, clamped to %.2f'
                            % (g, l, SKY130_MIN_L))
            l = SKY130_MIN_L
        init[g] = {'W': w, 'L': l, 'M': max(1, m)}
    ib_init = amp.design_vars[amp.bias['val_param']]
    cap_init = {}
    for c in caps_sorted:
        val = amp.design_vars[c['val_param']]
        cap_init[c['name']] = max(1, int(round(val / CAP_UNIT)))
    res_len = {}
    res_init = {}
    for r in res_sorted:
        val = amp.design_vars[r['val_param']]
        length = val * RES_W * RES_INIT_MULT / RES_RSHEET
        res_len[r['name']] = max(0.5, round(length, 2))
        res_init[r['name']] = RES_INIT_MULT
    cload = amp.design_vars.get('CLOAD', 100e-12)

    # ---------------- netlist ----------------
    def net_out(n):
        return 'Ib' if n == amp.bias_net else n

    nl = ['.subckt %s gnda vdda vinn vinp vout Ib' % subckt_name]
    for d in amp.mos:  # keep upstream device order
        g = d['group']
        l_expr = d['exprs']['l'].format(P='L_%s' % g)
        w_expr = d['exprs']['w'].format(P='W_%s' % g)
        m_expr = d['exprs']['m'].format(P='M_%s' % g)
        l_s = "l='%s'" % l_expr if '*' in l_expr else 'l=%s' % l_expr
        w_s = "w='%s'" % w_expr
        m_s = "m='%s'" % m_expr if '*' in m_expr else 'm=%s' % m_expr
        nl.append('%s %s %s %s' % (d['name'], ' '.join(net_out(n) for n in d['nets']),
                                   d['model'], ' '.join([l_s, w_s, m_s])))
    for r in res_sorted:
        nl.append('X%s %s gnda sky130_fd_pr__res_high_po_0p35 L=%s mult=M_%s m=M_%s'
                  % (r['name'], ' '.join(net_out(n) for n in r['nets']),
                     res_len[r['name']], r['name'], r['name']))
    for c in caps_sorted:
        nl.append('X%s %s sky130_fd_pr__cap_mim_m3_1 W=%d L=%d MF=M_%s m=M_%s'
                  % (c['name'], ' '.join(net_out(n) for n in c['nets']),
                     int(CAP_W), int(CAP_L), c['name'], c['name']))
    nl.append('.ends %s' % subckt_name)
    netlist_file = '%s_Pin_3_HSPICE_130.txt' % prefix
    write_file(os.path.join(sim_dir, netlist_file), '\n'.join(nl) + '\n')

    # ---------------- vars.spice ----------------
    vl = []
    for g in group_order:
        vl.append('.param W_%s=%s' % (g, fmt(init[g]['W'])))
        vl.append('.param L_%s=%s' % (g, fmt(init[g]['L'])))
        vl.append('.param M_%s=%d' % (g, init[g]['M']))
    vl.append('.param I_Ib=%s' % fmt(ib_init))
    for c in caps_sorted:
        vl.append('.param M_%s=%d' % (c['name'], cap_init[c['name']]))
    for r in res_sorted:
        vl.append('.param M_%s=%d' % (r['name'], res_init[r['name']]))
    write_file(os.path.join(sim_dir, '%s_vars.spice' % prefix), '\n'.join(vl) + '\n')

    # ---------------- testbenches ----------------
    cload_str = si_cap(cload)
    tb = ACDC_TEMPLATE.format(PREFIX=prefix, SUBCKT=subckt_name, NETLIST=netlist_file,
                              CLOAD=cload_str)
    write_file(os.path.join(sim_dir, '%s_ACDC.cir' % prefix), tb)
    tb = TRAN_TEMPLATE.format(PREFIX=prefix, SUBCKT=subckt_name, NETLIST=netlist_file,
                              CLOAD=cload_str)
    write_file(os.path.join(sim_dir, '%s_Tran.cir' % prefix), tb)

    # ---------------- ckt_hierarchy ----------------
    hierarchy = []
    for d in mos_sorted:
        hierarchy.append([('M%d' % d['idx']), 'x1.X%s' % ('M%d' % d['idx']),
                          d['model'].replace('sky130_fd_pr__', ''), 'm'])
    hierarchy.append(['Ib', '', 'Ib', 'i'])
    for c in caps_sorted:
        hierarchy.append([c['name'], 'x1.X%s' % c['name'], 'cap_mim_m3_1', 'c'])

    # ---------------- dev_params.spice ----------------
    from dev_params import DeviceParams
    dp_lines = DeviceParams(tuple(tuple(x) for x in hierarchy)).gen_dev_params(
        file_name='%s_op' % prefix)
    write_file(os.path.join(sim_dir, '%s_dev_params.spice' % prefix),
               '\n'.join(dp_lines) + '\n')

    # ---------------- graph ----------------
    node_names = ['M%d' % d['idx'] for d in mos_sorted] + ['Ib', 'VDD', 'GND'] \
        + [c['name'] for c in caps_sorted] + [r['name'] for r in res_sorted]
    node_of = {n: i for i, n in enumerate(node_names)}
    edges = set()

    def add_edge(a, b):
        if a != b:
            edges.add((min(a, b), max(a, b)))

    # device-device edges: shared d/g/s net (supplies excluded, bias net included)
    mos_sig_nets = {}
    for d in mos_sorted:
        sig = set(n for n in d['nets'][:3] if n not in ('vdda', 'gnda'))
        mos_sig_nets['M%d' % d['idx']] = sig
    names_m = ['M%d' % d['idx'] for d in mos_sorted]
    for i, a in enumerate(names_m):
        for b in names_m[i + 1:]:
            if mos_sig_nets[a] & mos_sig_nets[b]:
                add_edge(node_of[a], node_of[b])
    # device-supply / device-bias edges
    for d in mos_sorted:
        nm = 'M%d' % d['idx']
        dgs = d['nets'][:3]
        if 'vdda' in dgs:
            add_edge(node_of[nm], node_of['VDD'])
        if 'gnda' in dgs:
            add_edge(node_of[nm], node_of['GND'])
        if amp.bias_net in dgs:
            add_edge(node_of[nm], node_of['Ib'])
    # bias source ties the bias net to ground in the testbench
    add_edge(node_of['Ib'], node_of['GND'])
    # passive devices
    for p in caps_sorted + res_sorted:
        pn = node_of[p['name']]
        for net in p['nets']:
            if net == 'vdda':
                add_edge(pn, node_of['VDD'])
            elif net == 'gnda':
                add_edge(pn, node_of['GND'])
            elif net == amp.bias_net:
                add_edge(pn, node_of['Ib'])
            else:
                for nm in names_m:
                    if net in mos_sig_nets[nm]:
                        add_edge(pn, node_of[nm])

    special = {node_of['Ib'], node_of['VDD'], node_of['GND']}
    edge_index, edge_type = [], []
    for a, b in sorted(edges):
        t = 1 if (a in special or b in special) else 0
        edge_index.extend([[a, b], [b, a]])
        edge_type.extend([t, t])

    # ---------------- observation matrix ----------------
    obs = {}
    for d in mos_sorted:
        obs['M%d' % d['idx']] = [0, 0, 0, 0, 0, 'id', 'gm', 'gds', 'vth', 'vdsat', 'vds', 'vgs']
    obs['Ib'] = [0, 0, 'current'] + [0] * 9
    obs['VDD'] = ['v'] + [0] * 11
    obs['GND'] = [0] * 12
    col = 3
    for c in caps_sorted:
        row = [0] * 12
        row[col] = 'c'
        obs[c['name']] = row
        col += 1
    for r in res_sorted:
        row = [0] * 12
        row[col] = 'r'
        obs[r['name']] = row
        col += 1

    # ---------------- config yaml ----------------
    action_dim = 3 * len(group_order) + 1 + len(caps_sorted) + len(res_sorted)
    y = []
    y.append('# Auto-generated by tools/import_analoggym_amp.py from AnalogGym %s' % upstream_name)
    y.append('train_device: "cpu"')
    y.append('')
    y.append('base_dir: ../simulation_files/amp_%s' % short)
    y.append('file_paths:')
    y.append('  ACDC_cir_path: "%s_ACDC.cir"' % prefix)
    y.append('  Tran_cir_path: "%s_Tran.cir"' % prefix)
    y.append('  netlist_path: "%s"' % netlist_file)
    y.append('  vars_path: "%s_vars.spice"' % prefix)
    y.append('  dev_params_path: "%s_dev_params.spice"' % prefix)
    y.append('  op_mean_std_path: "%s_op_mean_std.json"' % prefix)
    y.append('')
    y.append('  dc_results_path: "%s_ACDC_DC"' % prefix)
    y.append('  ac_results_path: "%s_ACDC_AC"' % prefix)
    y.append('  op_results_path: "%s_op"' % prefix)
    y.append('  GBW_PM_path: "%s_ACDC_GBW_PM"' % prefix)
    y.append('  tran_results_path: "%s_Tran"' % prefix)
    y.append('  tran_dat_path: "%s_tran.dat"' % prefix)
    y.append('')
    y.append('device:')
    for g in group_order:
        m_hi = 50 if init[g]['M'] <= 50 else 500
        w_lo = SKY130_MIN_W if init[g]['W'] < 0.5 else 0.5
        l_lo = min(0.5, init[g]['L']) if init[g]['L'] < 0.5 else 0.5
        y.append('  %s: {range: {W: [%s, 10], L: [%s, 5], M: [1, %d]}, '
                 'step: {W: 0.1, L: 0.1, M: 1}, '
                 'init: {W: %s, L: %s, M: %d}, num: %d}'
                 % (g, fmt(w_lo), fmt(l_lo), m_hi,
                    fmt(init[g]['W']), fmt(init[g]['L']), init[g]['M'], len(groups[g])))
    ib_hi = max(30e-6, 2 * ib_init)
    y.append('  Ib: {range: {I: [1e-6, %s]}, init: {I: %s}, num: 1}' % (fmt(ib_hi), fmt(ib_init)))
    for c in caps_sorted:
        m_hi = max(30, 2 * cap_init[c['name']])
        y.append('  %s: {range: {M: [1, %d]}, step: {M: 1}, init: {M: %d}, num: 1}'
                 % (c['name'], m_hi, cap_init[c['name']]))
    for r in res_sorted:
        y.append('  %s: {range: {M: [1, 20]}, step: {M: 1}, init: {M: %d}, num: 1}'
                 % (r['name'], res_init[r['name']]))
    y.append('action_dim: %d' % action_dim)
    y.append('')
    y.append('# Performance targets: generic defaults copied from the AMP family; tune per circuit.')
    y.append('performance:')
    y.append('  phase_margin: {target: 60} #unit:deg')
    y.append('  dcgain: {target: 100, objective: max, baseline: 80} #unit:dB')
    y.append('  PSRP: {target: -80, objective: min, baseline: -60} #unit:dB')
    y.append('  PSRN: {target: -80, objective: min, baseline: -60} #unit:dB')
    y.append('  PSRR: {target: -80, objective: min, baseline: -60} #unit:dB')
    y.append('  cmrrdc: {target: -80, objective: min, baseline: -60} #unit:dB')
    y.append('  vos: {target: 0.06e-3, objective: min, baseline: 0.1e-3} #unit:V')
    y.append('  TC: {target: 10e-6, objective: min, baseline: 50e-6}')
    y.append('  settlingTime: {target: 1e-6, objective: min, baseline: 5e-6} #unit:s')
    y.append('  FOML: {target: 100, objective: max, baseline: 30} #V/us*pF/mW')
    y.append('  FOMS: {target: 400, objective: max, baseline: 100} #MHz*pF/mW')
    y.append('  Active_Area: {target: 100, objective: min, baseline: 200} #unit:mm^2')
    y.append('  Power: {target: 0.3, objective: min, baseline: 0.5} #unit:mW')
    y.append('  GBW: {target: 1e6, objective: max, baseline: 0.5e6} #unit:Hz')
    y.append('  sr: {target: 0.4, objective: max, baseline: 0.2} #unit:V/us')
    y.append('')
    y.append('# load capacitance in pF (must match PARAM_CLOAD in the testbenches)')
    y.append('PARAM_CLOAD: %s' % fmt(cload / 1e-12))
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
        tail = ',' if i + 8 < len(edge_index) else ''
        y.append('    %s%s' % (chunk, tail))
    y.append('  ]')
    y.append('  edge_type: [')
    for i in range(0, len(edge_type), 16):
        chunk = ','.join(str(t) for t in edge_type[i:i + 16])
        tail = ',' if i + 16 < len(edge_type) else ''
        y.append('    %s%s' % (chunk, tail))
    y.append('  ]')
    y.append('  observation_matrix:')
    for k, row in obs.items():
        cells = ', '.join(('"%s"' % v) if isinstance(v, str) else str(v) for v in row)
        y.append('      %s: [%s]' % (k, cells))
    y.append('')
    write_file(cfg_path, '\n'.join(y) + '\n')

    for w in warnings:
        print('  [warn] %s' % w)
    print('  [ok] amp_%s: %d transistors in %d groups, %d caps, %d res, action_dim=%d'
          % (short, len(amp.mos), len(group_order), len(caps_sorted), len(res_sorted),
             action_dim))
    return sim_dir, cfg_path


def fmt(x):
    if isinstance(x, int):
        return str(x)
    if x == int(x) and abs(x) < 1e6:
        return str(int(x))
    return ('%.6g' % x)


def si_cap(farads):
    """Format a capacitance for the testbench .PARAM (e.g. 100p, 1500p, 15n)."""
    pf = farads / 1e-12
    if pf >= 1000 and (pf / 1000) == int(pf / 1000):
        return '%dn' % int(pf / 1000)
    if pf == int(pf):
        return '%dp' % int(pf)
    return '%.4gp' % pf


def write_file(path, content):
    with open(path, 'w', newline='\n') as f:
        f.write(content)


ACDC_TEMPLATE = """Test OpAmp ACDC

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
.PARAM VCM_ratio = 0.25
.PARAM PARAM_CLOAD ={CLOAD}
.include ./{PREFIX}_vars.spice

V1 vdd 0 'supply_voltage'
V2 vss 0 0

Vindc opin 0 'supply_voltage*VCM_ratio'
Vin signal_in 0 dc 'supply_voltage*VCM_ratio' ac 1 sin('supply_voltage*VCM_ratio' 100m 500)

Lfb opout opout_dc 1T
Cin opout_dc signal_in 1T

* XOP gnda vdda vinn vinp vout Ib
*        |  |     |     |   |
*        |  |     |     |   Output
*        |  |     |     Non-inverting Input
*        |  |      Inverting Input
*        |  Positive Supply
*        Negative Supply

*    ADM TB
Ib Ib gnda DC='I_Ib'
x1 vss vdd opout_dc opin opout Ib {SUBCKT}
Cload1 opout 0 'PARAM_CLOAD'

*   ACM TB
vcmdc cm0 0 'supply_voltage*VCM_ratio'
vcmac1 cm1 cm0 0 ac=1
vcmac2 cm2 cm3 0 ac=1
Ib2 Ib2 gnda DC='I_Ib'
x2 vss vdd cm2 cm1 cm3 Ib2 {SUBCKT}
Cload2 cm3 0 'PARAM_CLOAD'

* PSRR   TB
VGNDApsrr gndpsrr 0 0 AC=1
VVDDApsrr vddpsrr 0 'supply_voltage'  AC=1
Ib3 Ib3 gnda DC='I_Ib'
x3 vss vddpsrr ppsr1 opin ppsr1 Ib3 {SUBCKT}
Cload3 ppsr1 0 'PARAM_CLOAD'

Ib4 Ib4 gnda DC='I_Ib'
x4 gndpsrr vdd npsr1 opin npsr1 Ib4 {SUBCKT}
Cload4 npsr1 0 'PARAM_CLOAD'

* DC ALL  TB
VVDDdc VDDdc 0 'supply_voltage'
Ib5 Ib5 gnda DC='I_Ib'
x5 vss vdddc vout6 opin vout6 Ib5 {SUBCKT}
Cload5 vout6 0 'PARAM_CLOAD'

.control
save all
.options savecurrents
set filetype=ascii
set units=degrees

DC temp -40 125 1
* TC meas
meas dc maxval MAX V(vout6) from=-40 to=125
meas dc minval MIN V(vout6) from=-40 to=125
meas dc avgval AVG V(vout6) from=-40 to=125
meas dc ppavl  PP V(vout6) from=-40 to=125
let TC = ppavl/avgval/165
* Power meas
meas dc Ivdd25 FIND I(VVDDDC) AT=25
let Power_ = -1 * Ivdd25 * 1.8
let Power = Power_ * 1e3
*   Vos.meas
meas dc vout25 FIND V(vout6) AT=25
let vos25 = vout25 - 1.8 * 0.25
wrdata {PREFIX}_ACDC_DC TC Power vos25
plot v(vout6)

ac dec 10 0.1 1G
meas ac cmrrdc find vdb(cm3) at = 0.1
meas ac dcgain_ find vdb(opout) at = 0.1
let dcgain = abs(dcgain_)
meas ac gain_bandwidth_product_ when vdb(opout)=0
let gbp = gain_bandwidth_product_*1e-4
meas ac phase_margin find vp(opout) when vdb(opout)=0
meas ac DCPSRp find vdb(ppsr1) at = 0.1
meas ac DCPSRn find vdb(npsr1) at = 0.1
wrdata {PREFIX}_ACDC_AC cmrrdc DCPSRp DCPSRn dcgain_
wrdata {PREFIX}_ACDC_GBW_PM gain_bandwidth_product_ phase_margin
plot vdb(opout) vdb(cm3) vdb(ppsr1) vdb(npsr1) vp(opout)

* OP
op
.include ./{PREFIX}_dev_params.spice
.endc

.end
"""

TRAN_TEMPLATE = """Test OpAmp Tran

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
.PARAM VCM_ratio = 0.25
.PARAM PARAM_CLOAD ={CLOAD}
.PARAM val0 = 3.000000e-01
.PARAM val1 = 5.000000e-01
.PARAM GBW_ideal = 5e4
.PARAM STEP_TIME = '10/GBW_ideal'
.PARAM TRAN_SIM_TIME = '20/GBW_ideal + 1e-6'

.include ./{PREFIX}_vars.spice

V1 vdd 0 'supply_voltage'
V2 vss 0 0

* XOP gnda vdda vinn vinp vout Ib

* Transient  TB
Ib Ib gnda DC='I_Ib'
VVISR visr 0 pulse('val0' 'val1' 1u 1p 1p '1*STEP_TIME' 1)
x1 vss vdd vout3 visr vout3 Ib {SUBCKT}
CLoad6 vout3 0 'PARAM_CLOAD'

.control
save all
.options savecurrents
set filetype=ascii
set units=degrees

tran 1u 4.01e-4
meas tran t_rise_edge when v(vout3)=0.4 rise=1
let t_rise_ = t_rise_edge-1u
let t_rise = t_rise_*1e6
let sr_rise = 0.1/t_rise
print t_rise
print sr_rise
meas tran t_fall_edge when v(vout3)=0.4 fall=1
let t_fall_ = t_fall_edge - 1u - 10/5e4
let t_fall = t_fall_*1e6
let sr_fall = 0.1/t_fall
print t_fall
print sr_fall
wrdata {PREFIX}_Tran sr_rise sr_fall
plot v(visr) v(vout3)
write {PREFIX}_tran.dat v(vout3) v(visr)

* OP
op
.include ./{PREFIX}_dev_params.spice
.endc

.end
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--upstream', required=True,
                    help='path to the AnalogGym repo checkout root')
    ap.add_argument('--name', help='upstream circuit name, e.g. Fan_SMC_Pin_3')
    ap.add_argument('--all', action='store_true',
                    help='convert every known circuit that has an upstream netlist')
    ap.add_argument('--force', action='store_true', help='overwrite existing instance')
    args = ap.parse_args()

    amp_dir = os.path.join(args.upstream, 'AnalogGym', 'Amplifier')
    names = [args.name] if args.name else (sorted(KNOWN_CIRCUITS) if args.all else [])
    if not names:
        ap.error('give --name or --all')
    for name in names:
        if name not in KNOWN_CIRCUITS:
            raise SystemExit('unknown circuit %s (add it to KNOWN_CIRCUITS)' % name)
        nl_path = os.path.join(amp_dir, 'spice_netlist', name)
        dv_path = os.path.join(amp_dir, 'design_variables', name)
        if not os.path.exists(nl_path) or os.path.getsize(nl_path) == 0:
            print('  [skip] %s: no upstream spice netlist' % name)
            continue
        print('converting %s ...' % name)
        amp = UpstreamAmp(nl_path, dv_path)
        convert(amp, name, KNOWN_CIRCUITS[name], force=args.force)


if __name__ == '__main__':
    main()
