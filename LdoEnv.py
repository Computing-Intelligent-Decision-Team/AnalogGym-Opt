# -*- coding: utf-8 -*-
"""
LDO evaluation environment.

Reuses all of AmpEnv's simulation machinery (vars writing, ngspice invocation,
parallel/PVT corners, observation building) and only replaces the performance
extraction with LDO metrics, parsed from the wrdata files produced by the
LDO testbenches (see tools/import_analoggym_ldo.py):

  <PREFIX>_Vdrop_maxload / _minload           v(vout) vs supply sweep 1..3V
  <PREFIX>_ACDC_LNR_maxload / _minload        line regulation
  <PREFIX>_ACDC_LR_Power_vos                  load regulation, power, offset
  <PREFIX>_ACDC_PSRR_dcgain_maxload/_minload  PSRR at DC, loop gain
  <PREFIX>_ACDC_GBW_PM_maxload/_minload       loop GBW and phase margin
  <PREFIX>_Tran_meas                          load-step undershoot/overshoot

So that the GRPO/VAE multi-objective machinery (vae_model.RL_OBJECTIVE_KEYS)
works unchanged, the LDO objectives are mapped onto the repo's objective keys:

  constraint_reward <- phase margin + loop gain + PSRR + LNR + LR + vos + vdrop
  FOML_score        <- load-step ripple = undershoot + overshoot [V]  (min)
  FOMS_score        <- worst-case GBW * CL / Power [MHz*pF/mW]        (max)
  Active_Area_score <- same area model as the amplifiers              (min)
  'phase_margin (deg)' carries the worst-case (over min/max load) PM so that
  the PM_violation constraint behaves as for the amplifiers.
"""
import math
import os

import numpy as np

from AmpEnv import AmpEnv

# one 30x30 MiM cap unit, must match the netlist convention (pF)
CAP_UNIT_PF = (30.0 * 30.0 * 2e-15 + (30.0 + 30.0) * 0.38e-15) / 1e-12
VOUT_NOM = 1.6  # 4 * Vref


class LdoEnv(AmpEnv):

    # ---------------- wrdata parsing helpers ----------------
    def _wrdata_rows(self, sim_dir, path_key):
        """Read a wrdata file into a list of float rows (None if missing/empty)."""
        fname = os.path.basename(self.path[path_key])
        try:
            rows = []
            with open(os.path.join(sim_dir, fname), 'r') as f:
                for line in f:
                    vals = [float(t) for t in line.split()]
                    if vals:
                        rows.append(vals)
            return rows or None
        except Exception:
            print(f"Simulation errors in {sim_dir}: no file: {fname}.")
            return None

    def _wrdata_scalars(self, sim_dir, path_key, count):
        """First data row of a wrdata file holding scalar meas results.

        wrdata writes 'x v1 x v2 ...' per row; scalars repeat on every row,
        so the first row's odd columns carry the values.
        """
        rows = self._wrdata_rows(sim_dir, path_key)
        if rows is None or len(rows[0]) < 2 * count:
            return None
        return [rows[0][2 * i + 1] for i in range(count)]

    def _vdrop_from_sweep(self, sim_dir, path_key):
        """Dropout voltage from a v(vout) vs VDD sweep: Vin_reg - Vout_reg,
        where Vin_reg is the lowest supply at which vout is within 50mV of
        its value at the top of the sweep. Returns a large penalty value if
        the output never regulates."""
        rows = self._wrdata_rows(sim_dir, path_key)
        if rows is None:
            return None
        pts = sorted((r[0], r[1]) for r in rows)
        vout_top = pts[-1][1]
        if vout_top < 0.5 * VOUT_NOM:
            return 10.0
        for vin, vout in pts:
            if vout >= vout_top - 0.05:
                return max(0.0, vin - vout)
        return 10.0

    # ---------------- performance evaluation ----------------
    def _get_info(self, sim_results, individual_params):
        performance = self.circuit_config['performance']
        sim_dir = sim_results.base_dir

        def target(key):
            return performance[key]['target']

        # ---- DC: LR / Power / vos ----
        lrpv = self._wrdata_scalars(sim_dir, 'LR_Power_vos_path', 5)
        if lrpv is None:
            LR, Power_maxload, Power, vos = 10.0, 10.0, 10.0, 10.0
        else:
            LR = abs(lrpv[0])
            Power_maxload = lrpv[1] * 1e3  # mW, includes the 55mA load
            Power = lrpv[2] * 1e3          # mW at min load (quiescent-dominated)
            vos = max(abs(lrpv[3]), abs(lrpv[4]))

        lnr1 = self._wrdata_scalars(sim_dir, 'LNR_maxload_path', 1)
        lnr2 = self._wrdata_scalars(sim_dir, 'LNR_minload_path', 1)
        LNR = 10.0 if (lnr1 is None or lnr2 is None) else max(abs(lnr1[0]), abs(lnr2[0]))

        vdrop_max = self._vdrop_from_sweep(sim_dir, 'Vdrop_maxload_path')
        vdrop_min = self._vdrop_from_sweep(sim_dir, 'Vdrop_minload_path')
        vdrop = 10.0 if (vdrop_max is None or vdrop_min is None) \
            else max(vdrop_max, vdrop_min)

        # ---- AC: PSRR / loop gain / GBW / PM at both loads ----
        pd1 = self._wrdata_scalars(sim_dir, 'PSRR_dcgain_maxload_path', 2)
        pd2 = self._wrdata_scalars(sim_dir, 'PSRR_dcgain_minload_path', 2)
        gp1 = self._wrdata_scalars(sim_dir, 'GBW_PM_maxload_path', 2)
        gp2 = self._wrdata_scalars(sim_dir, 'GBW_PM_minload_path', 2)

        PSRR = 0.0 if (pd1 is None or pd2 is None) else max(pd1[0], pd2[0])
        dcgain = 0.0 if (pd1 is None or pd2 is None) else min(pd1[1], pd2[1])

        def pm_score_of(pm):
            if pm is None or not np.isfinite(pm):
                return -10.0
            if 45 <= pm <= 90:
                return 0.0
            if 90 < pm < 120:
                return (90 - pm) / 30
            if 0 < pm < 45:
                return (pm - 45) / 45
            return -10.0

        gbws, pms = [], []
        for gp in (gp1, gp2):
            if gp is not None:
                gbws.append(gp[0])
                pms.append(gp[1])
        if gbws:
            GBW = min(gbws)
            phase_margin = min(pms, key=pm_score_of)
            phase_margin_score = pm_score_of(phase_margin)
            GBW_score = self._score_higher_better(GBW, target('GBW'))
        else:
            GBW, phase_margin, phase_margin_score, GBW_score = 0.0, 0.0, -10.0, -1.0

        # ---- Tran: load-step undershoot / overshoot ----
        tm = self._wrdata_scalars(sim_dir, 'tran_meas_path', 2)
        if tm is None:
            undershoot, overshoot = 10.0, 10.0
        else:
            undershoot, overshoot = abs(tm[0]), abs(tm[1])

        # ---- scores ----
        dcgain_score = -1.0 if dcgain <= 0 else \
            self._score_higher_better(dcgain, target('dcgain'))
        PSRR_score = -1.0 if PSRR > 0 else self._score_lower_better(PSRR, target('PSRR'))
        LNR_score = self._score_lower_better(LNR, target('LNR'))
        LR_score = self._score_lower_better(LR, target('LR'))
        vos_score = self._score_lower_better(vos, target('vos'))
        vdrop_score = self._score_lower_better(vdrop, target('vdrop'))
        undershoot_score = self._score_lower_better(undershoot, target('undershoot'))
        overshoot_score = self._score_lower_better(overshoot, target('overshoot'))
        Power_score = self._score_lower_better(Power, target('Power'))

        Active_Area = self._calculate_area(individual_params)
        Active_Area_score = self._score_lower_better(Active_Area, target('Active_Area'))

        # ---- objectives mapped onto the repo MOO keys ----
        FOML = undershoot + overshoot  # transient ripple, V (smaller is better)
        FOML_score = self._score_lower_better(FOML, target('FOML'))
        CL_pF = float(individual_params.get('M_CL', 0)) * CAP_UNIT_PF
        FOMS = (GBW * 1e-6 * CL_pF) / Power if Power > 0 else 0.0
        FOMS_score = self._score_higher_better(FOMS, target('FOMS'))

        constraint_reward = (
            phase_margin_score
            + dcgain_score
            + PSRR_score
            + LNR_score
            + LR_score
            + vos_score
            + vdrop_score
        )

        if not np.isfinite(phase_margin):
            pm_violation = 1.0
        elif 45.0 < phase_margin < 90.0:
            pm_violation = 0.0
        elif phase_margin <= 45.0:
            pm_violation = (45.0 - phase_margin) / 45.0
        else:
            pm_violation = (phase_margin - 90.0) / 45.0

        # ---- optional post-target bonus (same semantics as AmpEnv) ----
        base_constraint_reward = float(constraint_reward)
        base_FOML_score = float(FOML_score)
        base_FOMS_score = float(FOMS_score)
        base_Active_Area_score = float(Active_Area_score)

        bonus_metrics = set(self.post_target_bonus_config.get(
            'metrics', ['FOML', 'FOMS', 'Active_Area']))
        bonus_scale = float(self.post_target_bonus_config.get('scale', 1.0))
        bonus_max = float(self.post_target_bonus_config.get('max_per_metric', 1.0))
        bonus_enabled = bool(self.post_target_bonus_config.get('enabled', False))
        all_targets_satisfied = (
            base_constraint_reward >= -1e-12
            and base_FOML_score >= -1e-12
            and base_FOMS_score >= -1e-12
            and base_Active_Area_score >= -1e-12
        )
        FOML_bonus = FOMS_bonus = Active_Area_bonus = 0.0
        bonus_active = bool(bonus_enabled and all_targets_satisfied)
        if bonus_active:
            if 'FOML' in bonus_metrics:
                FOML_bonus = self._bonus_lower_better(
                    FOML, target('FOML'), scale=bonus_scale, max_bonus=bonus_max)
            if 'FOMS' in bonus_metrics:
                FOMS_bonus = self._bonus_higher_better(
                    FOMS, target('FOMS'), scale=bonus_scale, max_bonus=bonus_max)
            if 'Active_Area' in bonus_metrics:
                Active_Area_bonus = self._bonus_lower_better(
                    Active_Area, target('Active_Area'),
                    scale=bonus_scale, max_bonus=bonus_max)

        FOML_score = base_FOML_score + FOML_bonus
        FOMS_score = base_FOMS_score + FOMS_bonus
        Active_Area_score = base_Active_Area_score + Active_Area_bonus
        reward = constraint_reward + FOML_score + FOMS_score + Active_Area_score
        reward_vector = np.array(
            [constraint_reward, FOML_score, FOMS_score, Active_Area_score],
            dtype=np.float32,
        )
        objective_rewards = {
            'constraint_reward': float(constraint_reward),
            'FOML_score': float(FOML_score),
            'FOMS_score': float(FOMS_score),
            'Active_Area_score': float(Active_Area_score),
            'PM_violation': float(pm_violation),
        }

        return {
            'phase_margin (deg)': phase_margin,
            'dcgain': dcgain,
            'PSRR': PSRR,
            'LNR': LNR,
            'LR': LR,
            'vos': vos,
            'vdrop': vdrop,
            'undershoot': undershoot,
            'overshoot': overshoot,
            'FOML': FOML,
            'FOMS': FOMS,
            'Active Area': Active_Area,
            'Power': Power,
            'Power_maxload': Power_maxload,
            'GBW': GBW,
            'phase_margin_score': phase_margin_score,
            'dcgain_score': dcgain_score,
            'PSRR_score': PSRR_score,
            'LNR_score': LNR_score,
            'LR_score': LR_score,
            'vos_score': vos_score,
            'vdrop_score': vdrop_score,
            'undershoot_score': undershoot_score,
            'overshoot_score': overshoot_score,
            'FOML_score': FOML_score,
            'FOMS_score': FOMS_score,
            'Active_Area_score': Active_Area_score,
            'Power_score': Power_score,
            'GBW_score': GBW_score,
            'PM_violation': pm_violation,
            'constraint_reward': constraint_reward,
            'reward_vector': reward_vector.tolist(),
            'objective_rewards': objective_rewards,
            'reward_components': {
                'reward': float(reward),
                'constraint_reward': float(constraint_reward),
                'FOML_score': float(FOML_score),
                'FOMS_score': float(FOMS_score),
                'Active_Area_score': float(Active_Area_score),
                'PM_violation': float(pm_violation),
                'base_constraint_reward': float(base_constraint_reward),
                'base_FOML_score': float(base_FOML_score),
                'base_FOMS_score': float(base_FOMS_score),
                'base_Active_Area_score': float(base_Active_Area_score),
                'FOML_bonus': float(FOML_bonus),
                'FOMS_bonus': float(FOMS_bonus),
                'Active_Area_bonus': float(Active_Area_bonus),
                'bonus_active': bool(bonus_active),
            },
            'post_target_bonus_active': bool(bonus_active),
            'post_target_bonus': {
                'FOML_bonus': float(FOML_bonus),
                'FOMS_bonus': float(FOMS_bonus),
                'Active_Area_bonus': float(Active_Area_bonus),
                'scale': float(bonus_scale),
                'max_per_metric': float(bonus_max),
            },
            'reward': reward,
        }
