# -*- coding: utf-8 -*-
"""Pick the evaluation environment for a circuit config by its category.

The category is derived by circuit_config_loader from the config file name
(everything before the first underscore): amp_* -> AmpEnv, ldo_* -> LdoEnv.
"""
from AmpEnv import AmpEnv
from LdoEnv import LdoEnv

_ENV_BY_CATEGORY = {
    'amp': AmpEnv,
    'ldo': LdoEnv,
}


def make_env(circuit_config):
    category = circuit_config.get('category') or \
        str(circuit_config.get('name', '')).split('_')[0]
    env_cls = _ENV_BY_CATEGORY.get(category, AmpEnv)
    return env_cls(circuit_config)
