"""
Coverage Control environment.
"""

import gymnasium as gym

from .coverage_control_env import CoverageControlEnv
from .coverage_control_env_cfg import CoverageControlEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Coverage-Control-Direct-v0",
    entry_point="omni.isaac.coverage_control.tasks.coverage_control:CoverageControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CoverageControlEnvCfg,
    #     "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
