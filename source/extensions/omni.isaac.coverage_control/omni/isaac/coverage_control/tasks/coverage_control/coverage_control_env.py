from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectMARLEnv

from .coverage_control_env_cfg import CoverageControlEnvCfg


class CoverageControlEnv(DirectMARLEnv):
    cfg: CoverageControlEnvCfg
    
    def __init__(self, cfg: CoverageControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot)
        self.scene.rigid_objects["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict[AgentID, ObsType]:
        return 