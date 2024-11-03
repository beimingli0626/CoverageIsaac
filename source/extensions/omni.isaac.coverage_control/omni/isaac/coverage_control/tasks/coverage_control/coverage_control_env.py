from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject, RigidObjectCfg, ArticulationCfg
from omni.isaac.lab.envs import DirectMARLEnv

from .coverage_control_env_cfg import CoverageControlEnvCfg


class CoverageControlEnv(DirectMARLEnv):
    cfg: CoverageControlEnvCfg
    
    def __init__(self, cfg: CoverageControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("[INFO]: Number of agents per environments:", cfg.num_agents)
        
        self.vel_commands = {f"robot_{i+1}": torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device) for i in range(self.num_agents)}
        
    def _setup_scene(self):
        # TODO: modify robot cfg from rigid object to articulation; init_state initialization
        # create robots
        self.robots : dict[str, RigidObject] = {}
        for robot_name, robot_cfg in self.cfg.robots.items():
            robot = RigidObject(robot_cfg)
            self.scene.rigid_objects[robot_name] = robot
            self.robots[robot_name] = robot

        # create terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """ Preprocess actions output by network if needed
        
        E.g.: clamp, scale, velocity to thrust+moment
        """
        self.actions = actions  # current assumption is that network outputs velocity

    def _apply_action(self) -> None:
        """ Currently directly modify root velocity
        
        NOTE: maybe try directly modify root pose, more similar to LPAC
        """
        for robot_name, robot in self.robots.items():
            self.vel_commands[robot_name][:, 0] = self.actions[robot_name]
            robot.write_root_velocity_to_sim(self.vel_commands[robot_name])
        
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """ Local observation for each agent
        
        Returns:
            dict[agent_name, local_obs]
                local_obs:
                - local neighbour pose within communication/perception range
                - local observed IDF
                        
        # TODO
        """
        observations = {
           "robot_1" : torch.cat(
               (
                   torch.zeros(size=[1]),
               ),
               dim=-1,
           ),
        } 
        return observations
    
    def _get_states(self) -> torch.Tensor:
        """ Shared/global state for centralized value function

        Returns:
            - all agent poses
            - global IDF

        # TODO
        """
        states = torch.cat(
            (
                torch.zeros(size=[1]),
            ),
            dim=-1,
        )
        return states
    
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """ Return global/per-agent reward
        """
        # TODO
        return {"robot_1": torch.zeros(size=[1])}
    
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO
        return {"robot_1": torch.zeros(size=[1], dtype=torch.int)}, {"robot_1": torch.zeros(size=[1], dtype=torch.int)}
  
    # TODO  
    # def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
    #     pass