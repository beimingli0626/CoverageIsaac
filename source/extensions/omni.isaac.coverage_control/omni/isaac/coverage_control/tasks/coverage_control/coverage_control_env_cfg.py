import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg, ArticulationCfg
from omni.isaac.lab.envs import DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

@configclass
class CoverageControlEnvCfg(DirectMARLEnvCfg):
    # extra env param
    num_agents: int = 2
    """Number of agents per environment, default to be 2
    
    Note: cannot dynamically change it for now, because the environment is registered in __init__.py before take in CLI arg
    """
    
    # env
    decimation = 4
    episode_length_s = 10
    possible_agents = [f"robot_{i+1}" for i in range(num_agents)]
    action_spaces = {f"robot_{i+1}": 1 for i in range(num_agents)}
    observation_spaces = {f"robot_{i+1}": 1 for i in range(num_agents)}
    state_space = 0
    
    # simulation NOTE: dt set to be 1, assume network output velocity command in 1Hz 
    sim: SimulationCfg = SimulationCfg(
        dt= 1,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    
    # TODO: use a cube for now, change to articulation later
    # multiple robots
    robots = {
        f"robot_{i+1}": RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/robot_{i+1}",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, i+1))
        ) for i in range(num_agents)
    }
    
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", 
        terrain_type="plane", 
        collision_group=-1,
        debug_vis=False,
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.5, replicate_physics=True)