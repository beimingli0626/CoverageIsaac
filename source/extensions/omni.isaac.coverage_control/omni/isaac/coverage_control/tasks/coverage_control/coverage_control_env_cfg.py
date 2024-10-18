import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.envs import DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass


@configclass
class CoverageControlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10
    possible_agents = ["robot"]
    action_spaces = {"robot": 1}
    observation_spaces = {"robot": 1}
    state_space = 0
    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt= 0.005,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    
    # robot
    # TODO: use a cube for now, change to articulation later
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5)),
    )
    
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", 
        terrain_type="plane", 
        collision_group=-1,
        debug_vis=False,
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.5, replicate_physics=True)