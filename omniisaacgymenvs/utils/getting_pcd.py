from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni
import omni.replicator.core as rep

from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.scenes.scene import Scene
from omniisaacgymenvs.tasks.utils.pcd_writer import PointcloudWriter
from omniisaacgymenvs.tasks.utils.pcd_listener import PointcloudListener
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim

from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.semantics import add_update_semantics

from pxr import Usd, UsdGeom, Gf, UsdPhysics, Semantics


import torch

my_world = World(stage_units_in_meters=1.0)
# my_world.scene.add_default_ground_plane()
scene = Scene()

stage = omni.usd.get_context().get_stage()

# camera_positions = {0: [1.5, 1.5, 0.5],
#                     1: [2, -1.3, -0.5],
#                     2: [-0.5, -1.2, 0]}

# camera_positions = {0: [1.5, 1.5, 0.5],
#                     1: [2, -1.3, 0.5],
#                     2: [-0.5, -1.2, 0.5]}
                    # 2: [-1.5, -2.2, 0.5]}
                    # 2: [-4.8, -6.1, 0.5]}
camera_rotations = {0: [0, -35, 45],
                    1: [180, 145, -45],
                    2: [0, -35, -135],
                    3: [0, 35, -50]}

camera_positions = {0: [1, 1, 1],
                    1: [-1, 1, -1],
                    2: [-1, -1, 1],
                    3: [1, -1, -1]}

camera_width = 640
camera_height = 640

env_pos = torch.tensor([0, 0, 0], dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

render_products = []

# Load usd object
tool_name = 'spoon'
usd_file = f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/{tool_name}/{tool_name}.usd"
prim_path = f"/World/{tool_name}/{tool_name}"    # power_drill 앞에 숫자가 오면 안된다. 숫자가 있으면 오류남
# usd_file = "/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/cylinder/cylinder.usd"
# prim_path = "/World/cylinder"
# usd_file = f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/tool.usd"
# prim_path = "/World/mesh"

with rep.new_layer():

    # Add Default Light
    distance_light = rep.create.light(rotation=(1000,0,0), intensity=3000, light_type="distant")

    add_reference_to_stage(usd_path=usd_file, prim_path=prim_path)
    obj_prim_path = get_prim_at_path(prim_path)
    simulation_app.update()

    for i in range(len(camera_positions)):
        locals()[f"camera{i}"] = rep.create.camera(
                                                #    look_at=[0, 0, 0],
                                                   position=camera_positions[i],
                                                   rotation=camera_rotations[i],
                                                   name=f"camera{i}"
                                                   )
        render_product = rep.create.render_product(locals()[f"camera{i}"],
                                                   resolution=(camera_width, camera_height))
        render_products.append(render_product)

    rep.create.from_usd(usd_file, prim_path)

    rep.orchestrator._orchestrator._is_started = True
    simulation_app.update()

# initialize pytorch writer for vectorized collection
pointcloud_listener = PointcloudListener()
pointcloud_writer = rep.WriterRegistry.get("PointcloudWriter")
pointcloud_writer.initialize(listener=pointcloud_listener,
                             output_dir=f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/{tool_name}",
                            #  output_dir="/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/cylinder",
                            # output_dir=f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool",
                             pcd_sampling_num=4000,
                             pcd_normalize = False,
                             env_pos = env_pos,
                             camera_positions=camera_positions,
                             camera_orientations=camera_rotations,
                             name=tool_name,
                            #  name="cylinder",
                            #  name="mesh",
                             device=device,
                            )

# Attach render_product to the writer
pointcloud_writer.attach(render_products)
simulation_app.update()

# get object semantic data
obj_prim = stage.GetPrimAtPath(prim_path)
obj_semantics = Semantics.SemanticsAPI.Apply(obj_prim, "Semantics")
obj_semantics.CreateSemanticTypeAttr()
obj_semantics.CreateSemanticDataAttr()
obj_semantics.GetSemanticTypeAttr().Set("class")
obj_semantics.GetSemanticDataAttr().Set(f"obj_{i}")
add_update_semantics(obj_prim, '0')
simulation_app.update()

# # preview 부분까지 Script Editor에 복붙해서 실행시키면 어떤 환경을 찍어서 보여줄지 확인 가능
# # 이 때 simulation_app 관련된건 빼고 복붙해야함
# rep.orchestrator.preview()

# Run the simulation graph
rep.orchestrator.run()

# count = 0
# for i in range(2):
#     pointcloud = pointcloud_listener.get_pointcloud_data()
#     simulation_app.update()

# print("===================")
# print("Simulation is done")
# print("===================")

# simulation_app.close()

# exit()



while simulation_app.is_running():
    # rep.orchestrator.run()
    pointcloud = pointcloud_listener.get_pointcloud_data()
    simulation_app.update()
#     count += 1

#     if count > 1:
#         simulation_app.close()
#         print("===================")
#         print("Simulation is done")
#         print("===================")
#     break
#     #     exit()

    
# exit()

# # simulation_app.close()