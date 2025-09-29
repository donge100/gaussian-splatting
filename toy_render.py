import os
import sys
import torch
import numpy as np
import json
from argparse import ArgumentParser
from tqdm import tqdm
import torchvision
import plyfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from scene import Scene, GaussianModel
from gaussian_renderer import render, render_reflected_gaussians
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args


# PLY 파일로 저장
def save_ply_from_attrs(path, attrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = attrs["xyz"]
    

    xyz_np = xyz.detach().cpu().numpy()
    f_dc_np = attrs["features_dc"].detach().cpu().numpy().reshape(-1, 3)
    f_rest_np = attrs["features_rest"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities_np = torch.sigmoid(attrs["opacity"]).detach().cpu().numpy()
    scales_np = torch.exp(attrs["scaling"]).detach().cpu().numpy()
    rotations_np = torch.nn.functional.normalize(attrs["rotation"]).detach().cpu().numpy()


    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    for i in range(f_rest_np.shape[1]):
        dtype_full.append((f'f_rest_{i}', 'f4'))
    dtype_full.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])

    normals = np.zeros_like(xyz_np)
    elements = np.empty(xyz_np.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz_np, normals, f_dc_np, f_rest_np, opacities_np, scales_np, rotations_np), axis=1)
    elements[:] = list(map(tuple, attributes))

    el = plyfile.PlyElement.describe(elements, 'vertex')
    plyfile.PlyData([el], text=True).write(path)
    print(f"Combined gaussians saved to {path}")

# 원본 + 반사 가우시안 렌더링
def render_reflection_set(model_path, name, iteration, views, gaussians, pipeline, background, mirror_transform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        original_image = render(view, gaussians, pipeline, background)["render"]
        # 불투명도 > 0.1 인 가우시안만 반사하여 유의미한 객체만 선택
        opacity_threshold = 0.1
        reflect_mask = (gaussians.get_opacity > opacity_threshold).squeeze()
        reflected_attrs = gaussians.reflect(mirror_transform, reflect_mask)

        # 디버깅을 위해 첫 프레임의 반사된 가우시안을 PLY 파일로 저장
        if idx == 0 and reflected_attrs is not None:
            save_ply_from_attrs(os.path.join(model_path, name, "reflected_debug.ply"), reflected_attrs)
            print("\nDEBUG: Saved reflected gaussians for the first frame to reflected_debug.ply")

        reflected_image = render_reflected_gaussians(view, gaussians, pipeline, background, mirror_transform, reflect_mask)["render"]

        final_image = torch.where(reflected_image.sum(dim=0) > 0, reflected_image, original_image)
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(final_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

# 렌더링 준비
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, save_ply: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        plane_json_path = os.path.join(dataset.model_path, "mirror_plane.json")
        if not os.path.exists(plane_json_path):
            raise FileNotFoundError(f"Mirror plane file not found at {plane_json_path}.")

        with open(plane_json_path, 'r') as f:
            plane_params = json.load(f)
        
        a, b, c, d = plane_params['a'], plane_params['b'], plane_params['c'], plane_params['d']
        mirror_transform = torch.from_numpy(np.array([
            [1-2*a*a, -2*a*b,   -2*a*c,   -2*a*d],
            [-2*a*b,  1-2*b*b,  -2*b*c,   -2*b*d],
            [-2*a*c,  -2*b*c,   1-2*c*c,  -2*c*d],
            [0,       0,        0,        1]
        ], dtype=np.float32)).cuda()

        if save_ply:
            print("\nExporting combined PLY file...")
            original_attrs = {
                "xyz": gaussians.get_xyz,
                "rotation": gaussians._rotation,
                "scaling": gaussians._scaling,
                "opacity": gaussians._opacity,
                "features_dc": gaussians._features_dc,
                "features_rest": gaussians._features_rest
            }
            reflect_mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            reflected_attrs = gaussians.reflect(mirror_transform, reflect_mask)

            if reflected_attrs:
                combined_attrs = {key: torch.cat((original_attrs[key], reflected_attrs[key]), dim=0) for key in original_attrs}
                output_ply_path = os.path.join(dataset.model_path, f"combined_reflection_iter{iteration}.ply")
                save_ply_from_attrs(output_ply_path, combined_attrs)
                print(f"\n[SUCCESS] Combined PLY with reflections saved to: {output_ply_path}")
            else:
                print("[INFO] No gaussians were reflected. PLY export skipped.")

        if not skip_train:
            render_reflection_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, mirror_transform)

        if not skip_test:
            render_reflection_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, mirror_transform)

if __name__ == "__main__":
    parser = ArgumentParser(description="Render original and reflected gaussians, with an option to save the combined PLY.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=7_000, type=int, help="Iteration number of the model to load.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_ply", action="store_true", help="Save the combined (original + reflected) gaussians to a .ply file.")
    args = get_combined_args(parser)
    
    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.save_ply)
