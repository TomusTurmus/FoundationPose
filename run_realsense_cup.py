#!/usr/bin/env python3
"""
Run FoundationPose on RealSense cup data.

Run INSIDE the Docker container:
  python run_realsense_cup.py [--obj_id 77] [--debug 2]

KITchen object IDs (swap with --obj_id):
  77  mug           (dark red, with handle,  117×93×81 mm)
  29  cup_large     (large cup, with handle, 86×130×86 mm)
  80  FlowerCup     (rose mug,               108×111×76 mm)
   9  green-cup     (teal cup, with handle,  78×90×77 mm)
  83  g_cups        (orange cylindrical,     86×86×70 mm)
  81  h_cups        (blue cylindrical,       91×92×71 mm)
  66  i_cups        (green cylindrical,      98×97×72 mm)
  53  j_cups        (yellow cylindrical,    103×103×72 mm)

Data expected at:  demo_data/realsense_cup/
Results written to: debug_realsense/
"""

from estimater import *
from datareader import *
import argparse

KITCHEN_MODELS = "/home/pose/dipl/datasets/KITchen/models"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "--obj_id", type=int, default=77,
        help="KITchen object ID (see header comment for options)",
    )
    parser.add_argument(
        "--mesh_file", type=str, default="",
        help="Override mesh path directly (skips --obj_id)",
    )
    parser.add_argument(
        "--test_scene_dir",
        type=str,
        default=f"{code_dir}/demo_data/realsense_cup",
    )
    parser.add_argument(
        "--mask_dir", type=str, default="",
        help="Directory of per-frame mask PNGs (overrides default masks/ in scene dir). "
             "Use the SAM-6D per-object dir, e.g. demo_data/realsense_cup/masks/obj_000077/",
    )
    parser.add_argument("--est_refine_iter",   type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug",             type=int, default=2,
                        help="0=none  1=live vis  2=+save images  3=+point cloud")
    parser.add_argument("--debug_dir",         type=str,
                        default=f"{code_dir}/debug_realsense")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # Resolve mesh path: explicit override wins, otherwise use KITchen PLY
    mesh_path = args.mesh_file or f"{KITCHEN_MODELS}/obj_{args.obj_id:06d}.ply"
    mesh = trimesh.load(mesh_path, process=False)
    # KITchen PLY files are in mm; FoundationPose expects metres
    if mesh_path.endswith(".ply") and KITCHEN_MODELS in mesh_path:
        mesh.vertices /= 1000.0
    # Keep face count manageable for nvdiffrast VRAM budget
    if len(mesh.faces) > 50_000:
        mesh = mesh.simplify_quadric_decimation(50_000)
        logging.info(f"Mesh simplified to {len(mesh.faces)} faces")
    logging.info(f"Mesh: {mesh_path}  verts={len(mesh.vertices)}  faces={len(mesh.faces)}  "
                 f"extents(m)={mesh.extents.round(4)}")

    debug     = args.debug
    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("Estimator initialised")

    reader = YcbineoatReader(
        video_dir=args.test_scene_dir,
        shorter_side=None,   # keep native 640×360
        zfar=3.0,            # clip depth beyond 3 m
    )
    logging.info(
        f"Reader: {len(reader.color_files)} frames, "
        f"K=\n{reader.K}"
    )

    def load_mask(i):
        """Load mask for frame i. Checks --mask_dir first, then scene masks/, returns None if missing."""
        id_str = reader.id_strs[i]
        if args.mask_dir:
            path = os.path.join(args.mask_dir, f"{id_str}.png")
            m = cv2.imread(path, -1)
            if m is not None:
                return m.astype(bool)
        m = reader.get_mask(i)
        return m.astype(bool) if m is not None else None

    mask0 = load_mask(0)
    if mask0 is None or not mask0.any():
        raise RuntimeError(
            "Initial mask is empty! "
            "Run run_realsense_seg.sh (SAM-6D) or create_mask.py to create masks/000001.png"
        )

    pose = None
    for i in range(len(reader.color_files)):
        logging.info(f"Frame {i+1}/{len(reader.color_files)}")
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        frame_mask = load_mask(i)
        if frame_mask is None or not frame_mask.any():
            frame_mask = mask0

        if pose is None or frame_mask.any():
            # register: full hypothesis generation — needs a mask
            pose = est.register(
                K=reader.K,
                rgb=color,
                depth=depth,
                ob_mask=frame_mask,
                iteration=args.est_refine_iter,
            )
        else:
            # track: refine from previous pose — no mask needed, faster
            pose = est.track_one(
                rgb=color,
                depth=depth,
                K=reader.K,
                iteration=args.track_refine_iter,
            )

        if i == 0 and debug >= 3:
            m = mesh.copy()
            m.apply_transform(pose)
            m.export(f"{debug_dir}/model_tf.obj")
            xyz_map = depth2xyzmap(depth, reader.K)
            valid   = depth >= 0.001
            pcd     = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f"{debug_dir}/scene_complete.ply", pcd)

        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                color, ob_in_cam=center_pose, scale=0.05, K=reader.K,
                thickness=3, transparency=0, is_input_rgb=True,
            )
            if os.environ.get("DISPLAY"):
                cv2.imshow("FoundationPose – RealSense cup", vis[..., ::-1])
                cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)

    logging.info(f"Done. Poses saved to {debug_dir}/ob_in_cam/")
    logging.info(f"Visualisations saved to {debug_dir}/track_vis/")
    cv2.destroyAllWindows()
