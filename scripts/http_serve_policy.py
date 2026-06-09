import base64
import dataclasses
import io
import logging
import socket
from typing import Any, Dict, List, Optional

import numpy as np
np.set_printoptions(suppress=True, precision=4)
from PIL import Image
from robosuite.utils.transform_utils import axisangle2quat
from robosuite.utils.transform_utils import mat2pose
from robosuite.utils.transform_utils import pose2mat
from robosuite.utils.transform_utils import quat2axisangle
from scipy.spatial.transform import Rotation as R

try:
    # Prefer the canonical package imports if available
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
except Exception:  # pragma: no cover
    # Fall back to local files (as uploaded alongside this script)
    import policy as _policy  # type: ignore
    import policy_config as _policy_config  # type: ignore

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This server requires FastAPI and uvicorn. Please `pip install fastapi uvicorn`."
    ) from e


jaco_base_pose = np.array(
    [
        0,
        -0.45,
        0.91,  # copied from env.robots[0].base_pos
        0.0,
        0.0,
        0.0,
        1.0,  # copied from env.robots[0].base_ori
    ]
)  # jaco pose in world frame

# -0.5119, 1.5192, 0.1782
sim_gripper_rot = np.array([
    [0.064, -0.423, 0.904],
    [-0.202, 0.881, 0.427],
    [-0.977, -0.21, -0.029]
])

# 1.53, 0.75, 1.01
real_gripper_rot = np.array([
    [0.436, -0.059, 0.898],
    [0.877, -0.198, -0.438],
    [0.204, 0.978, -0.035]
])

sim2real_rot = real_gripper_rot @ sim_gripper_rot.T
real2sim_rot = sim_gripper_rot @ real_gripper_rot.T

open_gripper_state = np.array([1.51, 0.93, 1.51, 0.93])
close_gripper_state = np.array([0.02, -0.83, 0.02, -0.83])
gripper_state_range = open_gripper_state - close_gripper_state


# ----------------------------
# Request/Response Schemas
# ----------------------------


class InferRequest(BaseModel):
    # Single-camera (back-compat)
    image_b64: Optional[str] = None
    # Optional second camera (wrist)
    wrist_image_b64: Optional[str] = None
    # Or pass all images explicitly (len 1 or 2)
    images_b64: Optional[List[str]] = None

    instruction: Optional[str] = None  # client calls this 'instruction'
    state: Optional[List[float]] = None
    return_chunk: bool = True


class InferResponse(BaseModel):
    actions: List[List[float]]


# ----------------------------
# Utilities
# ----------------------------


def _decode_image(b64: str) -> Image.Image:
    """Decode base64-encoded image (RGB)."""
    data = base64.b64decode(b64, validate=True)
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return im


def _images_from_request(req: InferRequest) -> List[Image.Image]:
    """Decode images from the request. Supports single, dual, or list inputs.
    If only one image is provided but the policy expects two, callers can duplicate later.
    """
    imgs: List[Image.Image] = []
    if req.images_b64 is not None:
        if not isinstance(req.images_b64, list) or len(req.images_b64) == 0:
            raise ValueError("images_b64 must be a non-empty list")
        for s in req.images_b64:
            imgs.append(_decode_image(s))
    else:
        if req.image_b64:
            logging.info("Decoding third person view image")
            imgs.append(_decode_image(req.image_b64))
        if req.wrist_image_b64:
            logging.info("Decoding wrist image")
            imgs.append(_decode_image(req.wrist_image_b64))
    if len(imgs) == 0:
        raise ValueError("image_b64 (or images_b64) is required")
    return imgs


def _normalize_actions(result: Any) -> List[List[float]]:
    """Accept either a dict containing 'actions' or a raw array/list."""
    actions = result.get("actions") if isinstance(result, dict) else result
    if isinstance(actions, np.ndarray):
        actions = actions.tolist()
    if not isinstance(actions, list):
        raise TypeError(f"Policy returned unsupported type: {type(actions)}")
    # Ensure nested list of floats
    if actions and isinstance(actions[0], (int, float)):
        actions = [actions]  # single step -> list[step]
    return [[float(x) for x in step] for step in actions]


# ----------------------------
# HTTP App
# ----------------------------


def _make_http_app(
    policy: Any, port: int, default_prompt: Optional[str]
) -> FastAPI:
    app = FastAPI()
    hostname = socket.gethostname()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "hostname": hostname,
            "port": port,
            "has_infer": bool(hasattr(policy, "infer")),
            "has_act": bool(hasattr(policy, "act")),
            "has_predict": bool(hasattr(policy, "predict")),
        }

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest) -> InferResponse:
        try:
            images = _images_from_request(req)
            proprio = (
                np.asarray(req.state, dtype=np.float32)
                if req.state is not None
                else None
            )
            prompt = req.instruction if req.instruction else default_prompt

            # save images to disk
            for i, img in enumerate(images):
                img.save(f"image_{i}.png")

            actions = _call_policy_for_actions(
                policy, images, prompt, proprio, req.return_chunk
            )
            logging.info(f"sim_action: {actions[0]}")
            converted_actions = [
                jaco_sim2real_action_conversion(np.asarray(a)) for a in actions
            ]
            logging.info(f"real_action: {converted_actions[0]}")
            return InferResponse(actions=converted_actions)
        except Exception:
            logging.exception("Policy inference failed")
            # FastAPI will serialize this into a 500 with the exception text
            raise

    return app


def jaco_sim2real_action_conversion(action: np.ndarray) -> np.ndarray:
    """
    For evaluating VLA trained with sim data in real.
        - sim actions are in world frame
        - actions in real-world data are in jaco frame
    Converts actions from world frame to jaco frame so the actions output
    by VLA are executed correctly in real.
    """
    # jaco_world_rot = pose2mat((jaco_base_pose[:3], jaco_base_pose[3:7]))[
    #     :3, :3
    # ].T

    assert (
        action.shape[0] == 7
    ), "action array should consist of [dx,dy,dz,rx,ry,rz,g]"
    action_arr = action.copy()
    action_arr[1] *= -1
    # action_arr[4] *= -1

    return np.concatenate(
        (
            action_arr[:3] / 20,
            (sim2real_rot @ action_arr[3:6]),
            [action_arr[6]],
        )
    )
    
    # # real: jaco frame
    # # sim: world frame
    # # real2sim: R_{JW} @ delta_{w}
    # return np.concatenate(
    #     (
    #         jaco_world_rot @ action_arr[:3],
    #         jaco_world_rot @ action_arr[3:6],
    #         [action_arr[6]],
    #     )
    # )


def jaco_real2sim_state_conversion(state: np.ndarray) -> np.ndarray:
    """
    For evaluating VLA trained with sim data in real.
        - sim states are in world frame
        - states in real-world data are in jaco frame
    Converts states from jaco frame to world frame so VLA sees states in
    its familar frame.
    """
    world_jaco = pose2mat((jaco_base_pose[:3], jaco_base_pose[3:7]))

    assert (
        state.shape[0] == 10
    ), "state array should consist of [x,y,z,ax,ay,az,lfj1,lfj2,rfj1,rfj2]"
    state_arr = state.copy()
    state_arr[1] *= -1
    
    # sim: world frame
    # real: jaco frame
    # sim2real: T_{WJ} @ pose_{J}
    # jaco_frame_pose = np.concatenate(
    #     mat2pose(world_jaco @ pose2mat((state_arr[:3], axisangle2quat(state_arr[3:6]))))
    # )

    jaco_frame_pose = np.concatenate(
        mat2pose(world_jaco @ pose2mat((state_arr[:3], np.array([0,0,0,1]))))
    )

    real_rot = R.from_rotvec(state_arr[3:6]).as_matrix()

    return np.concatenate(
        (
            jaco_frame_pose[:3],
            R.from_matrix(real2sim_rot @ real_rot).as_rotvec(),
            state_arr[6:10],
        )
    )

    # return np.concatenate(
    #     (
    #         jaco_frame_pose[:3],
    #         quat2axisangle(jaco_frame_pose[3:7]),
    #         state_arr[6:10],
    #     )
    # )


# ----------------------------
# Policy calling (fixed)
# ----------------------------


def _call_policy_for_actions(
    policy: Any,
    images: List[Image.Image],
    prompt: Optional[str],
    proprio: Optional[np.ndarray],
    return_chunk: bool,
) -> List[List[float]]:
    """
    Try common inference entry points, preferring OpenPI-style `.infer(obs)`.
    Normalize to list[list[float]] for the client.
    """
    result: Any = None

    # 1) Preferred: OpenPI Policy.infer(obs)
    if hasattr(policy, "infer") and callable(getattr(policy, "infer")):
        # Build an observation dict that supports multiple schema variants used by
        # different OpenPI configs (CokeCan, Libero, ALOHA, etc.).
        obs: Dict[str, Any] = {}

        # --- State ---
        logging.info(f"real_state: {proprio}")
        state = (
            jaco_real2sim_state_conversion(proprio)
            if proprio is not None
            else []
        )
        logging.info(f"sim_state: {state}")
        # Common keys
        obs["state"] = state
        # Path-like keys used by certain transforms
        obs["observation/state"] = state
        obs["observation.state"] = state

        # --- Images ---
        if len(images) >= 1:
            img0 = images[0]
            obs["image"] = img0
            obs["observation/image"] = img0
            obs["observation.image"] = img0
        # Always provide a wrist image key: use second image if present, else duplicate first
        img1 = images[1] if len(images) >= 2 else images[0]
        obs["wrist_image"] = img1
        obs["observation/wrist_image"] = img1
        obs["observation.wrist_image"] = img1

        # --- Prompt / instruction ---
        if prompt is not None:
            obs["prompt"] = prompt
            obs["instruction"] = prompt
            obs["observation/prompt"] = prompt
            obs["observation.prompt"] = prompt

        result = policy.infer(obs)

    # 2) Legacy fallbacks
    elif hasattr(policy, "act") and callable(getattr(policy, "act")):
        result = policy.act(
            images=images,
            prompt=prompt,
            proprio=proprio,
            return_chunk=return_chunk,
        )
    elif hasattr(policy, "predict") and callable(getattr(policy, "predict")):
        result = policy.predict(images=images, prompt=prompt, proprio=proprio)
    elif callable(policy):
        result = policy(images=images, prompt=prompt, proprio=proprio)
    else:
        raise RuntimeError(
            "Policy has no 'infer', 'act', 'predict', or '__call__' entrypoint"
        )

    steps = _normalize_actions(result)

    if not return_chunk and len(steps) > 1:
        steps = steps[:1]
    return steps


# ----------------------------
# CLI / Policy construction
# ----------------------------


@dataclasses.dataclass
class Args:
    """Command-line arguments parsed by tyro."""

    port: int = 8030
    default_prompt: Optional[str] = None

    # # Optional: path to a trained checkpoint directory
    # policy_dir: Optional[str] = "checkpoints/pi05_jaco_custom_real"
    # # Name of the training config (see your config.get_config)
    # policy_config: str = "pi05_libero_custom_real"

    policy_dir: Optional[str] = (
        "checkpoints/pi05_jaco_custom_sim"
    )
    policy_config: str = "pi05_libero_custom_sim"

    # policy_dir: Optional[str] = (
    #     "checkpoints/pi05_libero_cma_mae/my_experiment/9999"
    # )
    # policy_config: str = "pi05_libero_custom_sim"

    # policy_dir: Optional[str] = (
    #     "checkpoints/pi05_libero_domain_randomization/my_experiment/9999"
    # )
    # policy_config: str = "pi05_libero_custom_sim"

    # policy_dir: Optional[str] = (
    #     "checkpoints/pi05_libero_cma_es/my_experiment/9999"
    # )
    # policy_config: str = "pi05_libero_custom_sim"

    # Device selection for PyTorch checkpoints (if applicable)
    pytorch_device: Optional[str] = None


def create_policy(args: Args) -> Any:
    """Load a trained policy using your local policy_config utilities."""
    # Import train config factory, trying canonical then local.
    try:
        from openpi.training import config as _config  # type: ignore
    except Exception:  # pragma: no cover
        import config as _config  # type: ignore

    # Build the TrainConfig by name
    train_cfg = _config.get_config(args.policy_config)
    if args.policy_dir is None:
        raise ValueError(
            "You must supply --policy_dir pointing to a checkpoint directory "
            "(e.g., /path/to/checkpoints/pi0_libero)."
        )

    logging.info(
        "Loading trained policy: config=%s dir=%s",
        args.policy_config,
        args.policy_dir,
    )

    policy = _policy_config.create_trained_policy(
        train_config=train_cfg,
        checkpoint_dir=args.policy_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )

    # Some factories return a bundle; unwrap common shapes
    if isinstance(policy, (list, tuple)) and len(policy) > 0:
        policy = policy[0]
    elif isinstance(policy, dict) and "policy" in policy:
        policy = policy["policy"]

    # If recording is needed, you can wrap with _policy.RecordingPolicy here.
    return policy


def main(args: Args) -> None:
    policy = create_policy(args)

    logging.info(
        "Loaded policy: %s act=%s predict=%s infer=%s callable=%s",
        type(policy),
        hasattr(policy, "act"),
        hasattr(policy, "predict"),
        hasattr(policy, "infer"),
        callable(policy),
    )

    app = _make_http_app(
        policy, port=args.port, default_prompt=args.default_prompt
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    import tyro  # noqa: F401

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
