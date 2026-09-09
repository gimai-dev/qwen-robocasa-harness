"""Reviewed coordinate-free visual recipes for official RoboCasa atomic tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRecipe:
    feature_kind: str
    guidance: str
    manipulation: str
    workspace_forward_m: float = 0.55
    continuation_base_m: float = 0.0
    workspace_reposition: bool = True


TASK_RECIPES = {
    "CloseDrawer": TaskRecipe(
        "handle",
        "the silver curved horizontal handle attached to the protruding already-open drawer front; avoid the countertop, microwave, vertical cabinet handles, and handles on closed drawers",
        "push",
        0.55,
        0.10,
    ),
    "OpenDrawer": TaskRecipe(
        "handle",
        "the handle attached to the currently closed task drawer, not a cabinet or appliance handle",
        "pull",
        0.38,
    ),
    "OpenMicrowave": TaskRecipe(
        "handle",
        "the microwave door handle, not the control panel, another appliance, or a cabinet handle",
        "pull",
    ),
    "CloseMicrowave": TaskRecipe(
        "front_surface",
        "the moving front surface of the already-open microwave door, not the fixed microwave body or controls",
        "push",
    ),
    "TurnOnMicrowave": TaskRecipe(
        "button",
        "the microwave start or power button on its control panel, not its display, dial, door, or another appliance",
        "push",
    ),
    "TurnOffMicrowave": TaskRecipe(
        "button",
        "the microwave stop button on its control panel, not its display, dial, door, start button, or another appliance",
        "push",
    ),
    "TurnOnBlender": TaskRecipe(
        "control",
        "the blender power button on the blender base, not its pitcher, lid, nearby fruit, or another appliance",
        "push",
        workspace_reposition=False,
    ),
    "OpenElectricKettleLid": TaskRecipe(
        "control",
        "the lid-release button on the electric kettle, not its handle, spout, power lever, lid surface, or another appliance",
        "push",
        workspace_reposition=False,
    ),
    "TurnOnElectricKettle": TaskRecipe(
        "control",
        "the small power lever on the electric kettle that the instruction says to press down, not its handle, spout, lid-release button, lid, or appliance body",
        "push_down",
        workspace_reposition=False,
    ),
    "TurnOnToaster": TaskRecipe(
        "control",
        "the toaster carriage lever that the instruction says to push down, not a bread slice, slot, crumb tray, button, dial, or appliance body",
        "push_down",
        workspace_reposition=False,
    ),
    "CloseFridge": TaskRecipe(
        "front_surface",
        "the moving front surface of the already-open refrigerator door, not a cabinet, drawer, or fixed fridge body",
        "push",
    ),
    "CloseToasterOvenDoor": TaskRecipe(
        "front_surface",
        "the moving front surface or safe broad edge of the already-open toaster oven door, not the fixed oven body, handle of a closed cabinet, rack, dial, or control",
        "push",
    ),
    "OpenFridgeDrawer": TaskRecipe(
        "handle",
        "the handle attached to the currently closed refrigerator drawer named by the instruction, not the main refrigerator door, shelf edge, or a cabinet handle",
        "pull",
        0.42,
    ),
}


def task_recipe(task: str) -> TaskRecipe:
    try:
        return TASK_RECIPES[task]
    except KeyError as error:
        raise ValueError("task has no reviewed visual recipe") from error


def recipe_system_prompt(task: str) -> str:
    recipe = task_recipe(task)
    return f"""You are the target-grounding component of an Inspect-style visual robot
controller for official RoboCasa365 PandaOmron. You receive exactly three synchronized
256x256 RGB images: left external, right external, wrist, plus only the official public
16D robot state. The wrist may be occluded. Do not infer depth, camera geometry, object
pose, joints, contacts, rewards, or success. Choose whichever one official external view
genuinely shows the task feature and return exactly one JSON object:
{{"kind":"center_feature","observation_id":"...","view":"robot0_agentview_left|robot0_agentview_right","feature_uv":[u,v],"feature_kind":"{recipe.feature_kind}"}}
Use precise pixel-derived normalized coordinates at least 12 pixels inside the image.
Choose a trackable textured edge or corner that belongs to the requested feature;
never cite a uniform blank center even when it is semantically correct. For a handle,
cite its approximate visible center; the deterministic public-RGB harness may refine
that semantic region to the nearest qualifying handle component.
For {task}, cite {recipe.guidance}. The later reviewed manipulation is
{recipe.manipulation}; do not execute it now. Do not cite the robot or invent an
occluded feature. Never output gripper pixels, depth, motion, done, or success."""
