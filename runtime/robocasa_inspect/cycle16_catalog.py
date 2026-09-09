"""Official catalog authorities for Cycle 16 invalid-pair stratification."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .cycle11_preflight import canonical_sha256

MODELSCOPE_REVISION = "f466f6ef5e9f98aed60d5c891bbd8d3f026bf7ff"

# Primary interaction fixture, reviewed from the official task definitions named
# here. Object-on-counter tasks use ``counter``; all other values are the exact
# FixtureType named by the task's setup reference.
TASK_CATALOG: dict[str, tuple[str, str]] = {
    "AdjustToasterOvenTemperature": ("toaster_oven", "kitchen_toaster_oven.py"),
    "AdjustWaterTemperature": ("sink", "kitchen_sink.py"),
    "CheesyBread": ("counter", "kitchen_pick_place.py"),
    "CloseBlenderLid": ("blender", "kitchen_blender.py"),
    "CloseToasterOvenDoor": ("toaster_oven", "kitchen_doors.py"),
    "CoffeeSetupMug": ("coffee_machine", "kitchen_coffee.py"),
    "LowerHeat": ("stove", "kitchen_stove.py"),
    "OpenElectricKettleLid": ("electric_kettle", "kitchen_electric_kettle.py"),
    "OpenFridgeDrawer": ("fridge", "kitchen_drawer.py"),
    "OpenToasterOvenDoor": ("toaster_oven", "kitchen_doors.py"),
    "PackDessert": ("counter", "kitchen_pick_place.py"),
    "PreheatOven": ("oven", "kitchen_oven.py"),
    "SlideToasterOvenRack": ("toaster_oven", "kitchen_toaster_oven.py"),
    "StartCoffeeMachine": ("coffee_machine", "kitchen_coffee.py"),
    "TurnOffStove": ("stove", "kitchen_stove.py"),
    "TurnOnBlender": ("blender", "kitchen_blender.py"),
    "TurnOnMicrowave": ("microwave", "kitchen_microwave.py"),
    "TurnOnToaster": ("toaster", "kitchen_toaster.py"),
    "TurnOnToasterOven": ("toaster_oven", "kitchen_toaster_oven.py"),
    "TurnSinkSpout": ("sink", "kitchen_sink.py"),
}


def task_catalog_authority(root: Path) -> dict[str, object]:
    source_root = (
        root
        / "upstream-robocasa/robocasa/environments/kitchen/atomic"
    )
    files = sorted({source for _, source in TASK_CATALOG.values()})
    return {
        "modelscope_revision": MODELSCOPE_REVISION,
        "task_fixture_mapping_sha256": canonical_sha256(TASK_CATALOG),
        "official_task_source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in files
        },
    }


def classify_catalog_pair(
    current: Mapping[str, object], source: Mapping[str, object]
) -> str:
    same_layout = int(current["layout_id"]) == int(source["layout_id"])
    same_fixture = str(current["fixture_class"]) == str(source["fixture_class"])
    return "_".join(
        (
            "same_layout" if same_layout else "different_layout",
            "same_fixture" if same_fixture else "different_fixture",
        )
    )


__all__ = [
    "MODELSCOPE_REVISION",
    "TASK_CATALOG",
    "classify_catalog_pair",
    "task_catalog_authority",
]
