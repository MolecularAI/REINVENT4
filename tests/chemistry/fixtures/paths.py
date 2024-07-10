import os
import json
from collections import defaultdict
from pathlib import Path


config_path = Path(__file__).parent / "integration_tests_config.json"
if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {}

MAIN_TEST_PATH = config.get("MAIN_TEST_PATH", "")

REACTION_DEFINITIONS_PATH = config.get("TEST_RESOURCES", {}).get("REACTION_DEFINITIONS_PATH", "")
AIZYNTH_PREDICTION_URL = config.get("AIZYNTH", {}).get("AIZYNTH_PREDICTION_URL", "")
AIZYNTH_BUILDING_BLOCKS_URL = config.get("AIZYNTH", {}).get("AIZYNTH_BUILDING_BLOCKS_URL", "")
AIZYNTH_TOKEN = config.get("AIZYNTH", {}).get("AIZYNTH_TOKEN", "")
