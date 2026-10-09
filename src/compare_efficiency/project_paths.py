from pathlib import Path

# Absolute paths, so notebooks work the same from the project root and from deep-dives/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
C_DIRECTORY = PROJECT_ROOT / "c"
BUILD_DIRECTORY = PROJECT_ROOT / "build"
SCRIPTS_DIRECTORY = PROJECT_ROOT / "scripts"
