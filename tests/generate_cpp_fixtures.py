from pathlib import Path

from stage_fixtures import parser

FIXTURES_ROOT = Path(__file__).parent / "fixtures"

# One entry per stage with its own cpp_tests/ suite. Add a new
# stage_fixtures/<stage>.py (exposing generate(fixtures_root)) and list it
# here as that stage gains C++ fixtures of its own.
STAGES = [parser]

if __name__ == "__main__":
    for stage in STAGES:
        stage.generate(FIXTURES_ROOT)
