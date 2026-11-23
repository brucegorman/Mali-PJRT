from pathlib import Path

from jax._src import xla_bridge


def initialize():
    xla_bridge.register_plugin(
        "mali",
        priority=-100,
        library_path=str(Path(__file__).with_name("libpjrt_mali.so")),
        options=None,
    )
