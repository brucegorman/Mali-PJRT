from pathlib import Path

from jax._src import xla_bridge
from jax._src.lib import xla_client


def initialize():
    xla_bridge.register_plugin(
        "mali",
        priority=-100,
        library_path=str(Path(__file__).with_name("libpjrt_mali.so")),
        options={"mali_compile_defaults": xla_client.CompileOptions().SerializeAsString().hex()},
    )
