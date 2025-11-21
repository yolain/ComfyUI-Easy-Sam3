from .nodes import *
from typing_extensions import override

class Sam3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadSam3Model,
            Sam3ImageSegmentation,
            Sam3VideoSegmentation,
            Sam3VideoModelExtraConfig
        ]

async def comfy_entrypoint() -> Sam3Extension:
    return Sam3Extension()