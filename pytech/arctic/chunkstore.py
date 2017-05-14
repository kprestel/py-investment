from arctic import CHUNK_STORE
from arctic.chunkstore.chunkstore import ChunkStore
from ..arctic import arctic

ARCTIC_LIB = ''

class PyTechChunkStore(ChunkStore):

    def __init__(self, arctic_lib):
        super().__init__(arctic_lib)
