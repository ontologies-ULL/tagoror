import asyncio
from owlready2 import Ontology
from serialization.base_serializer import BaseSerializer

class OntologyCache:
    """
    Static cache manager to avoid the expensive repetitive serialization of the 
    base ontology in high concurrency scenarios.
    """
    _cached_base_ontology_str = None
    _cache_lock = None

    @classmethod
    async def get_serialized(cls, base_ontology: Ontology, serializer: BaseSerializer) -> str:
        """
        Returns the serialized base ontology. If it's the first time it is called,
        it serializes it and temporarily blocks concurrent access (Double-checked locking).
        """
        if cls._cache_lock is None:
            cls._cache_lock = asyncio.Lock()
            
        if cls._cached_base_ontology_str is None:
            async with cls._cache_lock:
                if cls._cached_base_ontology_str is None:
                    cls._cached_base_ontology_str = await asyncio.to_thread(
                        serializer.process_ontology, base_ontology
                    )
                    
        return cls._cached_base_ontology_str

    @classmethod
    def clear_cache(cls):
        """
        Clears the cache. Very useful to invoke in the `teardown` of integration tests.
        """
        cls._cached_base_ontology_str = None
        cls._cache_lock = None