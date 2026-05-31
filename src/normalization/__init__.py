"""Normalization and identity resolution interfaces."""

from .rdkit_normalizer import RDKitNormalizer

# Neo4j client requires the ``neo4j`` package which may not be installed.
# Import it lazily so RDKitNormalizer can still be used without it.
_neo4j_exports: dict[str, type] = {}


def __getattr__(name: str):
    if name in _neo4j_exports:
        return _neo4j_exports[name]
    if name in {"Neo4jIdentityClient", "Neo4jIdentityConfig"}:
        from .neo4j_identity_client import (  # noqa: E402
            Neo4jIdentityClient,
            Neo4jIdentityConfig,
        )

        _neo4j_exports["Neo4jIdentityClient"] = Neo4jIdentityClient
        _neo4j_exports["Neo4jIdentityConfig"] = Neo4jIdentityConfig
        return _neo4j_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RDKitNormalizer",
    "Neo4jIdentityClient",
    "Neo4jIdentityConfig",
]
