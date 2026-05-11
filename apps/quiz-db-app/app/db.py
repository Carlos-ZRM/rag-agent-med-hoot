"""
db.py – Database connection layer.

Reads the environment variable JDB_Engine to choose the backend:
  • "mongodb"   → Motor (async MongoDB driver)
  • "couchbase" → Couchbase Python SDK

Any CRUD helper in crud.py calls the functions exposed here so the
rest of the app never touches driver-specific objects directly.
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract interface every engine must implement
# ---------------------------------------------------------------------------

class DBEngine(ABC):
    """Minimal async interface that crud.py programmes against."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def insert(self, collection: str, doc: dict) -> str: ...

    @abstractmethod
    async def find_one(self, collection: str, filter: dict) -> dict | None: ...

    @abstractmethod
    async def find_many(
        self, collection: str, filter: dict, skip: int = 0, limit: int = 50
    ) -> list[dict]: ...

    @abstractmethod
    async def update(self, collection: str, id: str, fields: dict) -> bool: ...

    @abstractmethod
    async def delete(self, collection: str, id: str) -> bool: ...

    @abstractmethod
    async def count(self, collection: str, filter: dict) -> int: ...


# ---------------------------------------------------------------------------
# MongoDB implementation  (Motor – async)
# ---------------------------------------------------------------------------

class MongoEngine(DBEngine):
    def __init__(self) -> None:
        self.uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("MONGO_DB", "health_kahoot")
        self._client: Any = None
        self._db: Any = None

    async def connect(self) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

        self._client = AsyncIOMotorClient(self.uri)
        self._db = self._client[self.db_name]
        logger.info("MongoDB connected → %s / %s", self.uri, self.db_name)

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()

    # -- helpers --
    def _col(self, name: str):
        return self._db[name]

    @staticmethod
    def _id_str(result) -> str:
        return str(result.inserted_id)

    @staticmethod
    def _normalise(doc: dict | None) -> dict | None:
        if doc is None:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    # -- CRUD --
    async def insert(self, collection: str, doc: dict) -> str:
        res = await self._col(collection).insert_one(doc)
        return self._id_str(res)

    async def find_one(self, collection: str, filter: dict) -> dict | None:
        from bson import ObjectId

        if "id" in filter:
            filter["_id"] = ObjectId(filter.pop("id"))
        doc = await self._col(collection).find_one(filter)
        return self._normalise(doc)

    async def find_many(
        self, collection: str, filter: dict, skip: int = 0, limit: int = 50
    ) -> list[dict]:
        cursor = self._col(collection).find(filter).skip(skip).limit(limit)
        return [self._normalise(d) async for d in cursor]

    async def update(self, collection: str, id: str, fields: dict) -> bool:
        from bson import ObjectId

        res = await self._col(collection).update_one(
            {"_id": ObjectId(id)}, {"$set": fields}
        )
        return res.modified_count > 0

    async def delete(self, collection: str, id: str) -> bool:
        from bson import ObjectId

        res = await self._col(collection).delete_one({"_id": ObjectId(id)})
        return res.deleted_count > 0

    async def count(self, collection: str, filter: dict) -> int:
        return await self._col(collection).count_documents(filter)


# ---------------------------------------------------------------------------
# Couchbase implementation
# ---------------------------------------------------------------------------

class CouchbaseEngine(DBEngine):
    """
    Uses Couchbase Python SDK 4.x.
    Expects env vars:
      COUCHBASE_CONNSTR  – e.g. couchbase://localhost
      COUCHBASE_USER     – cluster username
      COUCHBASE_PASS     – cluster password
      COUCHBASE_BUCKET   – bucket name (default: health_kahoot)
    """

    def __init__(self) -> None:
        self.connstr = os.getenv("COUCHBASE_CONNSTR", "couchbase://localhost")
        self.user = os.getenv("COUCHBASE_USER", "Administrator")
        self.password = os.getenv("COUCHBASE_PASS", "password")
        self.bucket_name = os.getenv("COUCHBASE_BUCKET", "health_kahoot")
        self._cluster: Any = None
        self._bucket: Any = None

    async def connect(self) -> None:
        from couchbase.cluster import Cluster  # type: ignore
        from couchbase.options import ClusterOptions  # type: ignore
        from couchbase.auth import PasswordAuthenticator  # type: ignore
        from datetime import timedelta

        auth = PasswordAuthenticator(self.user, self.password)
        self._cluster = Cluster(self.connstr, ClusterOptions(auth))
        self._cluster.wait_until_ready(timedelta(seconds=10))
        self._bucket = self._cluster.bucket(self.bucket_name)
        logger.info("Couchbase connected → %s / %s", self.connstr, self.bucket_name)

    async def disconnect(self) -> None:
        pass  # SDK manages connection lifecycle

    def _scope_col(self, collection: str):
        return self._bucket.default_scope().collection(collection)

    # -- CRUD --
    async def insert(self, collection: str, doc: dict) -> str:
        import uuid

        doc_id = str(uuid.uuid4())
        doc["id"] = doc_id
        self._scope_col(collection).upsert(doc_id, doc)
        return doc_id

    async def find_one(self, collection: str, filter: dict) -> dict | None:
        if "id" in filter:
            try:
                result = self._scope_col(collection).get(filter["id"])
                return result.content_as[dict]
            except Exception:
                return None
        docs = await self.find_many(collection, filter, limit=1)
        return docs[0] if docs else None

    async def find_many(
        self, collection: str, filter: dict, skip: int = 0, limit: int = 50
    ) -> list[dict]:
        clauses = [f'`{k}` = "{v}"' if isinstance(v, str) else f"`{k}` = {v}"
                   for k, v in filter.items() if k != "$regex"]

        # Handle text search via LIKE
        if "$regex" in filter:
            field, pattern = next(iter(filter["$regex"].items()))
            clauses.append(f'LOWER(`{field}`) LIKE LOWER("%{pattern}%")')

        where = " AND ".join(clauses) if clauses else "TRUE"
        query = (
            f"SELECT META().id AS id, q.* "
            f"FROM `{self.bucket_name}`.`_default`.`{collection}` q "
            f"WHERE {where} "
            f"LIMIT {limit} OFFSET {skip}"
        )
        result = self._cluster.query(query)
        return [row for row in result]

    async def update(self, collection: str, id: str, fields: dict) -> bool:
        try:
            col = self._scope_col(collection)
            existing = col.get(id).content_as[dict]
            existing.update(fields)
            col.upsert(id, existing)
            return True
        except Exception:
            return False

    async def delete(self, collection: str, id: str) -> bool:
        try:
            self._scope_col(collection).remove(id)
            return True
        except Exception:
            return False

    async def count(self, collection: str, filter: dict) -> int:
        docs = await self.find_many(collection, filter, limit=999999)
        return len(docs)


# ---------------------------------------------------------------------------
# Factory – reads JDB_Engine and returns the right engine
# ---------------------------------------------------------------------------

_ENGINE_MAP: dict[str, type[DBEngine]] = {
    "mongodb": MongoEngine,
    "couchbase": CouchbaseEngine,
}

_instance: DBEngine | None = None


def get_engine() -> DBEngine:
    """Return the singleton DB engine based on JDB_Engine env var."""
    global _instance
    if _instance is None:
        name = os.getenv("JDB_Engine", "mongodb").lower().strip()
        cls = _ENGINE_MAP.get(name)
        if cls is None:
            supported = ", ".join(_ENGINE_MAP)
            raise ValueError(
                f"Unsupported JDB_Engine='{name}'. Choose from: {supported}"
            )
        _instance = cls()
        logger.info("DB engine selected → %s", name)
    return _instance
