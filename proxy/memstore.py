"""In-memory stand-in for Firestore, for offline development only.

The access gate reads a record on every request, so once it exists the app
can't be used at all without a Firestore connection — which would have broken
the offline mock mode the rest of this codebase is careful to preserve (canned
agent responses, canned causal graphs, no credentials needed).

Opt in with ``ACCESS_STORE=memory``. Everything lives in a process-local dict
and is gone on restart, which is exactly right for a dev loop and obviously
wrong for production — hence the explicit env var rather than a fallback that
could silently engage when real credentials fail to load.

Implements only the slice of the Firestore surface proxy/ actually uses.
"""
from typing import Optional


def _is_increment(value) -> bool:
    """Recognise google.cloud.firestore.Increment without importing it."""
    return type(value).__name__ == "Increment"


class Snapshot:
    def __init__(self, doc_id: str, data: Optional[dict]):
        self.id = doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> Optional[dict]:
        return dict(self._data) if self._data is not None else None


class DocumentRef:
    def __init__(self, store: "MemoryDb", path: tuple):
        self._store = store
        self._path = path

    def collection(self, name: str) -> "CollectionRef":
        return CollectionRef(self._store, self._path + (name,))

    def get(self) -> Snapshot:
        return Snapshot(self._path[-1], self._store.docs.get(self._path))

    def set(self, data: dict, merge: bool = False) -> None:
        if merge and self._path in self._store.docs:
            self._store.docs[self._path].update(data)
        else:
            self._store.docs[self._path] = dict(data)

    def update(self, data: dict) -> None:
        doc = self._store.docs.setdefault(self._path, {})
        for key, value in data.items():
            doc[key] = doc.get(key, 0) + value.value if _is_increment(value) else value

    def delete(self) -> None:
        self._store.docs.pop(self._path, None)


class CollectionRef:
    def __init__(self, store: "MemoryDb", path: tuple):
        self._store = store
        self._path = path
        self._order_field: Optional[str] = None
        self._descending = False
        self._limit: Optional[int] = None
        self._start_after: Optional[dict] = None

    def document(self, doc_id: str) -> DocumentRef:
        return DocumentRef(self._store, self._path + (doc_id,))

    def add(self, data: dict) -> None:
        self._store.counter += 1
        self._store.docs[self._path + (f"auto-{self._store.counter:06d}",)] = dict(data)

    def order_by(self, field: str, direction=None) -> "CollectionRef":
        self._order_field = field
        self._descending = direction is not None and "DESC" in str(direction).upper()
        return self

    def limit(self, n: int) -> "CollectionRef":
        self._limit = n
        return self

    def start_after(self, cursor: dict) -> "CollectionRef":
        self._start_after = cursor
        return self

    def stream(self) -> list:
        depth = len(self._path) + 1
        items = [(p[-1], d) for p, d in self._store.docs.items()
                 if len(p) == depth and p[:-1] == self._path]
        if self._order_field:
            items = [kv for kv in items if kv[1].get(self._order_field) is not None]
            items.sort(key=lambda kv: kv[1][self._order_field], reverse=self._descending)
        if self._start_after is not None and self._order_field:
            after = self._start_after[self._order_field]
            items = [(i, d) for i, d in items
                     if (d[self._order_field] < after if self._descending
                         else d[self._order_field] > after)]
        if self._limit is not None:
            items = items[:self._limit]
        return [Snapshot(doc_id, data) for doc_id, data in items]


class MemoryDb:
    """Flat dict of path-tuple → document, mimicking Firestore's tree."""

    def __init__(self):
        self.docs: dict = {}
        self.counter = 0

    def collection(self, name: str) -> CollectionRef:
        return CollectionRef(self, (name,))
