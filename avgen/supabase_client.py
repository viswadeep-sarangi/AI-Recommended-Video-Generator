"""Minimal Supabase client (PostgREST + Storage) built on requests."""
import logging
from urllib.parse import quote

import requests

from .log import check_response

log = logging.getLogger(__name__)


class SupabaseClient:
    def __init__(self, cfg, session=None):
        self.url = cfg["url"].rstrip("/")
        self.key = cfg["service_key"]
        self.schema = cfg.get("schema", "avgen")
        self.tables = cfg.get("tables", {})
        self.buckets = cfg.get("buckets", {})
        self.http = session or requests
        if not self.url:
            log.error("Supabase URL is empty; is SUPABASE_URL set?")
        if not self.key:
            log.error("Supabase service key is empty; is SUPABASE_SERVICE_KEY set?")
        log.debug("Supabase client: schema=%s", self.schema)

    def table(self, key):
        return self.tables.get(key, key)

    def bucket(self, key):
        return self.buckets.get(key, key)

    def _headers(self, **extra):
        h = {"apikey": self.key, "Authorization": "Bearer " + self.key}
        h.update(extra)
        return h

    def _table_headers(self, profile_header, **extra):
        return self._headers(**{profile_header: self.schema, **extra})

    def _where(self, table):
        return f"{self.schema}.{self.table(table)}"

    def insert(self, table, row):
        log.debug("insert into %s (%d row(s))", self._where(table), len(row) if isinstance(row, list) else 1)
        r = self.http.post(
            f"{self.url}/rest/v1/{self.table(table)}",
            json=row,
            headers=self._table_headers(
                "Content-Profile",
                **{"Content-Type": "application/json", "Prefer": "return=representation"},
            ),
            timeout=60,
        )
        check_response(r, f"Supabase insert into {self._where(table)}")
        data = r.json()
        return data[0] if isinstance(data, list) and not isinstance(row, list) else data

    def upsert(self, table, row, conflict_column):
        log.debug("upsert into %s on_conflict=%s", self._where(table), conflict_column)
        r = self.http.post(
            f"{self.url}/rest/v1/{self.table(table)}",
            params={"on_conflict": conflict_column},
            json=row,
            headers=self._table_headers(
                "Content-Profile",
                **{
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates,return=representation",
                },
            ),
            timeout=60,
        )
        check_response(r, f"Supabase upsert into {self._where(table)}")
        data = r.json()
        return data[0] if isinstance(data, list) else data

    def select(self, table, filters=None, order=None, limit=None):
        """filters: {"status": "eq.new"} using PostgREST operators."""
        params = dict(filters or {})
        if order:
            params["order"] = order
        if limit:
            params["limit"] = limit
        log.debug("select from %s %s", self._where(table), params)
        r = self.http.get(
            f"{self.url}/rest/v1/{self.table(table)}",
            params=params,
            headers=self._table_headers("Accept-Profile"),
            timeout=60,
        )
        check_response(r, f"Supabase select from {self._where(table)}")
        rows = r.json()
        log.debug("select from %s returned %d row(s)", self._where(table), len(rows))
        return rows

    def update(self, table, filters, values):
        log.debug("update %s where %s set %s", self._where(table), filters, sorted(values))
        r = self.http.patch(
            f"{self.url}/rest/v1/{self.table(table)}",
            params=filters,
            json=values,
            headers=self._table_headers(
                "Content-Profile",
                **{"Content-Type": "application/json", "Prefer": "return=representation"},
            ),
            timeout=60,
        )
        check_response(r, f"Supabase update of {self._where(table)}")
        return r.json()

    def upload(self, bucket, path, data, content_type="application/octet-stream"):
        if isinstance(data, str):
            data = data.encode("utf-8")
        log.debug("upload %d byte(s) to storage %s/%s", len(data), self.bucket(bucket), path)
        r = self.http.post(
            f"{self.url}/storage/v1/object/{self.bucket(bucket)}/{quote(path)}",
            data=data,
            headers=self._headers(**{"Content-Type": content_type, "x-upsert": "true"}),
            timeout=600,
        )
        check_response(r, f"Supabase storage upload to {self.bucket(bucket)}/{path}")
        return path

    def download(self, bucket, path):
        log.debug("download from storage %s/%s", self.bucket(bucket), path)
        r = self.http.get(
            f"{self.url}/storage/v1/object/{self.bucket(bucket)}/{quote(path)}",
            headers=self._headers(),
            timeout=600,
        )
        check_response(r, f"Supabase storage download of {self.bucket(bucket)}/{path}")
        return r.content
