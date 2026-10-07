"""Minimal Supabase client (PostgREST + Storage) built on requests."""
from urllib.parse import quote

import requests


class SupabaseClient:
    def __init__(self, cfg, session=None):
        self.url = cfg["url"].rstrip("/")
        self.key = cfg["service_key"]
        self.tables = cfg.get("tables", {})
        self.buckets = cfg.get("buckets", {})
        self.http = session or requests

    def table(self, key):
        return self.tables.get(key, key)

    def bucket(self, key):
        return self.buckets.get(key, key)

    def _headers(self, **extra):
        h = {"apikey": self.key, "Authorization": "Bearer " + self.key}
        h.update(extra)
        return h

    def insert(self, table, row):
        r = self.http.post(
            f"{self.url}/rest/v1/{self.table(table)}",
            json=row,
            headers=self._headers(**{"Content-Type": "application/json", "Prefer": "return=representation"}),
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data[0] if isinstance(data, list) and not isinstance(row, list) else data

    def select(self, table, filters=None, order=None, limit=None):
        """filters: {"status": "eq.new"} using PostgREST operators."""
        params = dict(filters or {})
        if order:
            params["order"] = order
        if limit:
            params["limit"] = limit
        r = self.http.get(
            f"{self.url}/rest/v1/{self.table(table)}", params=params, headers=self._headers(), timeout=60
        )
        r.raise_for_status()
        return r.json()

    def update(self, table, filters, values):
        r = self.http.patch(
            f"{self.url}/rest/v1/{self.table(table)}",
            params=filters,
            json=values,
            headers=self._headers(**{"Content-Type": "application/json", "Prefer": "return=representation"}),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def upload(self, bucket, path, data, content_type="application/octet-stream"):
        if isinstance(data, str):
            data = data.encode("utf-8")
        r = self.http.post(
            f"{self.url}/storage/v1/object/{self.bucket(bucket)}/{quote(path)}",
            data=data,
            headers=self._headers(**{"Content-Type": content_type, "x-upsert": "true"}),
            timeout=600,
        )
        r.raise_for_status()
        return path

    def download(self, bucket, path):
        r = self.http.get(
            f"{self.url}/storage/v1/object/{self.bucket(bucket)}/{quote(path)}",
            headers=self._headers(),
            timeout=600,
        )
        r.raise_for_status()
        return r.content
