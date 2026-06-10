"""Pure tests for the in-memory Presence tracker (no DB, deterministic `now`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jeff.presence import Presence


_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


def test_never_seen_is_not_present():
    p = Presence()
    assert p.last_seen("EpeerD") is None
    assert p.is_present("EpeerD", now=_NOW, ttl_s=300) is False


def test_seen_within_ttl_is_present():
    p = Presence()
    p.mark("EpeerD", now=_NOW - timedelta(seconds=60))
    assert p.is_present("EpeerD", now=_NOW, ttl_s=300) is True


def test_seen_beyond_ttl_is_absent():
    p = Presence()
    p.mark("EpeerD", now=_NOW - timedelta(seconds=600))
    assert p.is_present("EpeerD", now=_NOW, ttl_s=300) is False


def test_ttl_boundary_is_inclusive():
    p = Presence()
    p.mark("EpeerD", now=_NOW - timedelta(seconds=300))
    # Exactly at the TTL still counts as present (<=).
    assert p.is_present("EpeerD", now=_NOW, ttl_s=300) is True


def test_mark_updates_last_seen():
    p = Presence()
    p.mark("EpeerD", now=_NOW - timedelta(hours=1))
    p.mark("EpeerD", now=_NOW)
    assert p.last_seen("EpeerD") == _NOW
    assert p.is_present("EpeerD", now=_NOW, ttl_s=60) is True


def test_presence_is_per_peer():
    p = Presence()
    p.mark("EaliceD", now=_NOW)
    assert p.is_present("EaliceD", now=_NOW, ttl_s=60) is True
    assert p.is_present("EbobD", now=_NOW, ttl_s=60) is False
