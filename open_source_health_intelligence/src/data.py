"""GitHub REST ingestion, schema validation and deterministic fallback data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

import numpy as np
import pandas as pd
import requests


API_ROOT = "https://api.github.com"
REST_DOCS_URL = "https://docs.github.com/en/rest"
RATE_LIMIT_URL = "https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api"
TERMS_URL = "https://docs.github.com/en/site-policy/github-terms/github-terms-of-service"
ISSUES_DOCS_URL = "https://docs.github.com/en/rest/issues/issues#list-repository-issues"
PULLS_DOCS_URL = "https://docs.github.com/en/rest/pulls/pulls#list-pull-requests"
USER_AGENT = "HendrikDataPortfolio/1.0 (https://github.com/hendrikpw/projects)"

PRESETS = {
    "streamlit/streamlit": "Streamlit",
    "pandas-dev/pandas": "pandas",
    "scikit-learn/scikit-learn": "scikit-learn",
    "fastapi/fastapi": "FastAPI",
    "plotly/plotly.py": "Plotly.py",
    "huggingface/transformers": "Transformers",
}


def normalize_repository(value: str) -> str:
    """Normalize a public GitHub URL or owner/repository string."""
    text = str(value).strip().removesuffix("/")
    text = re.sub(r"^https?://github\.com/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.git$", "", text, flags=re.IGNORECASE)
    parts = [part for part in text.split("/") if part]
    if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError("Use owner/repository or a public github.com repository URL")
    return "/".join(parts)


def _get(session: requests.Session, path: str, params: dict | None = None, timeout: int = 25) -> requests.Response:
    response = session.get(
        f"{API_ROOT}{path}",
        params=params,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        },
        timeout=timeout,
    )
    if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        raise RuntimeError("GitHub API rate limit reached")
    response.raise_for_status()
    return response


def _parse_issues(payload: list[dict], retrieved_at: pd.Timestamp, is_demo: bool = False) -> pd.DataFrame:
    rows = []
    for item in payload:
        if item.get("pull_request"):
            continue
        created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
        closed = pd.to_datetime(item.get("closed_at"), utc=True, errors="coerce")
        updated = pd.to_datetime(item.get("updated_at"), utc=True, errors="coerce")
        if pd.isna(created):
            continue
        end = closed if pd.notna(closed) else retrieved_at
        rows.append(
            {
                "number": int(item.get("number", 0)),
                "title": str(item.get("title") or "Untitled issue"),
                "state": str(item.get("state") or "unknown"),
                "created_at": created,
                "updated_at": updated,
                "closed_at": closed,
                "duration_days": max(float((end - created).total_seconds() / 86400), 0.0),
                "event_observed": int(pd.notna(closed)),
                "comments": int(item.get("comments") or 0),
                "author": str((item.get("user") or {}).get("login") or "unknown"),
                "labels": ", ".join(str(label.get("name")) for label in item.get("labels", []) if label.get("name")),
                "url": str(item.get("html_url") or ""),
                "is_demo": bool(is_demo),
            }
        )
    return pd.DataFrame(rows)


def _parse_pulls(payload: list[dict], retrieved_at: pd.Timestamp, is_demo: bool = False) -> pd.DataFrame:
    rows = []
    for item in payload:
        created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
        closed = pd.to_datetime(item.get("closed_at"), utc=True, errors="coerce")
        merged = pd.to_datetime(item.get("merged_at"), utc=True, errors="coerce")
        updated = pd.to_datetime(item.get("updated_at"), utc=True, errors="coerce")
        if pd.isna(created):
            continue
        terminal = merged if pd.notna(merged) else closed
        end = terminal if pd.notna(terminal) else retrieved_at
        rows.append(
            {
                "number": int(item.get("number", 0)),
                "title": str(item.get("title") or "Untitled pull request"),
                "state": str(item.get("state") or "unknown"),
                "created_at": created,
                "updated_at": updated,
                "closed_at": closed,
                "merged_at": merged,
                "duration_days": max(float((end - created).total_seconds() / 86400), 0.0),
                "event_observed": int(pd.notna(terminal)),
                "is_merged": bool(pd.notna(merged)),
                "draft": bool(item.get("draft", False)),
                "author": str((item.get("user") or {}).get("login") or "unknown"),
                "url": str(item.get("html_url") or ""),
                "is_demo": bool(is_demo),
            }
        )
    return pd.DataFrame(rows)


def _parse_contributors(payload: list[dict], is_demo: bool = False) -> pd.DataFrame:
    rows = [
        {
            "contributor": str(item.get("login") or item.get("name") or "anonymous"),
            "contributions": int(item.get("contributions") or 0),
            "profile_url": str(item.get("html_url") or ""),
            "is_demo": bool(is_demo),
        }
        for item in payload
        if int(item.get("contributions") or 0) > 0
    ]
    return pd.DataFrame(rows).sort_values("contributions", ascending=False).reset_index(drop=True)


def _parse_releases(payload: list[dict], is_demo: bool = False) -> pd.DataFrame:
    rows = []
    for item in payload:
        published = pd.to_datetime(item.get("published_at"), utc=True, errors="coerce")
        if pd.isna(published) or item.get("draft"):
            continue
        rows.append(
            {
                "tag": str(item.get("tag_name") or "untagged"),
                "name": str(item.get("name") or item.get("tag_name") or "Unnamed release"),
                "published_at": published,
                "prerelease": bool(item.get("prerelease", False)),
                "url": str(item.get("html_url") or ""),
                "is_demo": bool(is_demo),
            }
        )
    return pd.DataFrame(rows).sort_values("published_at", ascending=False).reset_index(drop=True)


def _parse_commits(payload: list[dict], is_demo: bool = False) -> pd.DataFrame:
    rows = []
    for item in payload:
        commit = item.get("commit") or {}
        authored = pd.to_datetime((commit.get("author") or {}).get("date"), utc=True, errors="coerce")
        if pd.isna(authored):
            continue
        rows.append(
            {
                "sha": str(item.get("sha") or "")[:10],
                "authored_at": authored,
                "author": str(((item.get("author") or {}).get("login")) or (commit.get("author") or {}).get("name") or "unknown"),
                "message": str(commit.get("message") or "").splitlines()[0],
                "url": str(item.get("html_url") or ""),
                "is_demo": bool(is_demo),
            }
        )
    return pd.DataFrame(rows).sort_values("authored_at", ascending=False).reset_index(drop=True)


def fetch_repository_data(repository: str, timeout: int = 25) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fetch bounded public repository evidence from seven GitHub REST endpoints."""
    repository = normalize_repository(repository)
    session = requests.Session()
    retrieved_at = pd.Timestamp.now(tz="UTC")
    base = f"/repos/{repository}"
    repo_response = _get(session, base, timeout=timeout)
    repo = repo_response.json()
    if repo.get("private"):
        raise ValueError("Only public repositories are supported")
    calls = [
        _get(session, f"{base}/issues", {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100}, timeout),
        _get(session, f"{base}/pulls", {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100}, timeout),
        _get(session, f"{base}/contributors", {"per_page": 100, "anon": "1"}, timeout),
        _get(session, f"{base}/releases", {"per_page": 50}, timeout),
        _get(session, f"{base}/commits", {"per_page": 100}, timeout),
        _get(session, f"{base}/languages", timeout=timeout),
    ]
    issues, pulls, contributors, releases, commits, languages = (response.json() for response in calls)
    frames = {
        "issues": _parse_issues(issues, retrieved_at),
        "pulls": _parse_pulls(pulls, retrieved_at),
        "contributors": _parse_contributors(contributors),
        "releases": _parse_releases(releases),
        "commits": _parse_commits(commits),
        "languages": pd.DataFrame(
            [{"language": key, "bytes": int(value), "is_demo": False} for key, value in languages.items()]
        ).sort_values("bytes", ascending=False),
    }
    remaining = int(repo_response.headers.get("X-RateLimit-Remaining", -1))
    reset_value = repo_response.headers.get("X-RateLimit-Reset")
    reset_at = datetime.fromtimestamp(int(reset_value), timezone.utc).isoformat(timespec="seconds") if reset_value else None
    metadata = {
        "mode": "live",
        "repository": str(repo.get("full_name") or repository),
        "description": str(repo.get("description") or "No description provided."),
        "homepage": str(repo.get("homepage") or ""),
        "html_url": str(repo.get("html_url") or f"https://github.com/{repository}"),
        "default_branch": str(repo.get("default_branch") or ""),
        "primary_language": str(repo.get("language") or "Unknown"),
        "license": str((repo.get("license") or {}).get("spdx_id") or "Not declared"),
        "stars": int(repo.get("stargazers_count") or 0),
        "forks": int(repo.get("forks_count") or 0),
        "subscribers": int(repo.get("subscribers_count") or 0),
        "open_issues_reported": int(repo.get("open_issues_count") or 0),
        "archived": bool(repo.get("archived", False)),
        "created_at": str(repo.get("created_at") or ""),
        "pushed_at": str(repo.get("pushed_at") or ""),
        "retrieved_at": retrieved_at.isoformat(),
        "rate_limit_remaining_at_first_call": remaining,
        "rate_limit_reset_at": reset_at,
        "sample_limits": {"issues": 100, "pulls": 100, "contributors": 100, "releases": 50, "commits": 100},
    }
    return frames, metadata


def build_demo_data(repository: str = "streamlit/streamlit") -> tuple[dict[str, pd.DataFrame], dict]:
    """Create deterministic synthetic repository activity for rate-limit resilience."""
    repository = normalize_repository(repository)
    rng = np.random.default_rng(20260803)
    now = pd.Timestamp("2026-08-03T06:00:00Z")
    users = ["alex", "sam", "mira", "lee", "devon", "robin", "jules", "kai", "taylor", "casey"]
    issue_payload, pull_payload = [], []
    for number in range(1, 101):
        created = now - timedelta(days=int(rng.integers(2, 720)))
        is_closed = bool(rng.random() < 0.72)
        close_days = max(1, int(rng.lognormal(3.15, 1.0)))
        closed = min(created + timedelta(days=close_days), now) if is_closed else None
        issue_payload.append({
            "number": number, "title": f"Demo issue {number}: improve workflow", "state": "closed" if closed else "open",
            "created_at": created.isoformat(), "updated_at": (closed or now).isoformat(),
            "closed_at": closed.isoformat() if closed else None, "comments": int(rng.integers(0, 18)),
            "user": {"login": users[number % len(users)]}, "labels": [{"name": "demo"}], "html_url": "",
        })
        pr_created = now - timedelta(days=int(rng.integers(1, 540)))
        merged = bool(rng.random() < 0.78)
        pr_end = min(pr_created + timedelta(days=max(1, int(rng.lognormal(2.0, 1.0)))), now) if merged else None
        pull_payload.append({
            "number": number, "title": f"Demo pull request {number}", "state": "closed" if pr_end else "open",
            "created_at": pr_created.isoformat(), "updated_at": (pr_end or now).isoformat(),
            "closed_at": pr_end.isoformat() if pr_end else None, "merged_at": pr_end.isoformat() if pr_end else None,
            "draft": False, "user": {"login": users[(number * 3) % len(users)]}, "html_url": "",
        })
    contributions = np.maximum((1700 / (np.arange(1, 31) ** 1.18)).astype(int), 1)
    contributors = [{"login": f"contributor-{index:02d}", "contributions": int(value), "html_url": ""} for index, value in enumerate(contributions, 1)]
    release_dates = [now - timedelta(days=int(day)) for day in np.cumsum(rng.integers(18, 54, 20))]
    releases = [{"tag_name": f"v1.{20-index}.0", "name": f"Demo release {20-index}", "published_at": date.isoformat(), "draft": False, "prerelease": False, "html_url": ""} for index, date in enumerate(release_dates)]
    commits = [{"sha": f"demo{index:06d}", "commit": {"author": {"date": (now - timedelta(hours=index * 19.2)).isoformat(), "name": users[index % len(users)]}, "message": f"Demo commit {index}"}, "author": {"login": users[index % len(users)]}, "html_url": ""} for index in range(100)]
    frames = {
        "issues": _parse_issues(issue_payload, now, True),
        "pulls": _parse_pulls(pull_payload, now, True),
        "contributors": _parse_contributors(contributors, True),
        "releases": _parse_releases(releases, True),
        "commits": _parse_commits(commits, True),
        "languages": pd.DataFrame([{"language": name, "bytes": value, "is_demo": True} for name, value in {"Python": 2_450_000, "TypeScript": 880_000, "CSS": 175_000, "Shell": 72_000}.items()]),
    }
    return frames, {
        "mode": "demo", "repository": repository, "description": "Deterministic synthetic repository activity.",
        "homepage": "", "html_url": f"https://github.com/{repository}", "default_branch": "main",
        "primary_language": "Python", "license": "Apache-2.0", "stars": 41_800, "forks": 3_700,
        "subscribers": 530, "open_issues_reported": 620, "archived": False,
        "created_at": "2018-01-01T00:00:00Z", "pushed_at": (now - timedelta(hours=8)).isoformat(),
        "retrieved_at": now.isoformat(), "rate_limit_remaining_at_first_call": None, "rate_limit_reset_at": None,
        "sample_limits": {"issues": 100, "pulls": 100, "contributors": 100, "releases": 50, "commits": 100},
    }


def load_data(repository: str) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return live public GitHub evidence or a clearly labelled fallback."""
    try:
        return fetch_repository_data(repository)
    except (requests.RequestException, RuntimeError, ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        frames, metadata = build_demo_data(repository if "/" in str(repository) else "streamlit/streamlit")
        metadata["fallback_reason"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        return frames, metadata
