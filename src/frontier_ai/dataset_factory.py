from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from frontier_ai.pipeline import (
    DATA_PROCESSED,
    DATA_RAW,
    LMARENA_RESOLVE,
    LMARENA_TREE_URL,
    OPENROUTER_URL,
    REFERENCE_DATE,
    ROOT,
    clean_label,
    clean_text,
    classify_access,
    classify_entity,
    classify_family,
    download,
    read_json,
    safe_float,
    slug,
    write_json,
    write_run_manifest,
)

DATASET_DIR = ROOT / "data" / "dataset"
RICH_RAW = DATA_RAW / "rich"
CAPTURED_AT = datetime.now(UTC).replace(microsecond=0).isoformat()

HF_AUTHORS = [
    "meta-llama",
    "mistralai",
    "Qwen",
    "deepseek-ai",
    "google",
    "microsoft",
    "openai",
    "NousResearch",
    "nvidia",
    "bigscience",
    "allenai",
    "tiiuae",
    "EleutherAI",
    "openbmb",
    "01-ai",
    "databricks",
    "stabilityai",
    "HuggingFaceH4",
    "Open-Orca",
    "baichuan-inc",
    "Salesforce",
    "ibm-granite",
    "Snowflake",
    "togethercomputer",
    "TheBloke",
    "teknium",
    "mosaicml",
    "lmsys",
    "xai-org",
]

HF_SEARCH_QUERIES = [
    "large language model",
    "instruction tuned",
    "reasoning model",
    "code model",
    "multimodal",
    "open weights",
    "chat model",
    "mixture of experts",
]

OPENALEX_QUERIES = [
    "large language model",
    "frontier AI model",
    "language model benchmark",
    "open weight language model",
    "retrieval augmented generation",
    "AI agent benchmark",
    "LLM evaluation",
    "mixture of experts language model",
]

GITHUB_QUERIES = [
    "topic:large-language-model stars:>50",
    "topic:llm stars:>100",
    "topic:rag stars:>100",
    "topic:llm-agent stars:>50",
    "topic:llmops stars:>30",
    "topic:open-source-llm stars:>20",
    "topic:ai-agent stars:>100",
    "large language model stars:>200",
    "LLM benchmark stars:>20",
]

MODEL_MENTION_PATTERNS = {
    "GPT": r"\bGPT[- ]?(?:3(?:\.5)?|4(?:o|\.1)?|5(?:\.\d+)?)\b|\bo[1345]\b",
    "Claude": r"\bClaude(?:\s+(?:Opus|Sonnet|Haiku))?(?:\s+\d(?:\.\d)?)?\b",
    "Gemini": r"\bGemini(?:\s+\d(?:\.\d)?)?(?:\s+Pro|\s+Flash|\s+Ultra)?\b",
    "Llama": r"\bLlama[- ]?\d(?:\.\d)?(?:[- ]?\d+B)?\b",
    "DeepSeek": r"\bDeepSeek[- ]?(?:R1|V\d|Coder|V3|V4)?\b",
    "Qwen": r"\bQwen(?:\d(?:\.\d)?)?(?:[- ]?\d+B)?\b",
    "Mistral": r"\bMistral|Mixtral\b",
    "Grok": r"\bGrok[- ]?\d(?:\.\d+)?\b",
    "Gemma": r"\bGemma[- ]?\d?\b",
    "Phi": r"\bPhi[- ]?\d(?:\.\d)?\b",
}


class Client:
    def __init__(self, sleep: float = 0.15) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ai-capability-signals/0.1 (public reproducible research)"})
        self.sleep = sleep

    def get_json(self, url: str, path: Path | None = None, overwrite: bool = False, timeout: int = 90) -> Any:
        if path and path.exists() and not overwrite:
            return read_json(path)
        response = self.session.get(url, timeout=timeout)
        if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
            reset = int(response.headers.get("X-RateLimit-Reset", "0") or "0")
            wait = max(1, min(60, reset - int(time.time()) + 1))
            time.sleep(wait)
            response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        time.sleep(self.sleep)
        data = response.json()
        if path:
            write_json(path, data)
        return data


def ensure_dirs() -> None:
    for path in [DATASET_DIR, RICH_RAW / "hf_models", RICH_RAW / "hf_trees", RICH_RAW / "openalex", RICH_RAW / "github", RICH_RAW / "openllm", RICH_RAW / "lmarena_full"]:
        path.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str, source_ids: list[str], manifest: list[dict[str, Any]]) -> pd.DataFrame:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    csv_path = DATASET_DIR / f"{name}.csv"
    parquet_path = DATASET_DIR / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        parquet_df = sanitize_for_parquet(df)
        parquet_df.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception:
        parquet_written = False
    manifest.append(
        {
            "table": name,
            "rows": len(df),
            "columns": len(df.columns),
            "csv_path": str(csv_path.relative_to(ROOT)),
            "parquet_path": str(parquet_path.relative_to(ROOT)) if parquet_written else "",
            "source_ids": ",".join(source_ids),
            "captured_at": CAPTURED_AT,
        }
    )
    return df


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(normalize_cell_for_storage)
    return out


def normalize_cell_for_storage(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return value
    return [value]


def hf_license(tags: list[str], card_data: Any) -> str:
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    if isinstance(card_data, dict):
        license_value = card_data.get("license")
        if isinstance(license_value, list):
            return ",".join(str(v) for v in license_value)
        return clean_text(license_value)
    return ""


def infer_params_from_text(text: str) -> float | None:
    text = text or ""
    matches = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([bBmMtT])(?:\b|[-_])", text)
    if not matches:
        return None
    value, unit = matches[-1]
    multiplier = {"m": 1e6, "b": 1e9, "t": 1e12}[unit.lower()]
    return float(value) * multiplier


def flatten_model_item(item: dict[str, Any], collection: str) -> dict[str, Any]:
    tags = [str(tag) for tag in listify(item.get("tags"))]
    model_id = clean_text(item.get("id") or item.get("modelId"))
    author = clean_text(item.get("author") or model_id.split("/", 1)[0])
    entity = classify_entity(model_id, author)
    card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
    return {
        "model_id": model_id,
        "author": author,
        "collection": collection,
        "vendor": entity.vendor,
        "model_family": entity.model_family,
        "product_line": entity.product_line,
        "family": entity.model_family,
        "access_class": classify_access(model_id, author, "", hf_license(tags, card_data)),
        "downloads": safe_float(item.get("downloads")),
        "likes": safe_float(item.get("likes")),
        "pipeline_tag": clean_text(item.get("pipeline_tag")),
        "library_name": clean_text(item.get("library_name")),
        "created_at": clean_text(item.get("createdAt")),
        "last_modified": clean_text(item.get("lastModified")),
        "private": item.get("private"),
        "gated": item.get("gated"),
        "disabled": item.get("disabled"),
        "sha": clean_text(item.get("sha")),
        "license": hf_license(tags, card_data),
        "base_model": clean_text(card_data.get("base_model")),
        "language": ",".join(str(v) for v in listify(card_data.get("language"))),
        "datasets": ",".join(str(v) for v in listify(card_data.get("datasets"))),
        "tags_count": len(tags),
        "parameters_inferred_from_id": infer_params_from_text(model_id),
        "source_url": f"https://huggingface.co/{model_id}",
        "source_id": "huggingface_models_api",
    }


def fetch_huggingface_models(client: Client, overwrite: bool, per_author: int, per_query: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: dict[str, dict[str, Any]] = {}
    tag_rows = []
    for author in HF_AUTHORS:
        url = f"https://huggingface.co/api/models?author={quote(author)}&limit={per_author}&full=true&sort=downloads&direction=-1"
        path = RICH_RAW / "hf_models" / f"author_{slug(author)}.json"
        try:
            items = client.get_json(url, path=path, overwrite=overwrite)
        except Exception as exc:
            write_json(RICH_RAW / "hf_models" / f"author_{slug(author)}_error.json", {"url": url, "error": str(exc)})
            continue
        for item in items:
            flat = flatten_model_item(item, f"author:{author}")
            rows[flat["model_id"]] = {**rows.get(flat["model_id"], {}), **flat}
            for tag in listify(item.get("tags")):
                tag_rows.append({"model_id": flat["model_id"], "tag": str(tag), "tag_slug": slug(str(tag)), "source_id": "huggingface_models_api"})
    for query in HF_SEARCH_QUERIES:
        url = f"https://huggingface.co/api/models?search={quote(query)}&limit={per_query}&full=true&sort=downloads&direction=-1"
        path = RICH_RAW / "hf_models" / f"search_{slug(query)}.json"
        try:
            items = client.get_json(url, path=path, overwrite=overwrite)
        except Exception as exc:
            write_json(RICH_RAW / "hf_models" / f"search_{slug(query)}_error.json", {"url": url, "error": str(exc)})
            continue
        for item in items:
            flat = flatten_model_item(item, f"search:{query}")
            if flat["model_id"] in rows:
                rows[flat["model_id"]]["collection"] += f"|search:{query}"
            else:
                rows[flat["model_id"]] = flat
            for tag in listify(item.get("tags")):
                tag_rows.append({"model_id": flat["model_id"], "tag": str(tag), "tag_slug": slug(str(tag)), "source_id": "huggingface_models_api"})
    models = pd.DataFrame(rows.values()).sort_values(["downloads", "likes"], ascending=False, na_position="last")
    tags = pd.DataFrame(tag_rows).drop_duplicates()
    return models, tags


def fetch_hf_model_files(client: Client, hf_models: pd.DataFrame, overwrite: bool, top_n: int) -> pd.DataFrame:
    rows = []
    top = hf_models.sort_values(["downloads", "likes"], ascending=False, na_position="last").head(top_n)
    for _, model in top.iterrows():
        model_id = model["model_id"]
        encoded = quote(model_id, safe="/")
        url = f"https://huggingface.co/api/models/{encoded}/tree/main?recursive=true"
        path = RICH_RAW / "hf_trees" / f"{slug(model_id)}.json"
        try:
            tree = client.get_json(url, path=path, overwrite=overwrite)
        except Exception as exc:
            rows.append({"model_id": model_id, "path": "", "error": str(exc), "source_id": "huggingface_model_tree_api"})
            continue
        for item in tree:
            file_path = clean_text(item.get("path"))
            ext = Path(file_path).suffix.lower()
            lfs = item.get("lfs") or {}
            rows.append(
                {
                    "model_id": model_id,
                    "path": file_path,
                    "type": clean_text(item.get("type")),
                    "extension": ext,
                    "size_bytes": safe_float(item.get("size")),
                    "lfs_size_bytes": safe_float(lfs.get("size")),
                    "is_lfs": bool(lfs),
                    "is_weight_file": ext in {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".ckpt"},
                    "is_config_file": Path(file_path).name in {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"},
                    "source_url": f"https://huggingface.co/{model_id}/tree/main/{file_path}",
                    "source_id": "huggingface_model_tree_api",
                    "error": "",
                }
            )
    return pd.DataFrame(rows)


def build_hf_model_rollups(hf_models: pd.DataFrame, hf_files: pd.DataFrame, hf_tags: pd.DataFrame) -> pd.DataFrame:
    required_file_cols = {"model_id", "path", "size_bytes", "lfs_size_bytes", "is_weight_file", "is_config_file"}
    if hf_files.empty or not required_file_cols.issubset(hf_files.columns):
        file_rollup = pd.DataFrame(columns=["model_id"])
    else:
        file_rollup = hf_files.groupby("model_id").agg(
            file_count=("path", "count"),
            total_file_bytes=("size_bytes", "sum"),
            lfs_file_bytes=("lfs_size_bytes", "sum"),
            weight_file_count=("is_weight_file", "sum"),
            config_file_count=("is_config_file", "sum"),
        ).reset_index()
    tag_pivot = pd.DataFrame(columns=["model_id"])
    if not hf_tags.empty:
        tag_pivot = hf_tags.groupby("model_id").agg(
            has_eval_results=("tag", lambda s: any(str(v) == "eval-results" for v in s)),
            has_safetensors=("tag", lambda s: any(str(v) == "safetensors" for v in s)),
            has_text_generation_inference=("tag", lambda s: any(str(v) == "text-generation-inference" for v in s)),
            arxiv_tags=("tag", lambda s: ",".join(sorted(v for v in map(str, s) if v.startswith("arxiv:")))),
            base_model_tags=("tag", lambda s: ",".join(sorted(v for v in map(str, s) if v.startswith("base_model:"))[:5])),
        ).reset_index()
    out = hf_models.merge(file_rollup, on="model_id", how="left").merge(tag_pivot, on="model_id", how="left")
    for col in ["file_count", "total_file_bytes", "lfs_file_bytes", "weight_file_count", "config_file_count"]:
        if col in out:
            out[col] = out[col].fillna(0)
    return out


def fetch_lmarena_full(overwrite: bool) -> pd.DataFrame:
    tree_path = download(LMARENA_TREE_URL, RICH_RAW / "lmarena_full" / "tree.json", overwrite=overwrite).path
    tree = read_json(tree_path)
    selected = {"text", "vision", "webdev", "search", "document", "text_to_image", "image_edit"}
    rows = []
    for item in tree:
        path = item.get("path", "")
        if item.get("type") != "file" or not path.endswith("full-00000-of-00001.parquet"):
            continue
        category = path.split("/", 1)[0]
        if category not in selected:
            continue
        local = RICH_RAW / "lmarena_full" / path
        download(LMARENA_RESOLVE.format(path=path), local, overwrite=overwrite)
        frame = pd.read_parquet(local)
        frame = frame.copy()
        frame["category"] = category
        frame["source_id"] = "lmarena_full_dataset"
        frame["source_url"] = f"https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/tree/main/{category}"
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def fetch_openllm_results(client: Client, overwrite: bool, limit: int | None, org_limit: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    url = "https://huggingface.co/api/datasets/open-llm-leaderboard/results/tree/main"
    root_tree = client.get_json(url, path=RICH_RAW / "openllm" / "root_tree.json", overwrite=overwrite, timeout=120)
    org_dirs = [item["path"] for item in root_tree if item.get("type") == "directory"]
    priority_slugs = {slug(author) for author in HF_AUTHORS}
    priority_slugs.update({"qwen", "01-ai", "meta-llama", "deepseek-ai", "mistralai", "microsoft", "google", "tiiuae", "eleutherai", "allenai"})
    priority = [org for org in org_dirs if slug(org) in priority_slugs]
    remainder = [org for org in org_dirs if org not in priority]
    selected_orgs = (priority + remainder)[:org_limit]
    json_paths = []
    for org in selected_orgs:
        org_url = f"https://huggingface.co/api/datasets/open-llm-leaderboard/results/tree/main/{quote(org, safe='')}?recursive=true"
        org_path = RICH_RAW / "openllm" / "org_trees" / f"{slug(org)}.json"
        try:
            org_tree = client.get_json(org_url, path=org_path, overwrite=overwrite, timeout=120)
        except Exception as exc:
            write_json(RICH_RAW / "openllm" / "org_trees" / f"{slug(org)}_error.json", {"url": org_url, "error": str(exc)})
            continue
        json_paths.extend([item["path"] for item in org_tree if item.get("type") == "file" and item.get("path", "").endswith(".json")])
    by_model: dict[str, list[str]] = defaultdict(list)
    for path in json_paths:
        model_key = path.rsplit("/", 1)[0]
        by_model[model_key].append(path)
    latest = []
    for model_key, paths in by_model.items():
        latest.append(sorted(paths)[-1])
    latest = sorted(latest)
    if limit:
        latest = latest[:limit]
    result_rows = []
    metric_rows = []
    for rel_path in latest:
        url = f"https://huggingface.co/datasets/open-llm-leaderboard/results/resolve/main/{quote(rel_path, safe='/')}"
        local = RICH_RAW / "openllm" / rel_path
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            if local.exists() and not overwrite:
                payload = read_json(local)
            else:
                response = client.session.get(url, timeout=90)
                response.raise_for_status()
                payload = response.json()
                write_json(local, payload)
                time.sleep(client.sleep)
        except Exception as exc:
            result_rows.append({"model_path": rel_path.rsplit("/", 1)[0], "result_path": rel_path, "error": str(exc), "source_id": "open_llm_leaderboard_results"})
            continue
        model_path = rel_path.rsplit("/", 1)[0]
        config = payload.get("config") or {}
        results = payload.get("results") or {}
        model_name = config.get("model_name") or model_path.split("/", 1)[-1]
        result_rows.append(
            {
                "model_path": model_path,
                "model_name": model_name,
                "author": model_path.split("/", 1)[0],
                "result_path": rel_path,
                "result_timestamp": re.search(r"results_(.*?)\.json$", rel_path).group(1) if re.search(r"results_(.*?)\.json$", rel_path) else "",
                "source_url": url,
                "source_id": "open_llm_leaderboard_results",
                "error": "",
            }
        )
        for benchmark, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                metric_rows.append(
                    {
                        "model_path": model_path,
                        "model_name": model_name,
                        "benchmark": benchmark,
                        "metric": metric,
                        "value": safe_float(value),
                        "raw_value": clean_text(value),
                        "source_url": url,
                        "source_id": "open_llm_leaderboard_results",
                    }
                )
    return pd.DataFrame(result_rows), pd.DataFrame(metric_rows)


def fetch_openalex(client: Client, overwrite: bool, per_query: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    works_rows = {}
    author_rows = []
    concept_rows = []
    mention_rows = []
    per_page = min(200, per_query)
    for query in OPENALEX_QUERIES:
        fetched = 0
        cursor = "*"
        page = 0
        while fetched < per_query:
            page += 1
            url = (
                "https://api.openalex.org/works"
                f"?search={quote(query)}"
                f"&filter=from_publication_date:2018-01-01,to_publication_date:{REFERENCE_DATE}"
                f"&per-page={per_page}&cursor={quote(cursor)}"
            )
            path = RICH_RAW / "openalex" / f"{slug(query)}_{page}.json"
            try:
                payload = client.get_json(url, path=path, overwrite=overwrite)
            except Exception as exc:
                write_json(RICH_RAW / "openalex" / f"{slug(query)}_{page}_error.json", {"url": url, "error": str(exc)})
                break
            results = payload.get("results") or []
            if not results:
                break
            for work in results:
                work_id = clean_text(work.get("id"))
                abstract = reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
                works_rows[work_id] = {
                    "work_id": work_id,
                    "doi": clean_text(work.get("doi")),
                    "title": clean_text(work.get("title") or work.get("display_name")),
                    "publication_date": clean_text(work.get("publication_date")),
                    "publication_year": work.get("publication_year"),
                    "type": clean_text(work.get("type")),
                    "cited_by_count": safe_float(work.get("cited_by_count")),
                    "is_open_access": (work.get("open_access") or {}).get("is_oa"),
                    "oa_status": clean_text((work.get("open_access") or {}).get("oa_status")),
                    "primary_location_source": clean_text(((work.get("primary_location") or {}).get("source") or {}).get("display_name")),
                    "query": query if work_id not in works_rows else f"{works_rows[work_id].get('query')}|{query}",
                    "abstract": abstract[:5000],
                    "source_url": clean_text(work.get("id")),
                    "source_id": "openalex_works_api",
                }
                for authorship in work.get("authorships") or []:
                    author = authorship.get("author") or {}
                    institutions = authorship.get("institutions") or []
                    author_rows.append(
                        {
                            "work_id": work_id,
                            "author_id": clean_text(author.get("id")),
                            "author_name": clean_text(author.get("display_name")),
                            "author_position": clean_text(authorship.get("author_position")),
                            "countries": ",".join(authorship.get("countries") or []),
                            "institution_ids": ",".join(clean_text(i.get("id")) for i in institutions),
                            "institution_names": ",".join(clean_text(i.get("display_name")) for i in institutions),
                            "source_id": "openalex_works_api",
                        }
                    )
                for concept in work.get("concepts") or []:
                    concept_rows.append(
                        {
                            "work_id": work_id,
                            "concept_id": clean_text(concept.get("id")),
                            "concept_name": clean_text(concept.get("display_name")),
                            "level": concept.get("level"),
                            "score": safe_float(concept.get("score")),
                            "source_id": "openalex_works_api",
                        }
                    )
                mention_text = f"{work.get('title') or ''}\n{abstract}"
                for family, pattern in MODEL_MENTION_PATTERNS.items():
                    for match in sorted(set(re.findall(pattern, mention_text, flags=re.I))):
                        mention_rows.append({"work_id": work_id, "family": family, "mention": match if isinstance(match, str) else match[0], "source_id": "derived_regex_mentions"})
            fetched += len(results)
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
    return pd.DataFrame(works_rows.values()), pd.DataFrame(author_rows).drop_duplicates(), pd.DataFrame(concept_rows).drop_duplicates(), pd.DataFrame(mention_rows).drop_duplicates()


def reconstruct_openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions: dict[int, str] = {}
    for word, locs in index.items():
        for loc in locs:
            positions[int(loc)] = word
    return " ".join(positions[i] for i in sorted(positions))


def fetch_github_repos(client: Client, overwrite: bool, pages_per_query: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repos = {}
    topic_rows = []
    mention_rows = []
    for query in GITHUB_QUERIES:
        for page in range(1, pages_per_query + 1):
            url = f"https://api.github.com/search/repositories?q={quote(query)}&sort=stars&order=desc&per_page=100&page={page}"
            path = RICH_RAW / "github" / f"{slug(query)}_{page}.json"
            try:
                payload = client.get_json(url, path=path, overwrite=overwrite)
            except Exception as exc:
                write_json(RICH_RAW / "github" / f"{slug(query)}_{page}_error.json", {"url": url, "error": str(exc)})
                break
            items = payload.get("items") or []
            if not items:
                break
            for item in items:
                full_name = clean_text(item.get("full_name"))
                license_obj = item.get("license") or {}
                repos[full_name] = {
                    "full_name": full_name,
                    "owner": full_name.split("/", 1)[0] if "/" in full_name else "",
                    "name": clean_text(item.get("name")),
                    "html_url": clean_text(item.get("html_url")),
                    "description": clean_text(item.get("description")),
                    "language": clean_text(item.get("language")),
                    "stars": safe_float(item.get("stargazers_count")),
                    "forks": safe_float(item.get("forks_count")),
                    "watchers": safe_float(item.get("watchers_count")),
                    "open_issues": safe_float(item.get("open_issues_count")),
                    "license_spdx": clean_text(license_obj.get("spdx_id")),
                    "created_at": clean_text(item.get("created_at")),
                    "updated_at": clean_text(item.get("updated_at")),
                    "pushed_at": clean_text(item.get("pushed_at")),
                    "query": query if full_name not in repos else f"{repos[full_name].get('query')}|{query}",
                    "source_url": clean_text(item.get("html_url")),
                    "source_id": "github_search_api",
                }
                for topic in item.get("topics") or []:
                    topic_rows.append({"full_name": full_name, "topic": topic, "topic_slug": slug(topic), "source_id": "github_search_api"})
                mention_text = f"{item.get('full_name') or ''}\n{item.get('description') or ''}\n{' '.join(item.get('topics') or [])}"
                for family, pattern in MODEL_MENTION_PATTERNS.items():
                    for match in sorted(set(re.findall(pattern, mention_text, flags=re.I))):
                        mention_rows.append({"full_name": full_name, "family": family, "mention": match if isinstance(match, str) else match[0], "source_id": "derived_regex_mentions"})
    return pd.DataFrame(repos.values()), pd.DataFrame(topic_rows).drop_duplicates(), pd.DataFrame(mention_rows).drop_duplicates()


def load_swebench_verified_instances(overwrite: bool = False) -> list[str]:
    url = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet"
    path = download(url, RICH_RAW / "swebench" / "verified_instances.parquet", overwrite=overwrite).path
    frame = pd.read_parquet(path)
    if "instance_id" not in frame.columns:
        return []
    return sorted(frame["instance_id"].dropna().astype(str).unique())


def build_swebench_instance_outcomes(overwrite: bool = False) -> pd.DataFrame:
    rows = []
    all_instances = load_swebench_verified_instances(overwrite=overwrite)
    swe_dirs = sorted((DATA_RAW / "swebench").glob("*/results.json"))
    for result_path in swe_dirs:
        submission = result_path.parent.name
        match = re.match(r"(\d{8})_", submission)
        if match and match.group(1) > REFERENCE_DATE.replace("-", ""):
            continue
        payload = read_json(result_path)
        resolved = set(payload.get("resolved") or [])
        no_generation = set(payload.get("no_generation") or [])
        no_logs = set(payload.get("no_logs") or [])
        instances = all_instances or sorted(resolved | no_generation | no_logs)
        for instance_id in instances:
            rows.append(
                {
                    "submission": submission,
                    "instance_id": instance_id,
                    "repo": instance_id.rsplit("-", 1)[0] if "-" in instance_id else instance_id,
                    "resolved": instance_id in resolved,
                    "no_generation": instance_id in no_generation,
                    "no_logs": instance_id in no_logs,
                    "unresolved_or_not_listed": instance_id not in resolved and instance_id not in no_generation and instance_id not in no_logs,
                    "source_url": f"https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/{submission}",
                    "source_id": "swe_bench_verified",
                }
            )
    return pd.DataFrame(rows)


def build_unified_model_index(
    hf_models: pd.DataFrame,
    openrouter: pd.DataFrame,
    epoch: pd.DataFrame,
    openllm: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in hf_models.iterrows():
        rows.append(
            {
                "entity_id": f"hf::{row['model_id']}",
                "name": row["model_id"],
                "family": row.get("family"),
                "model_family": row.get("model_family", row.get("family")),
                "product_line": row.get("product_line"),
                "vendor_or_author": row.get("author"),
                "source_namespace": "huggingface",
                "access_class": row.get("access_class"),
                "release_or_created_at": row.get("created_at"),
                "downloads": row.get("downloads"),
                "likes": row.get("likes"),
                "parameters": row.get("parameters_inferred_from_id"),
                "source_url": row.get("source_url"),
            }
        )
    for _, row in openrouter.iterrows():
        rows.append(
            {
                "entity_id": row.get("model_id"),
                "name": row.get("canonical_model"),
                "family": row.get("family"),
                "model_family": row.get("model_family", row.get("family")),
                "product_line": row.get("product_line"),
                "vendor_or_author": row.get("vendor"),
                "source_namespace": "openrouter",
                "access_class": row.get("access_class"),
                "release_or_created_at": row.get("release_date"),
                "context_window": row.get("context_window"),
                "input_usd_per_1m": row.get("input_usd_per_1m"),
                "output_usd_per_1m": row.get("output_usd_per_1m"),
                "source_url": row.get("source_url"),
            }
        )
    for _, row in epoch.iterrows():
        rows.append(
            {
                "entity_id": row.get("model_id"),
                "name": row.get("canonical_model"),
                "family": row.get("family"),
                "model_family": row.get("model_family", row.get("family")),
                "product_line": row.get("product_line"),
                "vendor_or_author": row.get("vendor"),
                "source_namespace": "epoch",
                "access_class": row.get("access_class"),
                "release_or_created_at": row.get("release_date"),
                "parameters": row.get("parameters"),
                "training_compute_flop": row.get("training_compute_flop"),
                "source_url": row.get("source_url"),
            }
        )
    for _, row in openllm.iterrows():
        rows.append(
            {
                "entity_id": f"openllm::{row.get('model_path')}",
                "name": row.get("model_name"),
                "model_family": classify_family(str(row.get("model_name")), str(row.get("author"))),
                "family": classify_family(str(row.get("model_name")), str(row.get("author"))),
                "vendor_or_author": row.get("author"),
                "source_namespace": "open_llm_leaderboard",
                "source_url": row.get("source_url"),
            }
        )
    return pd.DataFrame(rows)


def write_dataset_docs(manifest_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    lines = [
        "# Rich Frontier AI Dataset",
        "",
        f"Captured at: `{CAPTURED_AT}`. Reference date: `{REFERENCE_DATE}`.",
        "",
        "This directory is the raw dataset layer for deeper analysis. It intentionally keeps broad, messy public ecosystem signals instead of reducing everything to one score.",
        "",
        "## Tables",
        "",
        manifest_df.sort_values("table").to_markdown(index=False),
        "",
        "## Source Families",
        "",
        source_df.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- No proprietary or paid data is used.",
        "- Missing public values are left blank.",
        "- `derived_regex_mentions` rows are weak joins: they record text mentions, not confirmed model usage.",
        "- OpenRouter prices are public catalog prices normalized to USD per 1M tokens.",
        "- Hugging Face file sizes come from repository tree metadata, not downloaded weights.",
    ]
    (DATASET_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def repair_missing_parquets(manifest_df: pd.DataFrame) -> pd.DataFrame:
    repaired = manifest_df.copy()
    for idx, row in repaired.iterrows():
        if clean_text(row.get("parquet_path")):
            continue
        csv_path = ROOT / clean_text(row.get("csv_path"))
        if not csv_path.exists():
            continue
        parquet_path = DATASET_DIR / f"{row['table']}.parquet"
        try:
            frame = pd.read_csv(csv_path)
            sanitize_for_parquet(frame).to_parquet(parquet_path, index=False)
            repaired.loc[idx, "parquet_path"] = str(parquet_path.relative_to(ROOT))
        except Exception:
            pass
    return repaired


def validate_curated_inputs(openrouter: pd.DataFrame, epoch: pd.DataFrame, swe_summary: pd.DataFrame, livebench_raw: pd.DataFrame) -> None:
    checks = {
        "models_openrouter_normalized.csv": (openrouter, 50),
        "models_epoch_normalized.csv": (epoch, 50),
        "benchmarks_swebench_verified.csv": (swe_summary, 1),
        "livebench_model_judgment_raw_preview.csv": (livebench_raw, 1),
    }
    failures = []
    for name, (frame, minimum_rows) in checks.items():
        if len(frame) < minimum_rows:
            failures.append(f"{name} has {len(frame):,} rows; expected at least {minimum_rows:,} for rich dataset generation")
        if "source_id" in frame.columns and frame["source_id"].astype(str).str.startswith("sample").any():
            failures.append(f"{name} contains sample source_id rows")
        source_urls = frame["source_url"].astype(str) if "source_url" in frame.columns else pd.Series([], dtype=str)
        if not source_urls.empty and source_urls.str.startswith("sample://").any():
            failures.append(f"{name} contains sample:// source URLs")
    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(f"Refusing to build rich dataset from non-full curated inputs: {detail}")


def build_dataset(overwrite: bool = False, hf_tree_top_n: int = 220, hf_per_author: int = 100, hf_per_query: int = 120, openalex_per_query: int = 300, github_pages: int = 2, openllm_limit: int | None = None, openllm_org_limit: int = 180) -> None:
    ensure_dirs()
    client = Client()
    manifest: list[dict[str, Any]] = []
    sources = pd.DataFrame(
        [
            {"source_id": "huggingface_models_api", "name": "Hugging Face public models API", "url": "https://huggingface.co/api/models", "type": "public_api"},
            {"source_id": "huggingface_model_tree_api", "name": "Hugging Face model repository tree API", "url": "https://huggingface.co/api/models/{model}/tree/main", "type": "public_api"},
            {"source_id": "lmarena_full_dataset", "name": "LMArena full leaderboard parquet snapshots", "url": "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset", "type": "public_dataset"},
            {"source_id": "open_llm_leaderboard_results", "name": "Hugging Face Open LLM Leaderboard result JSON files", "url": "https://huggingface.co/datasets/open-llm-leaderboard/results", "type": "public_dataset"},
            {"source_id": "openalex_works_api", "name": "OpenAlex works search API", "url": "https://api.openalex.org/works", "type": "public_api"},
            {"source_id": "github_search_api", "name": "GitHub repository search API", "url": "https://api.github.com/search/repositories", "type": "public_api"},
            {"source_id": "swe_bench_verified", "name": "SWE-bench Verified public experiment repository", "url": "https://github.com/SWE-bench/experiments", "type": "public_dataset"},
            {"source_id": "derived_regex_mentions", "name": "Regex-derived model mentions from public text metadata", "url": "", "type": "derived"},
        ]
    )
    write_table(sources, "source_registry_rich", ["internal"], manifest)

    print("Loading existing curated tables...", flush=True)
    openrouter = pd.read_csv(DATA_PROCESSED / "models_openrouter_normalized.csv")
    epoch = pd.read_csv(DATA_PROCESSED / "models_epoch_normalized.csv")
    swe_summary = pd.read_csv(DATA_PROCESSED / "benchmarks_swebench_verified.csv")
    livebench_raw = pd.read_csv(DATA_PROCESSED / "livebench_model_judgment_raw_preview.csv")
    validate_curated_inputs(openrouter, epoch, swe_summary, livebench_raw)
    write_table(openrouter, "openrouter_models_catalog", ["openrouter_models_api"], manifest)
    write_table(epoch, "epoch_models_normalized", ["epoch_ai_models"], manifest)
    write_table(swe_summary, "swebench_submissions", ["swe_bench_verified"], manifest)
    write_table(livebench_raw, "livebench_judgments", ["livebench_model_judgment"], manifest)

    print("Fetching Hugging Face model ecosystem metadata...", flush=True)
    hf_models, hf_tags = fetch_huggingface_models(client, overwrite, hf_per_author, hf_per_query)
    hf_files = fetch_hf_model_files(client, hf_models, overwrite, hf_tree_top_n)
    hf_rollups = build_hf_model_rollups(hf_models, hf_files, hf_tags)
    write_table(hf_models, "huggingface_models", ["huggingface_models_api"], manifest)
    write_table(hf_tags, "huggingface_model_tags", ["huggingface_models_api"], manifest)
    write_table(hf_files, "huggingface_model_files", ["huggingface_model_tree_api"], manifest)
    write_table(hf_rollups, "huggingface_model_rollups", ["huggingface_models_api", "huggingface_model_tree_api"], manifest)

    print("Fetching full LMArena snapshots...", flush=True)
    lmarena_full = fetch_lmarena_full(overwrite)
    write_table(lmarena_full, "lmarena_full", ["lmarena_full_dataset"], manifest)

    print("Fetching Open LLM Leaderboard result JSONs...", flush=True)
    openllm_results, openllm_metrics = fetch_openllm_results(client, overwrite, openllm_limit, org_limit=openllm_org_limit)
    write_table(openllm_results, "openllm_leaderboard_results", ["open_llm_leaderboard_results"], manifest)
    write_table(openllm_metrics, "openllm_leaderboard_metrics_long", ["open_llm_leaderboard_results"], manifest)

    print("Fetching OpenAlex AI paper metadata...", flush=True)
    works, authors, concepts, paper_mentions = fetch_openalex(client, overwrite, openalex_per_query)
    write_table(works, "openalex_ai_papers", ["openalex_works_api"], manifest)
    write_table(authors, "openalex_ai_paper_authorships", ["openalex_works_api"], manifest)
    write_table(concepts, "openalex_ai_paper_concepts", ["openalex_works_api"], manifest)
    write_table(paper_mentions, "openalex_model_mentions", ["openalex_works_api", "derived_regex_mentions"], manifest)

    print("Fetching GitHub AI repository ecosystem metadata...", flush=True)
    repos, repo_topics, repo_mentions = fetch_github_repos(client, overwrite, github_pages)
    write_table(repos, "github_ai_repositories", ["github_search_api"], manifest)
    write_table(repo_topics, "github_ai_repository_topics", ["github_search_api"], manifest)
    write_table(repo_mentions, "github_model_mentions", ["github_search_api", "derived_regex_mentions"], manifest)

    print("Building SWE-bench instance-level outcomes...", flush=True)
    swe_instances = build_swebench_instance_outcomes(overwrite=overwrite)
    write_table(swe_instances, "swebench_instance_outcomes", ["swe_bench_verified"], manifest)

    print("Building unified model index...", flush=True)
    unified = build_unified_model_index(hf_rollups, openrouter, epoch, openllm_results)
    write_table(unified, "unified_model_index", ["huggingface_models_api", "openrouter_models_api", "epoch_ai_models", "open_llm_leaderboard_results"], manifest)

    manifest_df = pd.DataFrame(manifest).drop_duplicates("table", keep="last").sort_values("table")
    manifest_df = repair_missing_parquets(manifest_df)
    manifest_df.to_csv(DATASET_DIR / "dataset_manifest.csv", index=False)
    try:
        manifest_df.to_parquet(DATASET_DIR / "dataset_manifest.parquet", index=False)
    except Exception:
        pass
    write_dataset_docs(manifest_df, sources)
    write_run_manifest("dataset", DATASET_DIR, list(DATASET_DIR.glob("*.csv")), upstream_manifest=DATA_PROCESSED / "run_manifest.json")
    print(f"Done. Rich dataset tables written to {DATASET_DIR}", flush=True)
    print(manifest_df[["table", "rows", "columns"]].sort_values("rows", ascending=False).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a wide public dataset package for frontier AI analysis.")
    parser.add_argument("--overwrite", action="store_true", help="Refresh raw public snapshots.")
    parser.add_argument("--hf-tree-top-n", type=int, default=220)
    parser.add_argument("--hf-per-author", type=int, default=100)
    parser.add_argument("--hf-per-query", type=int, default=120)
    parser.add_argument("--openalex-per-query", type=int, default=300)
    parser.add_argument("--github-pages", type=int, default=2)
    parser.add_argument("--openllm-limit", type=int, default=None)
    parser.add_argument("--openllm-org-limit", type=int, default=180)
    args = parser.parse_args()
    build_dataset(
        overwrite=args.overwrite,
        hf_tree_top_n=args.hf_tree_top_n,
        hf_per_author=args.hf_per_author,
        hf_per_query=args.hf_per_query,
        openalex_per_query=args.openalex_per_query,
        github_pages=args.github_pages,
        openllm_limit=args.openllm_limit,
        openllm_org_limit=args.openllm_org_limit,
    )


if __name__ == "__main__":
    main()
