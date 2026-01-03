#!/usr/bin/env python3
"""Download papers from existing metadata.

Reads metadata.json and downloads JATS XML from PubMed Central via OAI-PMH.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

from sickle import Sickle
from sickle.oaiexceptions import BadArgument, IdDoesNotExist, NoRecordsMatch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PMC_OAI_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
OAI_DELAY = 1.0
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0


def download_article(sickle: Sickle, pmcid: str) -> str | None:
    """Download JATS XML for a single article via OAI-PMH.

    Args:
        sickle: Sickle OAI-PMH client.
        pmcid: PMC ID (with or without 'PMC' prefix).

    Returns:
        XML content as string, or None if not available.
    """
    pmcid_num = pmcid.lstrip("PMC")
    identifier = f"oai:pubmedcentral.nih.gov:{pmcid_num}"

    delay = OAI_DELAY

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            jitter = random.uniform(0, delay * 0.1)  # noqa: S311
            sleep_time = delay + jitter
            time.sleep(sleep_time)
            delay = min(delay * BACKOFF_FACTOR, 30.0)
        else:
            time.sleep(OAI_DELAY)

        try:
            record: Any = sickle.GetRecord(
                identifier=identifier,
                metadataPrefix="pmc",
            )
            raw_xml: str = record.raw
            if raw_xml:
                return raw_xml
            return None

        except (IdDoesNotExist, NoRecordsMatch, BadArgument):
            return None
        except Exception as e:
            logger.debug(f"OAI-PMH request failed for {pmcid}: {e}")
            continue

    return None


def download_corpus(
    corpus_dir: Path,
    delay: float = 1.0,
    max_docs: int | None = None,
) -> None:
    """Download papers listed in metadata.json.

    Args:
        corpus_dir: Corpus directory containing metadata.json.
        delay: Additional delay between downloads (on top of OAI_DELAY).
        max_docs: Maximum number of papers to download (None for all).
    """
    metadata_path = corpus_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    papers: list[dict[str, Any]] = metadata.get("papers", [])

    if max_docs is not None:
        papers = papers[:max_docs]

    print(f"Downloading {len(papers)} papers to {corpus_dir}")

    sickle: Any = Sickle(PMC_OAI_URL)
    failed = 0

    for paper in tqdm(papers, desc="Downloading", unit="paper"):
        pmcid: str = paper.get("pmcid", "")
        file_path: str = paper.get("file", "")

        if not pmcid or not file_path:
            continue

        output_path = corpus_dir / file_path
        if output_path.exists():
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        xml_content = download_article(sickle, pmcid)
        if xml_content:
            output_path.write_text(xml_content, encoding="utf-8")
        else:
            failed += 1
            tqdm.write(f"Failed: {pmcid}")

        time.sleep(delay)

    if failed:
        print(f"Done ({failed} failed)")
    else:
        print("Done")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download papers from existing metadata"
    )
    parser.add_argument("corpus", help="Corpus name (e.g., oncology)")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of papers to download (default: all)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    corpus_dir = repo_root / args.corpus

    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    download_corpus(corpus_dir, args.delay, args.max_docs)


if __name__ == "__main__":
    main()
