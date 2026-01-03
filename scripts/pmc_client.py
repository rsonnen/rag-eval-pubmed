"""PubMed Central API client for E-utilities search and OAI-PMH retrieval.

Provides two main capabilities:
1. E-utilities (esearch) - Search PMC by MeSH terms to find PMCIDs
2. OAI-PMH (GetRecord) - Retrieve full-text JATS XML for specific PMCIDs

Rate limiting:
- E-utilities: 3 req/sec without API key, 10 req/sec with NCBI_API_KEY
- OAI-PMH: Sequential requests only, run outside peak hours (Mon-Fri 5am-9pm ET)
"""

import logging
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from Bio import Entrez
from sickle import Sickle
from sickle.oaiexceptions import BadArgument, IdDoesNotExist, NoRecordsMatch

logger = logging.getLogger(__name__)

PMC_OAI_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"

EUTILS_DELAY_NO_KEY = 0.35
EUTILS_DELAY_WITH_KEY = 0.11
OAI_DELAY = 1.0
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
MAX_BACKOFF = 120.0


@dataclass
class SearchResult:
    """Result from an E-utilities search."""

    pmcids: list[str]
    total_count: int
    query_key: str
    web_env: str


@dataclass
class ArticleRecord:
    """A record retrieved from OAI-PMH."""

    pmcid: str
    xml_content: str
    datestamp: str


def _get_delay() -> float:
    """Get the appropriate delay based on whether an API key is configured."""
    if os.environ.get("NCBI_API_KEY"):
        return EUTILS_DELAY_WITH_KEY
    return EUTILS_DELAY_NO_KEY


def _configure_entrez() -> None:
    """Configure Entrez with email and optional API key."""
    email = os.environ.get("NCBI_EMAIL", "")
    if not email:
        raise ValueError(
            "NCBI_EMAIL environment variable is required for E-utilities. "
            "Set it to your email address for NCBI contact purposes."
        )

    # Bio.Entrez module attributes are dynamically typed at runtime.
    # The Entrez module doesn't have type stubs - these assignments are valid.
    Entrez.email = email  # type: ignore[assignment]

    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key  # type: ignore[assignment]
        logger.info("Using NCBI API key (10 requests/second)")
    else:
        logger.info("No NCBI API key - limited to 3 requests/second")


class PMCClient:
    """Client for PubMed Central E-utilities and OAI-PMH APIs."""

    def __init__(self) -> None:
        """Initialize the PMC client with configured Entrez and Sickle."""
        _configure_entrez()
        self._sickle: Any = Sickle(PMC_OAI_URL)
        self._delay = _get_delay()

    def search_pmc(
        self,
        query: str,
        retmax: int = 500,
        retstart: int = 0,
        use_history: bool = True,
    ) -> SearchResult:
        """Search PMC using E-utilities esearch.

        Args:
            query: The search query (can include MeSH terms, filters).
            retmax: Maximum number of results to return per request.
            retstart: Starting index for pagination.
            use_history: Whether to use NCBI history server for large result sets.

        Returns:
            SearchResult with PMCIDs and pagination info.
        """
        delay = self._delay
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                jitter = random.uniform(0, delay * 0.1)  # noqa: S311
                sleep_time = delay + jitter
                logger.info(f"Retry {attempt}/{MAX_RETRIES}, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                delay = min(delay * BACKOFF_FACTOR, MAX_BACKOFF)
            else:
                time.sleep(self._delay)

            try:
                # Entrez.esearch and Entrez.read are untyped in biopython
                handle: Any = Entrez.esearch(  # type: ignore[no-untyped-call]
                    db="pmc",
                    term=query,
                    retmax=retmax,
                    retstart=retstart,
                    usehistory="y" if use_history else "n",
                )
                results: Any = Entrez.read(handle)  # type: ignore[no-untyped-call]
                handle.close()

                pmcids = [str(pmcid) for pmcid in results.get("IdList", [])]
                total = int(results.get("Count", 0))
                query_key = str(results.get("QueryKey", ""))
                web_env = str(results.get("WebEnv", ""))

                return SearchResult(
                    pmcids=pmcids,
                    total_count=total,
                    query_key=query_key,
                    web_env=web_env,
                )

            except Exception as e:
                last_error = e
                logger.warning(f"E-utilities search failed: {e}")
                continue

        if last_error:
            raise last_error
        raise RuntimeError("All search retries exhausted")

    def iter_search_results(
        self,
        query: str,
        batch_size: int = 500,
        max_results: int | None = None,
    ) -> Iterator[str]:
        """Iterate through all PMCIDs matching a query.

        Args:
            query: The search query.
            batch_size: Number of results to fetch per request.
            max_results: Maximum total results to return (None = all).

        Yields:
            PMCID strings (numeric, without PMC prefix).
        """
        retstart = 0
        yielded = 0

        while True:
            result = self.search_pmc(
                query=query,
                retmax=batch_size,
                retstart=retstart,
            )

            if not result.pmcids:
                break

            for pmcid in result.pmcids:
                yield pmcid
                yielded += 1

                if max_results and yielded >= max_results:
                    return

            retstart += len(result.pmcids)

            if retstart >= result.total_count:
                break

    def get_article_xml(self, pmcid: str) -> ArticleRecord | None:
        """Retrieve full-text JATS XML for a specific article via OAI-PMH.

        Args:
            pmcid: The PMC ID (numeric only, without "PMC" prefix).

        Returns:
            ArticleRecord with XML content, or None if not available.
        """
        pmcid = pmcid.lstrip("PMC")
        identifier = f"oai:pubmedcentral.nih.gov:{pmcid}"

        delay = OAI_DELAY
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                jitter = random.uniform(0, delay * 0.1)  # noqa: S311
                sleep_time = delay + jitter
                logger.info(f"Retry {attempt}/{MAX_RETRIES}, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                delay = min(delay * BACKOFF_FACTOR, MAX_BACKOFF)
            else:
                time.sleep(OAI_DELAY)

            try:
                record: Any = self._sickle.GetRecord(
                    identifier=identifier,
                    metadataPrefix="pmc",
                )

                raw_xml: str = record.raw
                if not raw_xml:
                    logger.warning(f"Empty XML for PMC{pmcid}")
                    return None

                datestamp = ""
                if hasattr(record, "header") and hasattr(record.header, "datestamp"):
                    datestamp = str(record.header.datestamp)

                return ArticleRecord(
                    pmcid=f"PMC{pmcid}",
                    xml_content=raw_xml,
                    datestamp=datestamp,
                )

            except IdDoesNotExist:
                logger.warning(f"PMC{pmcid} does not exist in OAI-PMH")
                return None
            except NoRecordsMatch:
                logger.warning(f"No records match for PMC{pmcid}")
                return None
            except BadArgument as e:
                logger.warning(f"Bad argument for PMC{pmcid}: {e}")
                return None
            except Exception as e:
                last_error = e
                logger.warning(f"OAI-PMH request failed for PMC{pmcid}: {e}")
                continue

        if last_error:
            logger.error(f"All retries exhausted for PMC{pmcid}: {last_error}")
        return None

    def build_mesh_query(
        self,
        mesh_terms: list[str],
        filters: list[str] | None = None,
        date_range: tuple[int, int] | None = None,
    ) -> str:
        """Build a PMC search query from MeSH terms and filters.

        Args:
            mesh_terms: List of MeSH terms (can include field tags).
            filters: Additional filters (e.g., "open access"[filter]).
            date_range: Optional (start_year, end_year) tuple.

        Returns:
            Formatted query string for esearch.
        """
        parts: list[str] = []

        if len(mesh_terms) == 1:
            parts.append(mesh_terms[0])
        elif mesh_terms:
            mesh_query = " OR ".join(mesh_terms)
            parts.append(f"({mesh_query})")

        if filters:
            parts.extend(filters)

        if date_range:
            start_year, end_year = date_range
            parts.append(f"{start_year}:{end_year}[pdat]")

        return " AND ".join(parts)
