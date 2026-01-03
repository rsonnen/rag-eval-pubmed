#!/usr/bin/env python
"""Build curated PubMed Central open access corpora using LLM-evaluated filtering.

Searches PMC by MeSH terms using E-utilities, retrieves full-text JATS XML via
OAI-PMH, evaluates each paper for relevance using an LLM-as-judge, and builds
a quality corpus of on-topic biomedical papers.

Uses cursor-based iteration with persistent state for proper resume capability.
Can resume at exact position: which MeSH term, which page within that search.

Usage:
    uv run python build_pubmed.py \
        --config ../corpus_specs/oncology.yaml \
        --corpus oncology \
        --data-dir /path/to/output
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, ValidationError
from tqdm import tqdm

from jats_parser import (
    JATSParseError,
    ParsedArticle,
    article_to_evaluation_dict,
    parse_jats_xml,
)
from pmc_client import PMCClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_INTERVAL = 10
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0


@dataclass
class SearchCursor:
    """Tracks position in PMC search iteration for resume capability.

    The search proceeds through MeSH terms in order, with pagination
    within each term.
    """

    term_index: int
    retstart: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"term_index": self.term_index, "retstart": self.retstart}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchCursor":
        return cls(
            term_index=data["term_index"],
            retstart=data.get("retstart", 0),
        )


@dataclass
class BuildState:
    """Persistent state for corpus building with full resume capability."""

    corpus_name: str
    cursor: SearchCursor
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    processed_ids: set[str] = field(default_factory=set)
    total_evaluated: int = 0

    def save(self, state_path: Path) -> None:
        """Save state to disk atomically."""
        data = {
            "corpus_name": self.corpus_name,
            "cursor": self.cursor.to_dict(),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "processed_ids": list(self.processed_ids),
            "total_evaluated": self.total_evaluated,
        }
        tmp_path = state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.rename(state_path)

    @classmethod
    def load(cls, state_path: Path) -> "BuildState | None":
        """Load state from disk, or return None if not found."""
        if not state_path.exists():
            return None
        try:
            with state_path.open(encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                corpus_name=data["corpus_name"],
                cursor=SearchCursor.from_dict(data["cursor"]),
                accepted=data.get("accepted", []),
                rejected=data.get("rejected", []),
                processed_ids=set(data.get("processed_ids", [])),
                total_evaluated=data.get("total_evaluated", 0),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load state: {e}")
            return None


class RelevanceEvaluation(BaseModel):
    """Structured output from the LLM relevance evaluation."""

    relevant: bool = Field(description="Whether the paper belongs in this corpus")
    confidence: float = Field(
        description="Confidence in the decision (0.0 to 1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the decision")


EVALUATION_PROMPT = """
You are evaluating whether a scientific paper belongs in a specific biomedical corpus.

{validation_prompt}

PAPER INFORMATION:
PMC ID: {pmcid}
Title: {title}
Authors: {authors}
Journal: {journal}
Publication Date: {pub_date}
Keywords: {keywords}
Subjects: {subjects}
License: {license}

Abstract:
{abstract}

Body excerpt (first ~4000 characters):
{body_excerpt}

EVALUATION TASK:
Based on the corpus requirements above and the paper information provided,
determine if this paper belongs in the corpus.

Respond with:
- relevant: true/false
- confidence: your confidence in this decision (0.0 to 1.0)
- reasoning: brief explanation (1-2 sentences)"""


def create_evaluator(
    model_name: str = "gpt-5-mini",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Create an LLM instance for paper evaluation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    base_url = os.environ.get("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )


def evaluate_paper(
    article: ParsedArticle,
    validation_prompt: str,
    llm: ChatOpenAI,
    confidence_threshold: float = 0.7,
) -> RelevanceEvaluation | None:
    """Evaluate whether a paper is relevant to the corpus topic.

    Returns None if LLM call fails.
    """
    eval_data = article_to_evaluation_dict(article)

    prompt = EVALUATION_PROMPT.format(
        validation_prompt=validation_prompt,
        **eval_data,
    )

    delay = 1.0
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            jitter = random.uniform(0, delay * 0.1)  # noqa: S311
            sleep_time = delay + jitter
            logger.info(f"LLM retry {attempt}/{MAX_RETRIES}, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
            delay = min(delay * BACKOFF_FACTOR, 30.0)

        try:
            structured_llm = llm.with_structured_output(RelevanceEvaluation)
            raw_result = structured_llm.invoke(prompt)

            if raw_result is None:
                logger.warning(f"LLM returned None for {article.pmcid}")
                return None

            result = cast(RelevanceEvaluation, raw_result)

            if result.relevant and result.confidence < confidence_threshold:
                return RelevanceEvaluation(
                    relevant=False,
                    confidence=result.confidence,
                    reasoning=f"Below threshold ({result.confidence:.2f}). "
                    f"{result.reasoning}",
                )

            return result

        except ValidationError as e:
            logger.warning(f"Validation error for {article.pmcid}: {e}")
            return None
        except Exception as e:
            last_error = e
            logger.warning(f"LLM evaluation failed for {article.pmcid}: {e}")
            continue

    if last_error:
        logger.error(f"All LLM retries exhausted for {article.pmcid}: {last_error}")
    return None


def load_corpus_config(config_path: Path, corpus_name: str) -> dict[str, Any]:
    """Load corpus configuration from YAML file."""
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if corpus_name not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Corpus '{corpus_name}' not found. Available: {available}")

    corpus = config[corpus_name]

    required = ["source", "search_strategy", "target_count", "validation_prompt"]
    for req_field in required:
        if req_field not in corpus:
            raise ValueError(f"Corpus config missing required field: {req_field}")

    if corpus["source"] != "pubmed":
        raise ValueError(
            f"Corpus '{corpus_name}' has source '{corpus['source']}', "
            "not 'pubmed'. Use build_gutenberg.py or build_archive.py instead."
        )

    return cast(dict[str, Any], corpus)


def write_xml_file(xml_content: str, output_path: Path) -> bool:
    """Write XML content to file atomically."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".tmp")
        tmp_path.write_text(xml_content, encoding="utf-8")
        tmp_path.rename(output_path)
        return True
    except OSError as e:
        logger.error(f"Failed to write {output_path}: {e}")
        return False


def write_final_metadata(
    corpus_dir: Path,
    corpus_name: str,
    search_strategy: dict[str, Any],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    total_evaluated: int,
) -> None:
    """Write final corpus metadata file."""
    metadata = {
        "corpus": corpus_name,
        "source": "pubmed_central",
        "search_strategy": search_strategy,
        "curated_at": datetime.now(UTC).isoformat(),
        "total_papers": len(accepted),
        "papers_evaluated": total_evaluated,
        "acceptance_rate": len(accepted) / total_evaluated if total_evaluated else 0,
        "papers": accepted,
    }

    with (corpus_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if rejected:
        rejection_log = {
            "corpus": corpus_name,
            "total_rejected": len(rejected),
            "papers": rejected,
        }
        with (corpus_dir / "rejected.json").open("w", encoding="utf-8") as f:
            json.dump(rejection_log, f, indent=2, ensure_ascii=False)


def build_corpus(
    config_path: Path,
    corpus_name: str,
    data_dir: Path,
    limit: int | None = None,
    fresh: bool = False,
) -> None:
    """Build a curated corpus by searching PMC and evaluating papers.

    Uses cursor-based iteration with persistent state for proper resume.

    Args:
        config_path: Path to corpus config YAML.
        corpus_name: Name of corpus to build.
        data_dir: Base data directory for output.
        limit: Override target count (for testing).
        fresh: If True, ignore existing progress and start fresh.
    """
    corpus_config = load_corpus_config(config_path, corpus_name)

    corpus_dir = data_dir / corpus_name
    papers_dir = corpus_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    state_path = corpus_dir / "build_state.json"

    target_count = limit if limit is not None else corpus_config["target_count"]
    validation_prompt = corpus_config["validation_prompt"]
    search_strategy = corpus_config["search_strategy"]
    confidence_threshold = corpus_config.get("confidence_threshold", 0.7)
    evaluator_model = corpus_config.get("evaluator_model", "gpt-5-mini")

    state: BuildState | None = None
    if not fresh:
        state = BuildState.load(state_path)

    if state is None:
        state = BuildState(
            corpus_name=corpus_name,
            cursor=SearchCursor(term_index=0, retstart=0),
        )

    if len(state.accepted) >= target_count:
        logger.info(f"Target already reached: {len(state.accepted)}/{target_count}")
        return

    logger.info(f"Building corpus: {corpus_name}")
    logger.info(f"Target: {target_count} papers (have {len(state.accepted)})")
    logger.info(
        f"Resuming from: term[{state.cursor.term_index}] "
        f"retstart={state.cursor.retstart}"
    )

    client = PMCClient()
    llm = create_evaluator(model_name=evaluator_model)

    mesh_terms = search_strategy.get("mesh_terms", [])
    filters = search_strategy.get("filters", [])
    date_range_raw = search_strategy.get("date_range")
    date_range = tuple(date_range_raw) if date_range_raw else None

    papers_since_save = 0

    try:
        pbar = tqdm(desc="Evaluating", unit="paper")

        for term_idx in range(state.cursor.term_index, len(mesh_terms)):
            if len(state.accepted) >= target_count:
                break

            term = mesh_terms[term_idx]
            query = client.build_mesh_query(
                mesh_terms=[term],
                filters=filters,
                date_range=date_range,
            )

            logger.info(f"Searching: {term}")
            logger.info(f"Query: {query}")

            for pmcid in client.iter_search_results(query):
                if len(state.accepted) >= target_count:
                    break

                if pmcid in state.processed_ids:
                    continue

                pbar.set_postfix(
                    accepted=len(state.accepted),
                    evaluated=state.total_evaluated,
                )

                logger.info(f"Fetching PMC{pmcid}...")

                record = client.get_article_xml(pmcid)
                if record is None:
                    state.processed_ids.add(pmcid)
                    state.cursor = SearchCursor(term_index=term_idx)
                    continue

                try:
                    article = parse_jats_xml(record.xml_content)
                except JATSParseError as e:
                    logger.warning(f"Parse failed for PMC{pmcid}: {e}")
                    state.processed_ids.add(pmcid)
                    state.cursor = SearchCursor(term_index=term_idx)
                    continue

                logger.info(f"Evaluating: {article.title[:60]}...")

                evaluation = evaluate_paper(
                    article=article,
                    validation_prompt=validation_prompt,
                    llm=llm,
                    confidence_threshold=confidence_threshold,
                )

                if evaluation is None:
                    state.processed_ids.add(pmcid)
                    state.cursor = SearchCursor(term_index=term_idx)
                    continue

                state.processed_ids.add(pmcid)
                state.cursor = SearchCursor(term_index=term_idx)
                state.total_evaluated += 1
                papers_since_save += 1
                pbar.update(1)

                paper_meta = {
                    "pmcid": article.pmcid,
                    "title": article.title,
                    "authors": article.authors,
                    "journal": article.journal,
                    "pub_date": article.pub_date,
                    "keywords": article.keywords,
                    "license": article.license_type,
                }

                if evaluation.relevant:
                    filename = f"{article.pmcid}.xml"
                    final_path = papers_dir / filename

                    if write_xml_file(record.xml_content, final_path):
                        paper_meta["file"] = f"papers/{filename}"
                        state.accepted.append(paper_meta)
                        logger.info(
                            f"  ACCEPT ({len(state.accepted)}/{target_count}) "
                            f"[{evaluation.confidence:.2f}]: "
                            f"{evaluation.reasoning[:50]}"
                        )
                    else:
                        logger.error(f"Failed to save {article.pmcid}")
                else:
                    state.rejected.append(
                        {
                            **paper_meta,
                            "rejection_reason": evaluation.reasoning,
                            "rejection_confidence": evaluation.confidence,
                        }
                    )
                    logger.info(
                        f"  REJECT [{evaluation.confidence:.2f}]: "
                        f"{evaluation.reasoning[:60]}"
                    )

                if papers_since_save >= SAVE_INTERVAL:
                    state.save(state_path)
                    papers_since_save = 0

        pbar.close()

        state.save(state_path)

        write_final_metadata(
            corpus_dir=corpus_dir,
            corpus_name=corpus_name,
            search_strategy=search_strategy,
            accepted=state.accepted,
            rejected=state.rejected,
            total_evaluated=state.total_evaluated,
        )

        rate = (
            len(state.accepted) / state.total_evaluated if state.total_evaluated else 0
        )

        logger.info("=" * 60)
        logger.info(f"Build complete: {len(state.accepted)}/{target_count} papers")
        logger.info(f"Evaluated: {state.total_evaluated}, Acceptance rate: {rate:.1%}")
        logger.info(f"Output: {corpus_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted - saving state...")
        state.save(state_path)
        raise


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build curated PubMed Central corpus with LLM evaluation",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to corpus config YAML"
    )
    parser.add_argument(
        "--corpus", type=str, required=True, help="Name of corpus to build"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Data directory path"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Override target count (for testing)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing progress and start fresh",
    )

    args = parser.parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if not os.environ.get("NCBI_EMAIL"):
        logger.error("NCBI_EMAIL environment variable not set")
        sys.exit(1)

    data_dir = args.data_dir or Path(".")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        build_corpus(
            config_path=args.config,
            corpus_name=args.corpus,
            data_dir=data_dir,
            limit=args.limit,
            fresh=args.fresh,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
