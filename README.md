# rag-eval-pubmed

Evaluation corpus of open access biomedical papers from PubMed Central for testing RAG systems.

## What This Is

This repository contains **evaluation data for RAG systems**:

- **corpus.yaml** - Evaluation scenarios (in each corpus directory)
- **metadata.json** - Paper inventory with PMC IDs
- **Generated questions** - Validated Q/A pairs (where available)

The actual JATS XML papers are not included. Use `download_papers.py` to fetch them from PMC.

## Quick Start

```bash
cd scripts
uv sync
uv run python download_papers.py oncology --max-docs 10
```

## Available Corpora

| Corpus | Papers | Description |
|--------|--------|-------------|
| `oncology` | 300 | Cancer research and medical oncology |
| `cardiology` | 300 | Heart and cardiovascular diseases |
| `infectious_disease` | 300 | Communicable diseases and infection |
| `neurology` | 300 | Nervous system diseases |
| `pharmacology` | 300 | Drug therapy and pharmacology |
| `epidemiology` | 300 | Public health and epidemiology |
| `genetics` | 300 | Genetics and genomics |
| `psychiatry` | 300 | Mental disorders and psychiatry |

All corpora were built December 2025 from PMC Open Access subset.

## Directory Structure

```
<corpus>/
    corpus.yaml         # Evaluation configuration
    metadata.json       # Paper inventory
    papers/             # JATS XML files (gitignored)

scripts/
    download_papers.py  # Fetch papers from existing metadata
    build_pubmed.py     # Build new corpora with LLM curation
    pmc_client.py       # PMC API client
    jats_parser.py      # JATS XML parsing

corpus_specs/
    *.yaml              # Build configurations
```

## Metadata Format

```json
{
  "corpus": "oncology",
  "source": "pubmed_central",
  "search_strategy": {...},
  "curated_at": "2025-12-27T18:59:22.943763+00:00",
  "total_papers": 300,
  "papers_evaluated": 430,
  "acceptance_rate": 0.70,
  "papers": [
    {
      "pmcid": "PMC12728525",
      "title": "A Novel Cuproptosis-Related Gene Signature...",
      "authors": ["Jun Cao", "Shijia Zhang", "..."],
      "journal": "Combinatorial Chemistry & High Throughput Screening",
      "pub_date": "2024-08-27",
      "keywords": ["thyroid cancer", "cuproptosis", "..."],
      "license": "https://creativecommons.org/licenses/by/4.0/",
      "file": "papers/PMC12728525.xml"
    }
  ]
}
```

## Downloading Papers

The download script fetches JATS XML from PMC via OAI-PMH:

```bash
cd scripts
uv run python download_papers.py oncology --max-docs 50
uv run python download_papers.py cardiology
```

| Option | Description |
|--------|-------------|
| `corpus` | Corpus name (e.g., oncology) |
| `--max-docs` | Maximum papers to download (default: all) |
| `--delay` | Delay between requests in seconds (default: 0.5) |

## Building New Corpora

The build script searches PMC and curates using LLM evaluation. Requires `NCBI_EMAIL` and `OPENAI_API_KEY` environment variables.

```bash
cd scripts
uv run python build_pubmed.py \
    --config ../corpus_specs/oncology.yaml \
    --corpus oncology
```

## Licensing

**This repository**: MIT License

**Papers**: PMC Open Access papers have varying licenses (CC0, CC BY, CC BY-NC, etc.). The license for each paper is recorded in `metadata.json`.
