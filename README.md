# rag-eval-pubmed

Evaluation corpus of open access biomedical papers from PubMed Central for testing RAG (Retrieval-Augmented Generation) systems.

## What This Is

This repository contains **evaluation data for RAG systems**:

- **corpus.yaml** - Evaluation configuration defining domain context and testing scenarios
- **Generated questions** - Validated Q/A pairs for evaluation (where available)
- **metadata.json** - Paper inventory with PMC IDs
- **Build tools** - Scripts for curating new paper collections from PMC

The actual JATS XML papers are not included - individual papers have their own licenses. Use the build script to fetch them from PMC.

## Overview

This corpus builder creates domain-specific collections of biomedical research
papers from PMC's Open Access subset. Each paper is:

1. **Discovered** via NCBI E-utilities using MeSH term queries
2. **Retrieved** via PMC OAI-PMH as full-text JATS XML
3. **Evaluated** by an LLM for corpus relevance
4. **Saved** with structured metadata for RAG evaluation

## Available Corpora

| Corpus | MeSH Terms | Target |
|--------|-----------|--------|
| oncology | Neoplasms, Medical Oncology | 300 |
| cardiology | Heart Diseases, Cardiovascular Diseases | 300 |
| infectious_disease | Communicable Diseases, Infection | 300 |
| neurology | Nervous System Diseases, Neurosciences | 300 |
| pharmacology | Pharmacology, Drug Therapy | 300 |
| epidemiology | Epidemiology, Public Health | 300 |
| genetics | Genetics, Genomics | 300 |
| psychiatry | Psychiatry, Mental Disorders | 300 |

## Setup

### 1. Install Dependencies

```bash
cd scripts
uv sync --dev
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required variables:

- `OPENAI_API_KEY` - For LLM-based relevance evaluation
- `NCBI_EMAIL` - Required by NCBI for E-utilities access (your email address)
- `NCBI_API_KEY` - Optional but recommended (10 req/sec vs 3 req/sec)

Get an NCBI API key at: https://www.ncbi.nlm.nih.gov/account/settings/

### 3. Run Quality Checks

```bash
cd scripts
make all  # format, lint, security, typecheck
```

## Usage

### Build a Corpus

```bash
cd scripts
uv run python build_pubmed.py \
    --config ../corpus_specs/oncology.yaml \
    --corpus oncology \
    --data-dir /path/to/output
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--config` | Path to corpus YAML configuration |
| `--corpus` | Name of corpus to build (must match key in YAML) |
| `--data-dir` | Output directory for corpus (default: current dir) |
| `--limit` | Override target count (useful for testing) |
| `--fresh` | Ignore saved state, start from beginning |

### Test Run

Test with a small number of papers:

```bash
uv run python build_pubmed.py \
    --config ../corpus_specs/oncology.yaml \
    --corpus oncology \
    --limit 5
```

## Output Structure

```
oncology/
├── corpus.yaml         # Evaluation configuration
├── metadata.json       # Accepted papers with metadata
├── rejected.json       # Rejected papers (gitignored)
├── build_state.json    # Resume state (gitignored)
└── papers/             # Full JATS XML files (gitignored)
    ├── PMC9547123.xml
    ├── PMC9623456.xml
    └── ...

scripts/
├── build_pubmed.py     # Corpus builder (discovery + curation + download)
├── jats_parser.py      # JATS XML parsing utilities
└── pmc_client.py       # PMC API client
```

### Metadata Format

```json
{
  "corpus": "oncology",
  "source": "pubmed_central",
  "curated_at": "2024-01-15T10:30:00Z",
  "total_papers": 300,
  "papers_evaluated": 450,
  "acceptance_rate": 0.67,
  "papers": [
    {
      "pmcid": "PMC9547123",
      "title": "Novel Biomarkers in Breast Cancer",
      "authors": ["Smith J", "Jones A"],
      "journal": "Cancer Research",
      "pub_date": "2023-05-15",
      "keywords": ["breast cancer", "biomarkers"],
      "license": "CC BY",
      "file": "papers/PMC9547123.xml"
    }
  ]
}
```

## PMC Open Access Licensing

Papers in the PMC Open Access subset have varying licenses. The license for
each paper is recorded in `metadata.json`.

### Commercial Use Permitted

These licenses allow commercial use, modification, and redistribution:

- **CC0** (Public Domain Dedication) - No restrictions
- **CC BY** (Attribution) - Must give credit
- **CC BY-SA** (Attribution-ShareAlike) - Must give credit, derivatives same license
- **CC BY-ND** (Attribution-NoDerivatives) - Must give credit, no modifications

### Non-Commercial Only

These licenses restrict commercial use:

- **CC BY-NC** - Non-commercial use, must give credit
- **CC BY-NC-SA** - Non-commercial, attribution, share-alike
- **CC BY-NC-ND** - Non-commercial, attribution, no derivatives

### Other/Custom Licenses

Some papers have custom publisher licenses. Check the `license` field and the
original article for specific terms.

### Attribution Requirements

When using PMC Open Access content:

1. Cite the original article properly
2. Include the license type
3. For CC BY variants, give appropriate credit to authors
4. Do not imply endorsement by authors or publishers

### Usage Restrictions

**Permitted:**
- Text mining and computational analysis
- Building RAG evaluation datasets
- Academic research
- Non-commercial use (for all licenses)

**Prohibited:**
- Redistribution of full articles without license compliance
- Commercial use of NC-licensed content
- Modification of ND-licensed content
- Implying endorsement

## API Rate Limits

### NCBI E-utilities
- Without API key: 3 requests/second
- With API key: 10 requests/second
- Best practice: Run during off-peak hours

### PMC OAI-PMH
- No concurrent requests
- Run outside peak hours: Mon-Fri 5am-9pm Eastern Time
- Use HTTP compression (handled automatically)

## Resume Capability

The builder saves state every 10 papers. To resume after interruption:

```bash
# Just run the same command - it will resume automatically
uv run python build_pubmed.py \
    --config ../corpus_specs/oncology.yaml \
    --corpus oncology
```

To start fresh and ignore previous progress:

```bash
uv run python build_pubmed.py \
    --config ../corpus_specs/oncology.yaml \
    --corpus oncology \
    --fresh
```

## Creating Custom Corpora

Create a new YAML file in `corpus_specs/`:

```yaml
my_specialty:
  description: |
    Description of this corpus and its purpose.

  source: pubmed

  search_strategy:
    mesh_terms:
      - '"MeSH Term 1"[Mesh]'
      - '"MeSH Term 2"[Mesh]'
    filters:
      - '"open access"[filter]'
      - 'hasabstract'
    date_range: [2018, 2024]

  target_count: 300

  evaluator_model: gpt-5-mini
  confidence_threshold: 0.7

  validation_prompt: |
    You are evaluating whether a paper belongs in this corpus.

    INCLUDE:
    - Relevant paper types...

    EXCLUDE:
    - Irrelevant paper types...
```

Find MeSH terms at: https://www.ncbi.nlm.nih.gov/mesh/

## Troubleshooting

### "NCBI_EMAIL environment variable not set"

E-utilities requires an email for contact purposes. Set it in `.env`:

```
NCBI_EMAIL=your.email@example.com
```

### Rate limit errors

- Get an NCBI API key (free) for 10 req/sec
- Run during off-peak hours for OAI-PMH

### Empty XML responses

Some PMCIDs in search results may not have full-text available via OAI-PMH.
The builder skips these automatically.

### LLM evaluation failures

- Check `OPENAI_API_KEY` is valid
- Verify API quota/rate limits
- Builder retries 3 times with exponential backoff

## References

- [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
- [PMC OAI-PMH API](https://pmc.ncbi.nlm.nih.gov/tools/oai/)
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [MeSH Browser](https://www.ncbi.nlm.nih.gov/mesh/)
- [JATS Tag Suite](https://jats.nlm.nih.gov/)
