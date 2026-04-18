# Dataset Plan — Legal Acquisition for Red Team Kitchen

## The hard constraint

Red Team Kitchen is a **for-profit LLC**. That changes everything about what datasets you can use.

**Licensed academic databases (Purdue alumni, ISU sponsored, CPL) are off-limits for the commercial product.** They are licensed under "personal, non-commercial use only" terms with explicit prohibitions on:

- Systematic downloading (automated tools, bulk ingestion)
- Text/data mining without a separate TDM agreement
- Using content to train or fine-tune models for commercial products
- Building a searchable knowledge base for a for-profit service
- Sharing login credentials
- Redistributing content

Violating these terms puts at risk: your alumni access, your ISU sponsorship, potential vendor lawsuits (ProQuest, Sage, Elsevier have sued over TDM violations), and the LLC's standing.

**Rule of thumb:** if the content sits behind your alumni login, it's for reading, not ingesting.

## What you CAN use (confirmed commercial-use permissive)

### fMRI / neuroscience datasets

| Dataset | License | Scope | Where |
|---------|---------|-------|-------|
| OpenNeuro (most) | CC0 / CC-BY | Thousands of BIDS-format fMRI datasets, openly licensed | openneuro.org |
| Human Connectome Project (HCP) Young Adult | WU-Minn HCP Open Access Data Use Terms — permissive, attribution required | 1,200 subjects, high-quality fMRI | db.humanconnectome.org |
| HCP Aging / HCP Development | Open Access | Lifespan fMRI | db.humanconnectome.org |
| Natural Scenes Dataset (NSD) | CC-BY 4.0 | 73k scene responses, 8 subjects | naturalscenesdataset.org |
| Algonauts Project 2023 | CC-BY 4.0 | Movie-watching fMRI, directly relevant to TRIBE v2 | algonauts.csail.mit.edu |
| StudyForrest | CC0 / ODC-ODbL | Long movie-watching fMRI, multiple sessions | studyforrest.org |
| NeuroVault | CC0 | Unthresholded statistical maps | neurovault.org |

**These are the only fMRI-type datasets you should touch for training / fine-tuning a commercial model.** All have attribution obligations you must honor in your model card + website.

### Text corpora (for Gemma fine-tuning / RAG)

| Source | License | Scope |
|--------|---------|-------|
| PubMed Central Open Access subset | Various (check per-article) | ~7M biomedical articles, mostly CC-BY or similar |
| arXiv bulk access | Per-article license (many CC-BY; check metadata) | ~2M preprints |
| Wikipedia (all dumps) | CC-BY-SA 4.0 | Full text, requires attribution + share-alike |
| Wikidata | CC0 | Structured facts |
| Common Crawl | Terms of use — allowed for research/commercial with attribution | Web scrape |
| S2ORC (Allen AI) | Non-commercial for full text; metadata OK commercially | Research paper corpus |
| OpenWebText | Research use, check specifics | GPT-2 training-data recreation |
| Project Gutenberg | Public domain in US | Books pre-1929 |
| Government data (census.gov, nih.gov, nsf.gov) | Public domain (US federal works) | Reports, statistics |

**Flag:** S2ORC full-text is research-only. Use for your own research, NOT to build Red Team Kitchen's product.

### Pre-licensed training datasets (HuggingFace)

Search HuggingFace datasets with `license:cc-by-4.0`, `license:cc0-1.0`, `license:apache-2.0`, `license:mit`. Specifically vetted:

- `allenai/dolma` — open pretraining corpus, permissive
- `HuggingFaceFW/fineweb` — Apache 2.0
- `open-phi/textbooks` — MIT
- `teknium/OpenHermes-2.5` — instruction tuning, check license

Always verify license on the dataset card before download.

## What you CANNOT use

- **Any content behind Purdue alumni login** (ProQuest, Project Muse, Sage Journals, Adam Matthew, CQ Press, Weiss) — reading yes, ingesting no
- **Any content behind CPL database access** (JSTOR via library, ProQuest via library, LinkedIn Learning video/transcripts) — consumption yes, systematic download no
- **Any content behind ISU sponsored-account access** (Milner Library subscriptions) — same rule
- **Scraped copyrighted content** — news articles, paywalled magazines, etc., even if publicly reachable
- **YouTube bulk download of copyrighted videos** — terms violation and copyright issue
- **Social media data mass-scraped** (Twitter/X, Instagram, TikTok) — terms + copyright
- **Any dataset whose license you cannot confirm** — default to NO

## Download plan (legally clean)

### Phase 1 — fMRI datasets for TRIBE v2 validation

```bash
# Target bucket
BUCKET="gs://rtk-datasets"

# OpenNeuro — use datalad (BIDS standard)
pip install datalad datalad-osf
datalad install https://github.com/OpenNeuroDatasets/ds003020.git  # example
datalad get -d ds003020 .

# HCP — requires registration at db.humanconnectome.org, accept DUA,
# then AWS-hosted via awscli with HCP credentials
aws s3 sync s3://hcp-openaccess/HCP_1200/ D:/datasets/HCP_1200/ \
    --exclude "*" --include "*/MNINonLinear/Results/*/*fMRI*"

# NSD
wget -r https://naturalscenesdataset.org/download/

# Push to GCS
gcloud storage cp -r D:/datasets/* $BUCKET/
```

### Phase 2 — text corpora for reading + RAG (your own internal use)

```bash
# PubMed Central OA subset (commercial-allowed per-article; filter by license)
wget ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/  # commercial-use OK subset

# arXiv metadata dump (Kaggle has clean version)
kaggle datasets download -d Cornell-University/arxiv

# Wikipedia dump
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

### Phase 3 — PDF-to-markdown conversion of the legal corpora

Use **marker** (MIT license, fast, high-quality) for anything where OCR matters.

```bash
pip install marker-pdf
marker /path/to/pdfs/ /path/to/markdown_out/ --workers 4 --torch-device cuda
```

For papers specifically (where math rendering matters), use **docling** as fallback:

```bash
pip install docling
docling /path/to/hard_paper.pdf --to md
```

Alternatives by use case:
- **MinerU** — for any Chinese/Japanese/Korean papers
- **olmOCR** — for linearization when building training data
- **PyMuPDF4LLM** — for born-digital PDFs where speed > accuracy

## Attribution discipline

Every model card, every website page that mentions datasets used, must include:

- Dataset name
- License
- Original paper citation (BibTeX)
- Link to source

This is both a legal requirement for CC-BY datasets AND a credibility signal for grant reviewers.

Template for `docs/ATTRIBUTIONS.md` (create when first dataset ingested):

```markdown
## Datasets used in training / fine-tuning

### [Dataset name]
- Source: [URL]
- License: [CC-BY-4.0 / CC0 / etc.]
- Citation: [full academic citation]
- Used for: [training / validation / RAG / etc.]
- Version / date accessed: [...]
```

## TL;DR

- Purdue + ISU + CPL access: **read-only for personal research**. Do not download, do not ingest, do not fine-tune on this.
- For Red Team Kitchen's actual datasets: OpenNeuro, HCP, NSD, arXiv, PubMed Central OA, Wikipedia, government data. All confirmed commercial-use permissive.
- Marker for PDF→MD, Docling for hard cases, attribute everything.

---

*Last updated: April 2026*
