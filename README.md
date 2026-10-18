# 🇧🇫 Assistant IA Tourisme Burkina Faso — Système RAG open source

> Questions-réponses en français sur le tourisme burkinabè, fondées sur un
> corpus cité et vérifiable.
> Embeddings MiniLM locaux + ChromaDB + génération via l'API Groq.
> Le chemin historique (Ollama local) reste disponible en transition.

**État du projet :** pipeline v2 complet (retrieval manifesté, génération
citéée, abstention sans preuve) testé hors-ligne — **229 tests verts,
2 sautés** — et intégré en CI sur chaque PR. Le parcours **live** Groq
n'a pas encore été vérifié en conditions réelles : voir
`docs/phase4-groq-smoke-test.md` (procédure manuelle).

---

## Architecture réelle (v2)

```
Question (API FastAPI ou interface Gradio)
   │
   ▼
Sanitization (InputSanitizer, longueur max, détection d'injection)
   │
   ▼
Retriever v2 — embeddings paraphrase-multilingual-MiniLM-L12-v2 (locaux,
normalisés) → ChromaDB data/chroma_db_v2 (cosine, similarity = 1 − distance)
   │    ▲ validé au démarrage contre data/index_manifest.json (modèle,
   │      dimension, comptes, sha256 du corpus — tout écart = erreur claire)
   ▼
Générateur — prompt français anti-hallucination, passages délimités comme
données non fiables → LLMService (GroqService : API Groq, retries bornés,
Retry-After plafonné, erreurs typées)
   │
   ▼
Validation des citations ([Source N] doit mapper vers une preuve récupérée)
   │
   ▼
Réponse en français, sources citées — ou ABSTENTION explicite si preuve
insuffisante (aucun appel LLM sans preuve ; réponse sans citation valide
= abstention)
```

**Limite assumée :** la validation des citations prouve que chaque `[Source N]`
correspond à un document réellement récupéré ; elle ne prouve pas que le
passage soutient sémantiquement l'affirmation (évaluation à venir).

---

## Composants

| Composant | Technologie | Rôle |
|---|---|---|
| Embeddings | `sentence-transformers` + `paraphrase-multilingual-MiniLM-L12-v2` | requête et corpus (identiques, normalisés) |
| Base vectorielle | ChromaDB (`tourisme_burkina_v2`, cosine) | index persistant reconstruit hors-ligne |
| Génération (v2) | `groq` (API Groq, défaut `openai/gpt-oss-20b`) | réponses françaises citées |
| Génération (legacy) | `ollama` + `langgraph`/`langchain-core` | chemin historique, sélectionnable |
| Backend | FastAPI + slowapi | API + rate limiting |
| UI | Gradio | interface web locale |
| Ingestion | requests, trafilatura, BeautifulSoup, pdfplumber | collecte respectueuse (robots.txt, TLS vérifié, limites de taille) |
| Tests | pytest + reportlab + PyYAML | 229 tests hors-ligne en CI |

---

## Installation

```bash
git clone https://github.com/TaniaW777/TOURISME-BURKINA-RAG.git
cd TOURISME-BURKINA-RAG
python -m venv venv && venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Clé Groq (génération v2) : créez une clé sur https://console.groq.com/keys
puis exportez-la — **jamais** dans un fichier committé :

```bash
export GROQ_API_KEY="votre-clé"        # Windows PowerShell : $env:GROQ_API_KEY="..."
```

Sans clé, l'application démarre en **mode dégradé** (API et interface se
lancent, `/health` explique, `/query` répond 503) — il n'y a **jamais** de
repli silencieux vers Ollama.

### Variables d'environnement (voir `.env.example`)

| Variable | Défaut | Rôle |
|---|---|---|
| `GROQ_API_KEY` | — | clé API Groq (secrète, côté serveur uniquement) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | modèle de génération |
| `GROQ_TIMEOUT_S` / `GROQ_MAX_RETRIES` | 20 / 2 | bornes d'appel |
| `LLM_PROVIDER` | `groq` | `groq` (v2) ou `legacy` (Ollama — **API uniquement**, l'interface Gradio utilise toujours v2) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | doit correspondre au manifeste de l'index |
| `TOP_K` / `MIN_SIMILARITY` | 5 / 0.30 | paramètres de retrieval |
| `RATE_LIMIT_PER_MINUTE` | 10 | limite slowapi |

---

## Corpus et index

Corpus committé et reproductible : `data/chunks.jsonl` — **1 383 chunks
validés au tokenizer** (≤ 128 tokens), issus de **37 documents** (32 fr,
4 en, 1 non déterminé) couvrant **41 sources curées** (`data/sources.yaml` :
Wikipédia/Wikivoyage FR+EN, institutionnels `.bf`, UNESCO, presse, blogs)
et 2 PDFs locaux. Chaque fragment abandonné est audité dans
`data/ingest_report.json`.

```bash
# (Re)collecter les sources — respecte robots.txt, TLS vérifié, limites de taille
python scripts/ingest.py
# Rechunker sans réseau depuis data/documents.jsonl
python scripts/ingest.py --from-documents data/documents.jsonl

# Construire l'index v2 (hors-ligne, déterministe, manifeste atomique)
python scripts/build_index.py
# → data/chroma_db_v2/ (gitignoré) + data/index_manifest.json
```

L'index n'est **jamais** construit au démarrage de l'application : le
retriever charge et vérifie l'index existant via son manifeste.

## Lancer

```bash
# API — http://localhost:8000/docs (Swagger)
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Interface Gradio — http://localhost:7860
python frontend/app.py
```

### Endpoints API

| Route | Comportement |
|---|---|
| `POST /query` | `{question, top_k?}` → réponse française, `sources` (preuves citées), `documents` (preuves récupérées), `citations`, `abstained`, `response_time`, `safe` ; 503 en mode dégradé |
| `GET /health` | état de configuration **sans réseau** : provider, clé, manifeste, diagnostics |
| `GET /health/provider` | sonde Groq **réelle** (consomme du quota — à appeler à la demande) |
| `GET /stats` | comptes de l'index, modèles configurés |
| `POST /index` | chemin legacy uniquement ; 501 sur v2 (l'indexation est hors-ligne) |

---

## Tests et CI

```bash
pytest tests/ -q        # 229 passed, 2 skipped — hors-ligne, sans clé, sans modèle
```

GitHub Actions exécute la suite sur chaque PR et push vers `main`
(`.github/workflows/tests.yml`). Les tests d'acceptance bout-en-bout
traversent la pile v2 complète sur index factice déterministe
(`tests/test_acceptance_v2.py`).

## Sécurité

- Clé Groq lue uniquement de l'environnement ; jamais loguée, jamais
  renvoyée au client (testé).
- Entrées utilisateur assainies (longueur, HTML, heuristiques d'injection).
- Passages récupérés traités comme **données non fiables** (délimiteurs +
  consigne d'ignorer leurs instructions éventuelles).
- Scraper : robots.txt strict, TLS vérifié, limites de taille, User-Agent
  honnête, registre de sources auditable.

## Héritage et transition

- Le pipeline Ollama historique (`src/rag_pipeline.py`) reste utilisable via
  `LLM_PROVIDER=legacy` (API) et est archivé dans la branche
  `archive/legacy-ollama-pre-groq`. Il sera retiré après validation du
  parcours Groq en conditions réelles.
- Le test manuel live Groq : `docs/phase4-groq-smoke-test.md` (jamais en CI).

---

*Licence MIT — Contribution bienvenue — 🇧🇫*
