# 🇧🇫 Assistant IA Tourisme Burkina Faso — Système RAG Open Source

> Système de questions-réponses 100% open source sur le tourisme burkinabè.
> BGE-M3 + ChromaDB + Qwen3 via Ollama. Zéro coût cloud. Zéro API propriétaire.

## Sujet choisi : Tourisme Burkina Faso

**Justification :** Le tourisme burkinabè (Loropéni UNESCO, Cascades de Banfora,
Laongo, FESPACO, artisanat Mossi) est documentable, sous-représenté en IA francophone,
et pertinent pour valoriser le patrimoine local. Critères remplis : 500+ sources
accessibles, forte pertinence locale, données en français.

---

## Architecture technique

```
Question
   │
   ▼
[Sanitization] ← Prompt injection detection, bleach, max length
   │
   ▼
[Embeddings] ← BAAI/bge-m3 (multilingual, top MTEB, local)
   │
   ▼
[ChromaDB] ← Recherche vectorielle (cosine similarity, top-5)
   │
   ▼
[Agent LangGraph] ← Vérificateur → Synthétiseur
   │
   ▼
[Ollama + Qwen3] ← LLM local, réponse + sources citées
   │
   ▼
Réponse
```

**Couche agentique légère (LangGraph) :**
- Nœud 1 — Vérificateur : évalue la pertinence des documents récupérés
- Nœud 2 — Synthétiseur : génère la réponse avec sources obligatoires

---

## Technologies open source utilisées

| Composant | Technologie | Licence | Lien |
|-----------|------------|---------|------|
| Embeddings | sentence-transformers + BAAI/bge-m3 | Apache 2.0 / MIT | [github](https://github.com/UKPLab/sentence-transformers) |
| Base vectorielle | ChromaDB | Apache 2.0 | [github](https://github.com/chroma-core/chroma) |
| LLM | Ollama + Qwen2.5-7B | MIT / Apache 2.0 | [github](https://github.com/ollama/ollama) |
| Agent | LangGraph | MIT | [github](https://github.com/langchain-ai/langgraph) |
| Backend | FastAPI | MIT | [github](https://github.com/tiangolo/fastapi) |
| Frontend | Gradio | Apache 2.0 | [github](https://github.com/gradio-app/gradio) |
| Scraping | BeautifulSoup4 + trafilatura | MIT / Apache 2.0 | [github](https://github.com/adbar/trafilatura) |
| Sécurité inputs | bleach | Apache 2.0 | [github](https://github.com/mozilla/bleach) |
| Logs | loguru | MIT | [github](https://github.com/Delgan/loguru) |
| Rate limiting | slowapi | MIT | [github](https://github.com/laurentS/slowapi) |

---

## Installation pas-à-pas

### Prérequis
- Python 3.10+
- [Ollama](https://ollama.com) installé localement
- Git

### 1. Clone le repo
```bash
git clone https://github.com/TaniaW777/TOURISME-BURKINA-RAG.git
cd tourisme-burkina-rag
```

### 2. Installe les dépendances Python
```bash
pip install -r requirements.txt
```

### 3. Installe le modèle LLM via Ollama
```bash
# Démarre Ollama en arrière-plan
ollama serve &

# Télécharge le modèle (~4GB)
ollama pull qwen2.5:7b-instruct

# Alternative si RAM limitée (<8GB) :
ollama pull mistral:7b-instruct
```

### 4. Collecte les données (scraping)
```bash
# Lance le scraper (respecte robots.txt automatiquement)
python data/scraping_script.py

# Vérifie le corpus généré
wc -l data/corpus.json   # Doit afficher 500+ entrées
```

### 5. Lance l'interface
```bash
# Option A : Interface Gradio (recommandée)
python frontend/app.py
# → Ouvre http://localhost:7860

# Option B : API FastAPI
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# → Docs Swagger : http://localhost:8000/docs
```

### 6. Évalue le système
```bash
python evaluation/eval.py
# Résultats dans evaluation/results.json et evaluation/report.txt
```

---

## Structure du repo

```
tourisme-burkina-rag/
├── data/
│   ├── scraping_script.py   # Scraper web (BeautifulSoup + trafilatura)
│   ├── corpus.json          # Corpus généré (500+ chunks)
│   └── sources.txt          # URLs sources scrapées
├── src/
│   ├── rag_pipeline.py      # Pipeline RAG complet (commenté ligne/ligne)
│   └── api.py               # API FastAPI
├── frontend/
│   └── app.py               # Interface Gradio
├── evaluation/
│   ├── eval.py              # 20 questions + métriques
│   ├── results.json         # Résultats JSON
│   └── report.txt           # Rapport lisible
├── requirements.txt         # Dépendances + licences
├── LICENSE                  # MIT
└── README.md
```

---

## Résultats d'évaluation (à remplir après run)

| Métrique | Résultat | Objectif |
|----------|---------|---------|
| Précision Retrieval | _à compléter_ | >85% |
| Pertinence Réponse | _à compléter_ | >3.5/5 |
| Temps Réponse (moy) | _à compléter_ | <3s |

---

## Commits Git recommandés

```bash
git init
git add .
git commit -m "feat: initial project structure"

# Après scraping :
git add data/corpus.json data/sources.txt
git commit -m "data: add tourism corpus (500+ chunks)"

# Après chaque fichier code :
git commit -m "feat: add RAG pipeline with BGE-M3 + ChromaDB"
git commit -m "feat: add FastAPI backend with security"
git commit -m "feat: add Gradio frontend"
git commit -m "eval: add 20-question evaluation dataset"
```

---

*Licence MIT — Contribution bienvenue — 🇧🇫 Fait pour valoriser le tourisme burkinabè*
