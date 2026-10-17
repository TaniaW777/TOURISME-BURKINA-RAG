# Test manuel de fumée — Groq / pipeline v2 (Phase 4)

> **Procédure strictement MANUELLE.** Elle ne fait partie d'aucun test
> automatisé et ne s'exécute jamais en CI : elle consomme du quota Groq
> réel et nécessite une clé API. Réservée à un développeur autorisé,
> typiquement avant une acceptance ou un déploiement.

## 1. Prérequis

- Dépôt cloné, dépendances installées (`pip install -r requirements.txt`).
- Index v2 construit et vérifié :

  ```bash
  python scripts/build_index.py
  # attendu : « Index v2 prêt : N chunks, dim=384 » + « Vérification manifest OK »
  ```

- Une clé API Groq **valide** (https://console.groq.com/keys).
- Ollama n'est **pas** requis (le chemin v2 ne l'utilise jamais).

## 2. Configuration de l'environnement

```bash
# Linux/macOS
export GROQ_API_KEY="votre-clé"        # ne jamais committer ce fichier
export GROQ_MODEL="openai/gpt-oss-20b" # optionnel — valeur par défaut
export LLM_PROVIDER="groq"             # optionnel — valeur par défaut

# Windows (PowerShell)
$env:GROQ_API_KEY="votre-clé"
```

**Règles de sécurité :**
- Ne jamais écrire la clé dans le dépôt, un commit, un log ou une issue.
- Si vous utilisez un fichier `.env` local, il est déjà gitignoré —
  ne le partagez jamais ; `.env.example` (sans secret) sert de modèle.
- La clé n'apparaît ni dans `/health`, ni dans les réponses, ni dans les
  exceptions (garanti par les tests `test_groq_service.py`).

## 3. Lancer et interroger l'API

```bash
uvicorn src.api:app --port 8000
```

- Santé de base (AUCUN appel réseau) : `curl http://localhost:8000/health`
  — attendu `pipeline_ready: true`, `provider: "groq"`,
  `groq_configured: true`, `index_manifest_ok: true`, `problems: []`.
- Question réelle via le pipeline v2 complet :

  ```bash
  curl -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -d '{"question": "Que peut-on voir aux cascades de Banfora ?", "top_k": 5}'
  ```

Variante sans serveur (pipeline direct) :

```bash
python -c "
from src.pipeline import TourismRAGPipelineV2
p = TourismRAGPipelineV2(); p.initialize()
r = p.query('Que peut-on voir aux cascades de Banfora ?')
print(r['response']); print('sources:', r['sources']); print('citations:', r['citations'])
"
```

## 4. Ce qu'il faut vérifier

| Point | Attendu |
|---|---|
| Réponse | En français, 2 à 5 puces, chaque fait préfixé par `[Source N]` |
| `sources` | Des URLs réellement présentes dans le corpus (families Wikipedia/burkinatourism/…) |
| Citations | `citations` ⊆ {1..top_k} — chaque numéro correspond à un document de `retrieved_docs` |
| Preuves | `retrieved_docs` contient passages + métadonnées (url, titre, langue, similarité) |
| Abstention | Une question hors corpus (ex. « Quel est le prix de l'or aujourd'hui ? ») → message d'abstention, `abstained: true` |
| Modèle | `model` mentionne le modèle Groq configuré |

**Limite assumée :** la validité d'une citation prouve qu'elle mappe vers
un document récupéré — pas que le passage soutient sémantiquement le fait.
Une lecture manuelle des passages cités reste de mise.

## 5. Échecs courants et symptômes

| Symptôme | Cause probable |
|---|---|
| `/health` → `problems: ["manifeste index absent…"]` | Index non construit (`scripts/build_index.py`) |
| `/health` → `modèle d'embeddings incohérent` | `EMBEDDING_MODEL` ≠ modèle du build — reconstruire l'index |
| Démarrage dégradé, `GROQ_API_KEY absente` | Clé non exportée dans CETTE session shell |
| 401/`clé Groq rejetée` | Clé invalide ou révoquée |
| 429 avec `retry_after_s` | Quota/rythme dépassé — le pipeline attend et réessaie de façon bornée, puis remonte l'erreur |
| Timeout / `Groq injoignable` | Réseau, proxy, ou panne provider |

## 6. Sonde provider (en connaissance de cause et avec parcimonie)

`GET /health/provider` effectue **une vraie requête Groq** (`models.list`) :
elle consomme du quota et peut elle-même être limitée en débit. À appeler
à la demande, jamais en boucle, et à mettre en cache si utilisée par une
supervision.

## 7. Après le test

- Révoquer la clé si elle a été partagée par erreur.
- Ne pas committer de fichier `.env`, de log contenant des entrées de
  requêtes, ou de dump d'environnement.
- Consigner les résultats (répondus/abstentions, latence ressentie) dans
  l'issue ou le ticket d'acceptance concerné.
