"""
evaluation/eval.py
==================
Évaluation complète du système RAG Tourisme Burkina Faso.

Métriques mesurées :
  - Précision Retrieval : % docs pertinents parmi les Top-K récupérés
  - Pertinence Réponse  : score /5 (via ROUGE + heuristique)
  - Temps de Réponse    : moyenne, p50, p95

Dataset : 20 questions avec réponses attendues (ground truth).

Usage :
    python evaluation/eval.py

Sortie :
    evaluation/results.json   — résultats détaillés
    evaluation/report.txt     — rapport lisible
"""

import sys
import json
import time
import statistics
from pathlib import Path

sys.path.insert(0, ".")
from src.rag_pipeline import TourismRAGPipeline
from rouge_score import rouge_scorer
from loguru import logger


# ══════════════════════════════════════════════════════════
# DATASET — 20 questions avec réponses attendues
# ══════════════════════════════════════════════════════════
# Format : question + réponse_attendue (mots-clés obligatoires)
# + docs_pertinents (URLs ou mots-clés qu'on attend dans les sources)
EVAL_DATASET = [
    {
        "id": "Q01",
        "question": "Où se trouvent les ruines de Loropéni et depuis quand sont-elles classées UNESCO ?",
        "expected_keywords": ["Loropéni", "UNESCO", "2009", "sud-ouest", "patrimoine"],
        "expected_source_keywords": ["loropeni", "unesco", "whc"],
        "min_similarity": 0.6,
    },
    {
        "id": "Q02",
        "question": "Quelles sont les cascades naturelles à visiter à Banfora ?",
        "expected_keywords": ["Banfora", "cascades", "Karfiguéla", "Sindou"],
        "expected_source_keywords": ["banfora", "cascade"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q03",
        "question": "Qu'est-ce que le FESPACO et où se déroule-t-il ?",
        "expected_keywords": ["FESPACO", "Ouagadougou", "cinéma", "africain", "festival"],
        "expected_source_keywords": ["fespaco", "ouagadougou"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q04",
        "question": "Que peut-on voir au site de sculptures de Laongo ?",
        "expected_keywords": ["Laongo", "granit", "sculptures", "artistes"],
        "expected_source_keywords": ["laongo"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q05",
        "question": "Quelles sont les traditions culturelles du peuple Mossi au Burkina Faso ?",
        "expected_keywords": ["Mossi", "culture", "tradition", "Moro-Naba", "chefferie"],
        "expected_source_keywords": ["mossi", "burkina"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q06",
        "question": "Comment se rendre aux Dômes de Fabédougou depuis Banfora ?",
        "expected_keywords": ["Fabédougou", "Banfora", "dômes", "grès"],
        "expected_source_keywords": ["fabedougou", "banfora"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q07",
        "question": "Quelle est la meilleure période de l'année pour visiter le Burkina Faso ?",
        "expected_keywords": ["saison", "novembre", "février", "harmattan", "pluies"],
        "expected_source_keywords": ["burkina", "tourisme", "climat"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q08",
        "question": "Qu'est-ce que le lac Tengrela et quels animaux y vit-on ?",
        "expected_keywords": ["Tengrela", "lac", "hippopotames", "Banfora"],
        "expected_source_keywords": ["tengrela"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q09",
        "question": "Quel artisanat traditionnel est typique du Burkina Faso ?",
        "expected_keywords": ["bogolan", "bronze", "poterie", "tissage", "artisanat"],
        "expected_source_keywords": ["artisanat", "burkina"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q10",
        "question": "Qu'est-ce que la cérémonie du Moro-Naba à Ouagadougou ?",
        "expected_keywords": ["Moro-Naba", "cérémonie", "vendredi", "Ouagadougou", "roi"],
        "expected_source_keywords": ["moro-naba", "ouagadougou"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q11",
        "question": "Quels sont les musées à visiter à Ouagadougou ?",
        "expected_keywords": ["musée", "Ouagadougou", "national", "culture"],
        "expected_source_keywords": ["musee", "ouagadougou"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q12",
        "question": "Quels plats locaux doit-on goûter au Burkina Faso ?",
        "expected_keywords": ["tô", "riz", "sauce", "dolo", "cuisine"],
        "expected_source_keywords": ["cuisine", "burkina", "gastronomie"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q13",
        "question": "Quels sont les marchés artisanaux de Ouagadougou ?",
        "expected_keywords": ["marché", "artisanat", "Ouagadougou", "SIAO"],
        "expected_source_keywords": ["marche", "ouagadougou", "SIAO"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q14",
        "question": "Quelles activités peut-on faire dans la réserve du W au Burkina Faso ?",
        "expected_keywords": ["Réserve W", "faune", "safari", "éléphants", "parc"],
        "expected_source_keywords": ["reserve", "parc", "W"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q15",
        "question": "Comment s'appelle la grande mosquée de Bobo-Dioulasso et quelle est son histoire ?",
        "expected_keywords": ["Bobo-Dioulasso", "mosquée", "Dioulassoba"],
        "expected_source_keywords": ["bobo", "mosquee"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q16",
        "question": "Quel est le rôle du SIAO dans l'artisanat burkinabè ?",
        "expected_keywords": ["SIAO", "artisanat", "biennal", "Ouagadougou", "Salon"],
        "expected_source_keywords": ["SIAO", "artisanat"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q17",
        "question": "Quels sont les peuples premiers du Burkina Faso et leurs traditions touristiques ?",
        "expected_keywords": ["Lobi", "Gourounsi", "Peuls", "traditions"],
        "expected_source_keywords": ["burkina", "peuple", "tradition"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q18",
        "question": "Comment le Burkina Faso développe-t-il son tourisme écologique ?",
        "expected_keywords": ["écotourisme", "développement", "durable", "communautaire"],
        "expected_source_keywords": ["tourisme", "burkina", "ecotourisme"],
        "min_similarity": 0.4,
    },
    {
        "id": "Q19",
        "question": "Qu'est-ce que le patrimoine culturel immatériel du Burkina Faso reconnu par l'UNESCO ?",
        "expected_keywords": ["UNESCO", "patrimoine", "immatériel", "Zaouli", "balafon"],
        "expected_source_keywords": ["unesco", "patrimoine", "burkina"],
        "min_similarity": 0.5,
    },
    {
        "id": "Q20",
        "question": "Quels sont les hébergements touristiques disponibles à Banfora ?",
        "expected_keywords": ["hôtel", "campement", "hébergement", "Banfora"],
        "expected_source_keywords": ["banfora", "hotel", "hebergement"],
        "min_similarity": 0.3,
    },
]


# ══════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════
def compute_retrieval_precision(
    retrieved_docs: list[dict],
    expected_source_keywords: list[str],
) -> float:
    """
    Précision du retrieval : % de docs récupérés qui contiennent
    au moins un mot-clé attendu dans leur URL ou titre.

    Formule : docs_pertinents / total_docs_récupérés
    """
    if not retrieved_docs:
        return 0.0

    relevant = 0
    for doc in retrieved_docs:
        doc_text = (doc["url"] + " " + doc["title"]).lower()
        # Un doc est pertinent s'il contient au moins un keyword attendu
        if any(kw.lower() in doc_text for kw in expected_source_keywords):
            relevant += 1

    return relevant / len(retrieved_docs)


def compute_response_relevance(
    response: str,
    expected_keywords: list[str],
) -> float:
    """
    Pertinence de la réponse : score /5.
    Basé sur la présence des mots-clés attendus dans la réponse.

    Score /5 :
      5 = tous les keywords présents
      4 = ≥80% keywords
      3 = ≥60% keywords
      2 = ≥40% keywords
      1 = ≥20% keywords
      0 = aucun keyword
    """
    if not response or not expected_keywords:
        return 0.0

    response_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    ratio = found / len(expected_keywords)

    if ratio >= 1.0:   return 5.0
    if ratio >= 0.8:   return 4.0
    if ratio >= 0.6:   return 3.0
    if ratio >= 0.4:   return 2.0
    if ratio >= 0.2:   return 1.0
    return 0.0


def compute_rouge_l(hypothesis: str, reference_keywords: list[str]) -> float:
    """
    ROUGE-L simplifié : mesure le chevauchement de séquences.
    Utilisé comme métrique additionnelle.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    reference = " ".join(reference_keywords)
    score = scorer.score(reference, hypothesis)
    return score["rougeL"].fmeasure


# ══════════════════════════════════════════════════════════
# RUNNER D'ÉVALUATION
# ══════════════════════════════════════════════════════════
def run_evaluation(pipeline: TourismRAGPipeline) -> dict:
    """
    Exécute l'évaluation complète sur les 20 questions.
    """
    results = []
    response_times = []
    precisions = []
    relevances = []

    logger.info(f"=== ÉVALUATION — {len(EVAL_DATASET)} questions ===")

    for i, item in enumerate(EVAL_DATASET, 1):
        logger.info(f"[{i:02d}/{len(EVAL_DATASET)}] {item['id']}: {item['question'][:60]}...")

        try:
            # Appel pipeline
            start = time.time()
            result = pipeline.query(item["question"])
            elapsed = time.time() - start

            # Métriques
            precision = compute_retrieval_precision(
                result["retrieved_docs"],
                item["expected_source_keywords"],
            )
            relevance = compute_response_relevance(
                result["response"],
                item["expected_keywords"],
            )
            rouge_l = compute_rouge_l(
                result["response"],
                item["expected_keywords"],
            )

            # Stocke
            response_times.append(elapsed)
            precisions.append(precision)
            relevances.append(relevance)

            item_result = {
                "id": item["id"],
                "question": item["question"],
                "response": result["response"][:300] + "...",
                "response_time": round(elapsed, 2),
                "retrieval_precision": round(precision, 4),
                "response_relevance": round(relevance, 2),
                "rouge_l": round(rouge_l, 4),
                "nb_docs_retrieved": len(result["retrieved_docs"]),
                "safe": result["safe"],
            }
            results.append(item_result)

            logger.info(
                f"  ✓ Précision={precision:.0%} | Pertinence={relevance}/5 | "
                f"Temps={elapsed:.2f}s"
            )

        except Exception as e:
            logger.error(f"  ✗ Erreur : {e}")
            results.append({
                "id": item["id"],
                "question": item["question"],
                "error": str(e),
                "response_time": 0,
                "retrieval_precision": 0,
                "response_relevance": 0,
                "rouge_l": 0,
                "nb_docs_retrieved": 0,
                "safe": False,
            })

    # ── Calcul métriques globales ──
    avg_precision = statistics.mean(precisions) if precisions else 0
    avg_relevance = statistics.mean(relevances) if relevances else 0
    avg_time = statistics.mean(response_times) if response_times else 0
    p95_time = sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0

    # Vérification des objectifs
    target_precision = 0.85  # Objectif : >85%
    target_relevance = 3.5   # Objectif : >3.5/5
    target_time = 3.0        # Objectif : <3s

    summary = {
        "total_questions": len(EVAL_DATASET),
        "avg_retrieval_precision": round(avg_precision, 4),
        "avg_response_relevance": round(avg_relevance, 2),
        "avg_response_time_s": round(avg_time, 2),
        "p95_response_time_s": round(p95_time, 2),
        "targets": {
            "precision_ok": avg_precision >= target_precision,
            "relevance_ok": avg_relevance >= target_relevance,
            "time_ok": avg_time <= target_time,
        },
        "questions": results,
    }

    return summary


# ══════════════════════════════════════════════════════════
# GÉNÉRATION DU RAPPORT
# ══════════════════════════════════════════════════════════
def generate_report(summary: dict) -> str:
    """Génère un rapport lisible en texte."""
    t = summary["targets"]
    prec_ok = "✓" if t["precision_ok"] else "✗"
    rel_ok  = "✓" if t["relevance_ok"]  else "✗"
    time_ok = "✓" if t["time_ok"]       else "✗"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         RAPPORT ÉVALUATION — RAG TOURISME BURKINA FASO       ║
╚══════════════════════════════════════════════════════════════╝

MÉTRIQUES GLOBALES
──────────────────
  Précision Retrieval (moy) : {summary['avg_retrieval_precision']:.1%}  {prec_ok} (objectif : >85%)
  Pertinence Réponse (moy)  : {summary['avg_response_relevance']:.1f}/5  {rel_ok} (objectif : >3.5/5)
  Temps Réponse (moy)       : {summary['avg_response_time_s']:.2f}s     {time_ok} (objectif : <3s)
  Temps Réponse (p95)       : {summary['p95_response_time_s']:.2f}s

RÉSULTATS PAR QUESTION
──────────────────────
"""
    for q in summary["questions"]:
        if "error" in q:
            report += f"  [{q['id']}] ERREUR : {q['error']}\n"
        else:
            report += (
                f"  [{q['id']}] Précision={q['retrieval_precision']:.0%} | "
                f"Pertinence={q['response_relevance']}/5 | "
                f"Temps={q['response_time']}s\n"
                f"         Q: {q['question'][:70]}...\n"
            )

    targets_ok = sum(t.values())
    report += f"""
SYNTHÈSE
────────
  Objectifs atteints : {targets_ok}/3
  Questions évaluées : {summary['total_questions']}

  Stack évalué :
    Embeddings : BAAI/bge-m3 (Apache 2.0)
    Vector DB  : ChromaDB (Apache 2.0)
    LLM        : Qwen2.5-7B via Ollama (Apache 2.0)
    Backend    : FastAPI (MIT)
"""
    return report


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    output_dir = Path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialise le pipeline
    logger.info("Initialisation pipeline pour évaluation...")
    pipeline = TourismRAGPipeline()
    pipeline.index()  # S'assure que le corpus est indexé

    # Lance l'évaluation
    summary = run_evaluation(pipeline)

    # Sauvegarde résultats JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.success(f"Résultats sauvegardés : {results_path}")

    # Génère le rapport texte
    report = generate_report(summary)
    report_path = output_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    logger.success(f"Rapport sauvegardé : {report_path}")
