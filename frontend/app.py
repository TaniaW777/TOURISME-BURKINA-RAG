"""
frontend/app.py
===============
Interface utilisateur Gradio pour le système RAG Tourisme Burkina.
Gradio : Apache 2.0 — https://github.com/gradio-app/gradio

Usage local :
    python frontend/app.py

Déploiement HuggingFace Spaces :
    - Copie ce fichier à la racine comme app.py
    - Ajoute requirements.txt
    - Push sur ton Space
"""

import sys
import time
sys.path.insert(0, ".")

import gradio as gr
from loguru import logger

# Pipeline v2 (Groq) — chargé PARESSEUSEMENT à la première question :
# aucun modèle/index n'est construit à l'import, et l'index n'est JAMAIS
# reconstruit ici (lecture du v2 manifesté). Pas de repli Ollama silencieux.
from src.pipeline import PipelineError, TourismRAGPipelineV2

pipeline = None
_pipeline_error: str | None = None


def _get_pipeline():
    """Initialisation explicite à la demande ; None + message en mode dégradé."""
    global pipeline, _pipeline_error
    if pipeline is None and _pipeline_error is None:
        try:
            p = TourismRAGPipelineV2()
            p.initialize()
            pipeline = p
            logger.success("Pipeline v2 prêt.")
        except Exception as e:
            _pipeline_error = f"{e.__class__.__name__}: {e}"
            logger.error(f"Pipeline v2 non initialisable : {_pipeline_error}")
    return pipeline

# ──────────────────────────────────────────────
# EXEMPLES DE QUESTIONS
# ──────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Où se trouvent les ruines de Loropéni et pourquoi sont-elles classées UNESCO ?",
    "Quels sont les sites naturels à visiter à Banfora ?",
    "Comment se déroule le festival FESPACO à Ouagadougou ?",
    "Quelles sont les spécialités culinaires du Burkina Faso ?",
    "Comment visiter les sculptures de Laongo ?",
    "Quels artisanats traditionnels peut-on acheter au Burkina ?",
    "Quelles sont les meilleures périodes pour visiter le Burkina Faso ?",
    "Qu'est-ce que le Moro-Naba et quelles cérémonies peut-on observer ?",
]


# ──────────────────────────────────────────────
# FONCTION PRINCIPALE
# ──────────────────────────────────────────────
def answer_question(question: str, top_k: int = 5):
    """
    Traite une question et retourne la réponse RAG.
    Appelée par Gradio à chaque soumission (4 sorties câblées : réponse,
    sources, documents, métriques — toutes les branches retournent 4 valeurs).
    """
    if not question.strip():
        return "⚠️ Veuillez poser une question.", "", "", ""

    # Pipeline obtenu via l'initialisation paresseuse (jamais le global nu :
    # à la première question il vaut None → AttributeError sinon).
    active = _get_pipeline()
    if active is None:
        return (
            f"⚙️ Assistant non configuré : {_pipeline_error}",
            "", "", "",
        )

    try:
        result = active.query(question, top_k=top_k)

        # Réponse principale
        response = result["response"]

        # Alerte si query non safe
        if not result["safe"]:
            return (
                "🚫 Question non autorisée. Posez une question sur le tourisme burkinabè.",
                "",
                "",
                ""
            )

        # Formatage des sources
        if result["sources"]:
            sources_text = "\n".join(
                f"• {url}" for url in result["sources"]
            )
        else:
            sources_text = "Aucune source spécifique trouvée."

        # Formatage des documents récupérés
        docs_text = ""
        for i, doc in enumerate(result["retrieved_docs"], 1):
            docs_text += (
                f"**[{i}] {doc['title']}** (similarité: {doc['similarity']:.2%})\n"
                f"{doc['text'][:300]}...\n\n"
            )

        # Métriques
        metrics = (
            f"⏱ Temps de réponse : **{result['response_time']}s** | "
            f"📄 Documents utilisés : **{len(result['retrieved_docs'])}**"
        )

        return response, sources_text, docs_text, metrics

    except Exception as e:
        logger.error(f"Erreur interface : {e}")
        return f"❌ Erreur : {str(e)}", "", "", ""


# ──────────────────────────────────────────────
# INTERFACE GRADIO
# ──────────────────────────────────────────────
with gr.Blocks(
    title="🇧🇫 Assistant Tourisme Burkina Faso",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="green",
    ),
    css="""
    .header { text-align: center; margin-bottom: 20px; }
    .metric-box { background: #f0f4f0; padding: 10px; border-radius: 8px; }
    """,
) as demo:

    # ── En-tête ──
    gr.Markdown(
        """
        # 🇧🇫 Assistant IA — Tourisme Burkina Faso
        **Système RAG 100% open source** | BGE-M3 + ChromaDB + Qwen3 via Ollama

        Posez vos questions sur les sites touristiques, la culture, l'artisanat
        et les expériences uniques du Burkina Faso.
        """,
        elem_classes=["header"]
    )

    # ── Interface principale ──
    with gr.Row():
        with gr.Column(scale=2):
            # Zone de question
            question_input = gr.Textbox(
                label="Votre question sur le tourisme burkinabè",
                placeholder="Ex: Quels sont les sites classés UNESCO au Burkina Faso ?",
                lines=3,
                max_lines=5,
            )

            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Nombre de sources à consulter",
                )
                submit_btn = gr.Button(
                    "🔍 Rechercher",
                    variant="primary",
                    size="lg",
                )

            # Exemples cliquables
            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_input,
                label="Questions exemples (cliquez pour essayer)",
            )

        with gr.Column(scale=3):
            # Réponse principale
            response_output = gr.Markdown(
                label="Réponse de l'assistant",
                value="*La réponse apparaîtra ici...*",
            )

            # Métriques
            metrics_output = gr.Markdown(label="Métriques")

    # ── Onglets détails ──
    with gr.Tabs():
        with gr.TabItem("📚 Sources utilisées"):
            sources_output = gr.Markdown(
                label="URLs des sources",
                value="*Les sources apparaîtront ici après votre question.*",
            )

        with gr.TabItem("🔍 Documents récupérés"):
            gr.Markdown(
                "Extraits des documents les plus similaires à votre question :"
            )
            docs_output = gr.Markdown(
                value="*Les documents apparaîtront ici après votre question.*",
            )

        with gr.TabItem("ℹ️ À propos"):
            gr.Markdown(
                """
                ## Architecture technique
                - **Embeddings** : BAAI/bge-m3 (MIT, HuggingFace)
                - **Base vectorielle** : ChromaDB (Apache 2.0)
                - **LLM** : Qwen2.5-7B via Ollama (Apache 2.0)
                - **Backend** : FastAPI (MIT)
                - **Frontend** : Gradio (Apache 2.0)

                ## Corpus
                500+ documents sur le tourisme burkinabè :
                ruines de Loropéni (UNESCO), cascades de Banfora,
                sculptures de Laongo, FESPACO, artisanat Mossi, etc.

                ## Licence
                MIT — Code source disponible sur GitHub

                ## Respect de la vie privée
                Aucune donnée personnelle stockée.
                Les questions sont hashées dans les logs (non réversible).
                """
            )

    # ── Connexion événements ──
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, top_k_slider],
        outputs=[response_output, sources_output, docs_output, metrics_output],
        show_progress="full",
    )

    # Soumission avec Enter
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, top_k_slider],
        outputs=[response_output, sources_output, docs_output, metrics_output],
    )


# ──────────────────────────────────────────────
# LANCEMENT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # Accessible depuis le réseau local
        server_port=7860,
        share=False,              # True pour tunnel Gradio public temporaire
        show_error=True,
    )
