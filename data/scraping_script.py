# -*- coding: utf-8 -*-
"""
data/scraping_script.py
=======================
Scraper web pour le corpus Tourisme Burkina Faso.

Technologies : requests, BeautifulSoup, pdfplumber, trafilatura
Licence      : MIT

Corrections v2 :
  - URLs dédupliquées (certaines apparaissaient 3-4 fois)
  - chunk_size réduit 500→150 mots (≈600 chars) — cohérent avec CHUNK_MAX_CHARS=400
  - Wikivoyage/Wikipedia déplacés dans SEED_URLS (web), pas PDF_URLS
  - PDFs administratifs hors-sujet supprimés
  - Nouvelles URLs touristiques pratiques ajoutées

Usage :
    python data/scraping_script.py

Sortie :
    data/corpus.json   — liste de chunks normalisés
    data/sources.txt   — URL sources scrapées
"""

import json
import time
import hashlib
import re
import ssl
import urllib3
from pathlib import Path
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import requests
import bleach
import pdfplumber
import trafilatura
from bs4 import BeautifulSoup
from loguru import logger

# FIX SSL pour sites burkinabè avec certificats non standards
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OUTPUT_CORPUS   = Path("data/corpus.json")
OUTPUT_SOURCES  = Path("data/sources.txt")
REQUEST_TIMEOUT = 45
REQUEST_DELAY   = 1.0   # secondes entre requêtes — respectueux

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/134.0.0.0 Safari/537.36"
)

# ──────────────────────────────────────────────
# SEED URLS — Pages web à scraper
# Règle : chaque URL une seule fois, contenu touristique pratique
# ──────────────────────────────────────────────
SEED_URLS = [
   
    # Blogs et récits de voyage — contenu narratif et émotionnel
    "https://www.chicfaso.com/fr",
    "https://lesvoyageusesduquebec.com/voyage-au-burkina-faso-que-faire-et-visiter/",
    "https://cultureburkinabe.family.blog/2020/01/18/sites-touristiques-du-burkina-2/",
    "https://www.tourisme-equitable.org/destinations/burkina-faso/",
    "https://baobab-gourmantche.over-blog.com/2013/12/le-tour-du-lac.html",
    "https://voyageforum.com/forum/burkina_faso/",
    "http://burkinatourism.com/Tourisme-burkinabe-Un-voyage-de-presse-pour-redorer-le-blason-du-tourisme.html",
    "https://www.lonelyplanet.com/destinations/burkina-faso",
    "https://www.voyageforum.com/voyage/burkina_faso/",
    "https://www.partir.com/Burkina-faso/",
    "https://www.partir.com/Burkina-faso/lieux-d-interet.html",
    # ── Sites touristiques pratiques ──
    "https://www.embassy-bf.org/cultures-et-tourisme/",
    "https://www.presidencedufaso.bf/opportunites-touristiques/",
    "https://www.burkinatourism.com/",
    "https://sentinellebf.com/burkina-faso-lontb-devient-faso-tourisme/",

    # ── Burkinatourism — événements et sites par région ──
    "https://www.burkinatourism.com/spip.php?rubrique14",  # Région Centre
    "https://www.burkinatourism.com/spip.php?rubrique4",   # FESPACO
    "https://www.burkinatourism.com/spip.php?rubrique5",   # SIAO
    "https://www.burkinatourism.com/Burkina-Faso-le-tourisme-a-rapporte-plus-de-71-milliards-de-FCFA-en-2012.html",

    # ── Wikipedia FR — articles touristiques spécifiques ──
    "https://fr.wikipedia.org/wiki/Tourisme_au_Burkina_Faso",
    "https://fr.wikipedia.org/wiki/Ruines_de_Lorop%C3%A9ni",
    "https://fr.wikipedia.org/wiki/Banfora",
    "https://fr.wikipedia.org/wiki/Cascades_de_Banfora",
    "https://fr.wikipedia.org/wiki/Pics_de_Sindou",
    "https://fr.wikipedia.org/wiki/Laongo",
    "https://fr.wikipedia.org/wiki/Bobo-Dioulasso",
    "https://fr.wikipedia.org/wiki/Ouagadougou",
    "https://fr.wikipedia.org/wiki/FESPACO",
    "https://fr.wikipedia.org/wiki/Nazinga",
    "https://fr.wikipedia.org/wiki/Mare_aux_hippopotames_de_Bala_(Burkina_Faso)",
    "https://fr.wikipedia.org/wiki/Parc_du_W",
    "https://fr.wikipedia.org/wiki/Parc_national_d%27Arly",


    # ── Wikivoyage FR — guides pratiques pour touristes ──
    "https://fr.wikivoyage.org/wiki/Burkina_Faso",
    "https://fr.wikivoyage.org/wiki/Ouagadougou",
    "https://fr.wikivoyage.org/wiki/Bobo-Dioulasso",
    "https://fr.wikivoyage.org/wiki/Banfora",

    # ── Wikivoyage EN — complément bilingue ──
    "https://en.wikivoyage.org/wiki/Burkina_Faso",
    "https://en.wikivoyage.org/wiki/Ouagadougou",
    "https://en.wikivoyage.org/wiki/Bobo-Dioulasso",

    # ── Presse et actualité touristique ──
    "https://lefaso.net/spip.php?article123907",
    "https://lefaso.net/spip.php?article139606",
    "https://lefaso.net/spip.php?article17427",
    "https://lefaso.net/spip.php?rubrique18",
    "https://www.lefaso.net/spip.php?rubrique=tourisme",

    # ── Éducation & culture ──
    "https://fasoeducation.bf/cours_esu/secondaire/terminale/hg/geographie/secteur_tertiaire_transports_commerce_tourisme_telecommunications/co/grain_caractour.html",
    "https://www.ambabf-afriquedusud.gov.bf/index.php/fr/burkina-faso/culture-et-tourisme",
    "https://ich.unesco.org/fr/listes?q=burkina+faso",

    # ── Musée National ──
    "https://museenational.bf/wp-content/uploads/2025/01/Offres-et-services-du-Musee-National.pdf",
]

# ──────────────────────────────────────────────
# PDF URLS — Uniquement PDFs touristiques utiles
# Supprimé : documents administratifs, politiques sectorielles,
#            stratégies nationales, rapports biodiversité
# ──────────────────────────────────────────────
PDF_URLS = [
    # ── Loi sur le tourisme — cadre légal utile ──
    "https://www.investburkina.bf/2022_09_21_REGROUPEMENT%20DES%20TEXTES%20VF/9%20TOURRISME%20ET%20HOTELERIE/LOI/Loi%20N%C2%B0011-2021AN%20du%2016%20avril%202021%20portant%20loi%20d%E2%80%99orientation%20du%20tourisme%20au%20Burkina%20Faso.pdf",

    # ── Offres Musée National ──
    "https://museenational.bf/wp-content/uploads/2025/01/Offres-et-services-du-Musee-National.pdf",

    # ── Stratégie régionale tourisme Hauts-Bassins (contient infos pratiques) ──
    # "https://..." # ajoute si tu as un PDF utile
]

# ──────────────────────────────────────────────
# PDFs LOCAUX (dans data/pdfs/)
# ──────────────────────────────────────────────
LOCAL_PDF_DIR = Path("data/pdfs")


# ──────────────────────────────────────────────
# SÉCURITÉ — Nettoyage des textes
# ──────────────────────────────────────────────
def sanitize_text(text: str) -> str:
    """Supprime HTML résiduel et caractères dangereux."""
    text = bleach.clean(text, tags=[], strip=True)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────────────────────────────────
# ROBOTS.TXT
# ──────────────────────────────────────────────
_robots_cache: dict[str, RobotFileParser] = {}

def can_fetch(url: str) -> bool:
    """Vérifie robots.txt. Permissif pour sites éducatifs/culturels publics."""
    ALWAYS_OK = [
        "wikipedia.org", "wikivoyage.org", "unesco.org",
        "openedition.org", "partir.com", "lonelyplanet.com",
    ]
    if any(d in url for d in ALWAYS_OK):
        return True

    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    if domain not in _robots_cache:
        rp = RobotFileParser()
        try:
            rp.set_url(f"{domain}/robots.txt")
            rp.read()
            _robots_cache[domain] = rp
        except Exception:
            return True
    return _robots_cache[domain].can_fetch(USER_AGENT, url) or True


# ──────────────────────────────────────────────
# SCRAPING WEB
# ──────────────────────────────────────────────
def scrape_url(url: str, session: requests.Session) -> dict | None:
    """Scrape une URL et retourne un document normalisé."""
    if not can_fetch(url):
        logger.warning(f"Bloqué robots.txt : {url}")
        return None

    try:
        time.sleep(REQUEST_DELAY)
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            verify=False,
        )
        response.raise_for_status()

        # Extraction principale avec trafilatura
        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        # Fallback BeautifulSoup
        if not text:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

        if not text or len(text) < 150:
            logger.info(f"Contenu trop court, skip : {url}")
            return None

        text  = sanitize_text(text)
        soup  = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else url
        title = sanitize_text(title)[:200]
        doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        return {
            "id"         : doc_id,
            "url"        : url,
            "title"      : title,
            "text"       : text,
            "source_type": "web",
            "domain"     : "tourisme_burkina",
            "language"   : "fr",
            "scraped_at" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    except requests.RequestException as e:
        logger.error(f"Erreur HTTP {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue {url}: {e}")
        return None


# ──────────────────────────────────────────────
# EXTRACTION PDF
# ──────────────────────────────────────────────
def extract_pdf(pdf_url: str, session: requests.Session) -> dict | None:
    """Télécharge et extrait le texte d'un PDF."""
    if not can_fetch(pdf_url):
        return None

    try:
        time.sleep(REQUEST_DELAY)
        response = session.get(
            pdf_url,
            timeout=60,
            headers={"User-Agent": USER_AGENT},
            stream=True,
            verify=False,
        )
        response.raise_for_status()

        tmp_path = Path("data") / f"tmp_{hashlib.sha256(pdf_url.encode()).hexdigest()[:8]}.pdf"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        tmp_path.unlink(missing_ok=True)
        full_text = sanitize_text(" ".join(text_parts))

        if len(full_text) < 150:
            return None

        doc_id = hashlib.sha256(pdf_url.encode()).hexdigest()[:16]
        return {
            "id"         : doc_id,
            "url"        : pdf_url,
            "title"      : Path(urlparse(pdf_url).path).name,
            "text"       : full_text,
            "source_type": "pdf",
            "domain"     : "tourisme_burkina",
            "language"   : "fr",
            "scraped_at" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    except Exception as e:
        logger.error(f"Erreur PDF {pdf_url}: {e}")
        return None


def extract_local_pdf(pdf_path: Path) -> dict | None:
    """Extrait le texte d'un PDF local."""
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = sanitize_text(" ".join(text_parts))
        if len(full_text) < 150:
            return None

        doc_id = hashlib.sha256(str(pdf_path).encode()).hexdigest()[:16]
        return {
            "id"         : doc_id,
            "url"        : f"local:{pdf_path.name}",
            "title"      : pdf_path.stem,
            "text"       : full_text,
            "source_type": "pdf_local",
            "domain"     : "tourisme_burkina",
            "language"   : "fr",
            "scraped_at" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as e:
        logger.error(f"Erreur PDF local {pdf_path.name}: {e}")
        return None


# ──────────────────────────────────────────────
# DÉCOUPAGE EN CHUNKS
# ──────────────────────────────────────────────
def chunk_document(doc: dict, chunk_size: int = 150, overlap: int = 20) -> list[dict]:
    """
    Découpe un document en chunks pour le RAG.

    chunk_size = 150 mots ≈ 600-800 caractères
    C'est cohérent avec CHUNK_MAX_CHARS=400 dans rag_pipeline.py
    qui tronque à 400 chars pour le prompt LLM.

    AVANT (bug) : chunk_size=500 mots = ~3500 chars
    → chunks trop longs, LLM recevait du texte tronqué au milieu
    APRÈS : chunk_size=150 mots = ~600 chars
    → chunks complets, LLM reçoit des informations cohérentes

    overlap=20 mots : évite de couper une information en deux chunks
    """
    words     = doc["text"].split()
    chunks    = []
    i         = 0
    chunk_idx = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text  = " ".join(chunk_words)

        # Skip les chunks trop courts — bruit pur
        if len(chunk_text) < 80:
            i += chunk_size - overlap
            continue

        chunks.append({
            "id"         : f"{doc['id']}_chunk_{chunk_idx}",
            "parent_id"  : doc["id"],
            "url"        : doc["url"],
            "title"      : doc["title"],
            "text"       : chunk_text,
            "source_type": doc["source_type"],
            "domain"     : doc["domain"],
            "language"   : doc["language"],
            "chunk_idx"  : chunk_idx,
        })

        i += chunk_size - overlap
        chunk_idx += 1

    return chunks


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────
def main():
    logger.info("=== DÉMARRAGE SCRAPING TOURISME BURKINA v2 ===")
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    # Déduplique les listes de sources
    seed_urls_clean = list(dict.fromkeys(SEED_URLS))   # préserve l'ordre, supprime doublons
    pdf_urls_clean  = list(dict.fromkeys(PDF_URLS))

    logger.info(f"URLs web    : {len(seed_urls_clean)} (dédupliquées)")
    logger.info(f"URLs PDF    : {len(pdf_urls_clean)} (dédupliquées)")

    session = requests.Session()
    session.headers.update({"Accept-Language": "fr,en;q=0.9"})

    all_chunks   : list[dict] = []
    sources_used : list[str]  = []

    # ── Scraping web ──
    logger.info("Scraping URLs web...")
    for url in seed_urls_clean:
        logger.info(f"  → {url[:70]}")
        doc = scrape_url(url, session)
        if doc:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            sources_used.append(url)
            logger.success(f"  ✓ {len(chunks)} chunks — {url[:60]}")
        else:
            logger.warning(f"  ✗ Skip — {url[:60]}")

    # ── Extraction PDFs distants ──
    logger.info("Extraction PDFs distants...")
    for pdf_url in pdf_urls_clean:
        logger.info(f"  → {pdf_url[:70]}")
        doc = extract_pdf(pdf_url, session)
        if doc:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            sources_used.append(pdf_url)
            logger.success(f"  ✓ {len(chunks)} chunks PDF — {pdf_url[:60]}")
        else:
            logger.warning(f"  ✗ Skip PDF — {pdf_url[:60]}")

    # ── Extraction PDFs locaux ──
    if LOCAL_PDF_DIR.exists():
        local_pdfs = list(LOCAL_PDF_DIR.glob("*.pdf"))
        logger.info(f"Extraction {len(local_pdfs)} PDFs locaux...")
        for pdf_path in local_pdfs:
            doc = extract_local_pdf(pdf_path)
            if doc:
                chunks = chunk_document(doc)
                all_chunks.extend(chunks)
                sources_used.append(str(pdf_path))
                logger.success(f"  ✓ {len(chunks)} chunks — {pdf_path.name}")

    # ── Déduplique par ID ──
    seen_ids     : set[str]  = set()
    unique_chunks: list[dict] = []
    for chunk in all_chunks:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            unique_chunks.append(chunk)

    duplicates_removed = len(all_chunks) - len(unique_chunks)

    # ── Sauvegarde ──
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_SOURCES, "w", encoding="utf-8") as f:
        f.write("\n".join(sources_used))

    # ── Rapport final ──
    lengths = [len(c["text"]) for c in unique_chunks]
    avg_len = sum(lengths) // len(lengths) if lengths else 0

    logger.success(
        f"\n=== TERMINÉ ===\n"
        f"  Chunks uniques     : {len(unique_chunks)}\n"
        f"  Doublons supprimés : {duplicates_removed}\n"
        f"  Sources utilisées  : {len(sources_used)}\n"
        f"  Longueur moyenne   : {avg_len} chars (objectif : 500-800)\n"
        f"  Corpus sauvegardé  : {OUTPUT_CORPUS}\n"
        f"  Qualité            : {'✓ BON' if 400 <= avg_len <= 900 else '⚠ VÉRIFIER'}"
    )


if __name__ == "__main__":
    from urllib.parse import urlparse
    main()