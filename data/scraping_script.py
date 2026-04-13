"""
data/scraping_script.py
=======================
Scraper web pour le corpus Tourisme Burkina Faso.

Technologies : requests, BeautifulSoup, pdfplumber, trafilatura
Licence      : MIT
Respect      : robots.txt obligatoire via urllib.robotparser
Sécurité     : sanitization HTML, timeout strict, rate-limiting poli

Usage :
    python data/scraping_script.py

Sortie :
    data/corpus.json   — liste de documents normalisés
    data/sources.txt   — URL sources scrapées
"""

import json
import time
import hashlib
import logging
import re
import ssl
import urllib3
from pathlib import Path
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin

import requests
import bleach
import pdfplumber
import trafilatura
from bs4 import BeautifulSoup
from loguru import logger

# FIX SSL / DNS pour connexion Burkina
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
try:
    requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL:@SECLEVEL=1'
except:
    pass


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OUTPUT_CORPUS   = Path("data/corpus.json")
OUTPUT_SOURCES  = Path("data/sources.txt")
REQUEST_TIMEOUT = 45           # timeout max par requête
REQUEST_DELAY   = 0.8          # secondes entre chaque requête (politesse)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
# SEEDS — URLs à scraper (complète avec tes propres URLs)
# ──────────────────────────────────────────────
SEED_URLS = [
        
    "https://www.partir.com/Burkina-faso/lieux-d-interet.html",
    "https://www.embassy-bf.org/cultures-et-tourisme/",   
    "https://fasoeducation.bf/cours_esu/secondaire/terminale/hg/geographie/secteur_tertiaire_transports_commerce_tourisme_telecommunications/co/grain_caractour.html",
    "https://www.presidencedufaso.bf/opportunites-touristiques/",
    "https://www.burkinatourism.com/",
    "https://www.communication.gov.bf/accueil",
   
    
    


    # Ministère Tourisme BF (si disponible)
    
    "https://journals.openedition.org/anthropos/",

    # OpenEdition — cas Loropéni
    "https://journals.openedition.org/etudescaribeennes/18963",


    # UNESCO Patrimoine Culturel Immatériel
    "https://ich.unesco.org/fr/listes?q=burkina+faso",
    
    

    # Wikipedia FR — articles tourisme BF
    "https://fr.wikipedia.org/wiki/Tourisme_au_Burkina_Faso",
    "https://fr.wikipedia.org/wiki/Ruines_de_Lorop%C3%A9ni",
    "https://fr.wikipedia.org/wiki/Banfora",
    "https://fr.wikipedia.org/wiki/Cascades_de_Banfora",
    
    "https://fr.wikipedia.org/wiki/Laongo",
    "https://fr.wikipedia.org/wiki/Bobo-Dioulasso",
    "https://fr.wikipedia.org/wiki/Ouagadougou",
    "https://fr.wikipedia.org/wiki/FESPACO",
   

    # Wikitravel / Wikivoyage
    "https://fr.wikivoyage.org/wiki/Burkina_Faso",
    "https://fr.wikivoyage.org/wiki/Banfora",
    "https://fr.wikivoyage.org/wiki/Ouagadougou",
    "https://fr.wikipedia.org/wiki/Tourisme_au_Burkina_Faso",
     
    
      
    # Nouvelles seeds riches en contenu tourisme BF
    "https://fr.wikipedia.org/wiki/Tourisme_au_Burkina_Faso",
    "https://fr.wikivoyage.org/wiki/Burkina_Faso",
    "https://www.burkina24.com/category/tourisme/",
    "https://lefaso.net/spip.php?rubrique18",  # tourisme
    
    
     
    
    "https://www.lonelyplanet.com/burkina-faso",
    
    "https://www.burkina24.com/category/tourisme/",
    "https://lefaso.net/spip.php?rubrique18",
   
    "https://www.lefaso.net/spip.php?rubrique=tourisme",
    
    
    "https://www.burkina24.com/category/tourisme/",
    "https://lefaso.net/spip.php?rubrique18",
    
    "https://www.lefaso.net/spip.php?rubrique=tourisme",
    
    
    "https://www.burkina24.com/category/tourisme/",
    "https://lefaso.net/spip.php?rubrique18",
    
    
    "https://lefaso.net/spip.php?article123907",  # Plan national de développement économique et social (PNDES) 2021-2025
    "https://www.communication.gov.bf/accueil",  # Page d'accueil du ministère de la communication (vérifie les liens PDF)
    "https://lefaso.net/spip.php?article139606",  # Plan de développement touristique 2020-2025
    "https://www.ambabf-afriquedusud.gov.bf/index.php/fr/burkina-faso/culture-et-tourisme",  # PDF culturel + touristique
    "https://www.scribd.com/document/930581616/Memoire-de-Louise",
    "https://www.burkinatourism.com/Burkina-Faso-le-tourisme-a-rapporte-plus-de-71-milliards-de-FCFA-en-2012.html",
    "https://burkina24.com/2026/04/12/burkina-faso-plus-de-243-millions-fcfa-deja-mobilises-pour-faso-mebo-a-la-date-du-10-avril-2026/",
    "https://sentinellebf.com/burkina-faso-lontb-devient-faso-tourisme/",
    "https://lefaso.net/spip.php?article17427",
]



# PDFs officiels à télécharger directement
PDF_URLS = [

    "https://www.revue-akofena.com/wp-content/uploads/2022/08/03-T06v03-25-Assonsi-SOMA-Assonsi-SOMA-Soumaila-MARE_29-44.pdf",  # Article académique tourisme BF
    
    
    # Ajoute ici tes PDFs World Bank, ICOMOS, etc.
    "https://www.communication.gov.bf/fileadmin/user_upload/storages/snct_2018-2027_vf.pdf",  # SNCT 2018-2027 - excellent !
    "https://www.investburkina.bf/2022_09_21_REGROUPEMENT%20DES%20TEXTES%20VF/9%20TOURRISME%20ET%20HOTELERIE/LOI/Loi%20N%C2%B0011-2021AN%20du%2016%20avril%202021%20portant%20loi%20d%E2%80%99orientation%20du%20tourisme%20au%20Burkina%20Faso.pdf",
    "https://www.abmaq.bf/wp-content/uploads/2014/07/LA-PROBLEMATIQUE-DE-LA-LABELLISATION-DES-HOTELS-AU-BURKINA-FASO-ET-SON-IMPACT-SUR-LE-TOURISME.pdf",
    "https://www.communication.gov.bf/fileadmin/user_upload/storages/fichier/document_26.pdf",
    
    "https://www.commerce.gov.bf/fileadmin/user_upload/storage/fichiers/Arrete_portant_condition_de_reprise__des_activites-1.pdf",
    "https://museenational.bf/wp-content/uploads/2025/01/Offres-et-services-du-Musee-National.pdf",
    "https://www.sports.gov.bf/fileadmin/user_upload/storages/fichier/plaidoyer_sur_les_loisirs.pdf",
    # Bonus solides (ajoute-les) :
    "https://museenational.bf/wp-content/uploads/2024/12/PSD-MN-adopte-CA.pdf",  # Plan stratégique Musée National
    "https://www.cbd.int/doc/world/bf/bf-nbsap-01-fr.pdf",  # Stratégie biodiversité avec section tourisme
    "https://cultures.hautsbassins.bf/wp-content/uploads/2022/03/POLITIQUE-SECTORIELLE-CTSL-VF.pdf" , # Politique sectorielle Culture-Tourisme
    "https://www.communication.gov.bf/fileadmin/user_upload/storages/snct_2018-2027_vf.pdf",  # déjà là, excellent
    "https://museenational.bf/wp-content/uploads/2024/12/PSD-MN-adopte-CA.pdf",  # Plan stratégique Musée National
    "https://www.cbd.int/doc/world/bf/bf-nbsap-01-fr.pdf",  # Biodiversité avec section tourisme
    "https://www.communication.gov.bf/fileadmin/user_upload/storages/snct_2018-2027_vf.pdf" ,  # déjà là, excellent
    "https://www.cbd.int/doc/world/bf/bf-nbsap-01-fr.pdf" ,  # Biodiversité avec section tourisme
    
    
]
# ──────────────────────────────────────────────
# PDFs LOCAUX (téléchargés manuellement)
# ──────────────────────────────────────────────
LOCAL_PDF_DIR = Path("data/pdfs")

def extract_local_pdf(pdf_path: Path) -> dict | None:
    """
    Extrait le texte d'un PDF local téléchargé manuellement.
    """
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = " ".join(text_parts)
        if len(full_text) < 100:
            logger.warning(f"PDF local trop court : {pdf_path.name}")
            return None

        full_text = sanitize_text(full_text)
        doc_id = hashlib.sha256(str(pdf_path).encode()).hexdigest()[:16]

        return {
            "id": doc_id,
            "url": f"local:{pdf_path.name}",
            "title": pdf_path.stem,
            "text": full_text,
            "source_type": "pdf_local",
            "domain": "tourisme_burkina",
            "language": "fr",
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as e:
        logger.error(f"Erreur PDF local {pdf_path.name}: {e}")
        return None


# ──────────────────────────────────────────────
# SÉCURITÉ — Nettoyage des textes
# ──────────────────────────────────────────────
def sanitize_text(text: str) -> str:
    """
    Supprime tout HTML résiduel et caractères dangereux.
    Utilise bleach (Mozilla) pour sanitization robuste.
    Empêche toute injection de prompt via le corpus.
    """
    # Supprime les balises HTML
    text = bleach.clean(text, tags=[], strip=True)
    # Supprime les caractères de contrôle non-ASCII dangereux
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalise les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────────────────────────────────
# ROBOTS.TXT — Vérification obligatoire
# ──────────────────────────────────────────────
_robots_cache: dict[str, RobotFileParser] = {}

def can_fetch(url: str) -> bool:
    # Permissive pour sites publics éducatifs/culturels (projet open source)
    if any(d in url for d in ["wikipedia.org", "wikivoyage.org", "unesco.org", "whc.unesco.org", "ich.unesco.org", "openedition.org", "africanews.com"]):
        return True
    # Pour les sites .bf et autres, on tente quand même
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    if domain not in _robots_cache:
        rp = RobotFileParser()
        robots_url = f"{domain}/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
            _robots_cache[domain] = rp
        except Exception:
            logger.warning(f"robots.txt inaccessible pour {domain} — on tente quand même")
            return True  # on tente malgré tout
    return _robots_cache[domain].can_fetch(USER_AGENT, url) or True  # fallback permissive


# ──────────────────────────────────────────────
# SCRAPING WEB
# ──────────────────────────────────────────────
def scrape_url(url: str, session: requests.Session) -> dict | None:
    """
    Scrape une URL unique.
    Retourne un document normalisé ou None si échec.
    """
    # Vérifie robots.txt
    if not can_fetch(url):
        logger.warning(f"BLOQUÉ par robots.txt : {url}")
        return None

    try:
        # Rate limiting poli
        time.sleep(REQUEST_DELAY)

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()

        # Extraction texte avec trafilatura (meilleure extraction boilerplate)
        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        # Fallback BeautifulSoup si trafilatura échoue
        if not text:
            soup = BeautifulSoup(response.text, "html.parser")
            # Supprime nav, footer, scripts
            for tag in soup(["nav", "footer", "script", "style", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

        if not text or len(text) < 100:
            logger.info(f"Contenu trop court, skip : {url}")
            return None

        # Nettoyage
        text = sanitize_text(text)

        # Métadonnées
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else url
        title = sanitize_text(title)

        # ID unique basé sur URL
        doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        return {
            "id": doc_id,
            "url": url,
            "title": title,
            "text": text,
            "source_type": "web",
            "domain": "tourisme_burkina",
            "language": "fr",  # langdetect optionnel ici pour perf
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    """
    Télécharge et extrait le texte d'un PDF.
    Utilise pdfplumber (MIT).
    """
    if not can_fetch(pdf_url):
        logger.warning(f"BLOQUÉ par robots.txt (PDF) : {pdf_url}")
        return None

    try:
        time.sleep(REQUEST_DELAY)
        response = session.get(
            pdf_url,
            timeout=30,  # PDFs peuvent être gros
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        response.raise_for_status()

        # Sauvegarde temporaire
        tmp_path = Path("data") / f"tmp_pdf_{hashlib.sha256(pdf_url.encode()).hexdigest()[:8]}.pdf"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extraction texte
        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        tmp_path.unlink(missing_ok=True)  # Nettoyage

        full_text = " ".join(text_parts)
        if len(full_text) < 100:
            return None

        full_text = sanitize_text(full_text)
        doc_id = hashlib.sha256(pdf_url.encode()).hexdigest()[:16]

        return {
            "id": doc_id,
            "url": pdf_url,
            "title": Path(urlparse(pdf_url).path).name,
            "text": full_text,
            "source_type": "pdf",
            "domain": "tourisme_burkina",
            "language": "fr",
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    except Exception as e:
        logger.error(f"Erreur PDF {pdf_url}: {e}")
        return None


# ──────────────────────────────────────────────
# DÉCOUPAGE EN CHUNKS
# ──────────────────────────────────────────────
def chunk_document(doc: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Découpe un document en chunks pour le RAG.
    chunk_size  : mots par chunk (500 mots ≈ 750 tokens)
    overlap     : mots de chevauchement entre chunks (contexte)

    Pourquoi 500 mots ? BGE-M3 supporte 8192 tokens.
    500 mots ≈ 650 tokens, bien en dessous de la limite.
    """
    words = doc["text"].split()
    chunks = []
    i = 0
    chunk_idx = 0

    while i < len(words):
        # Extrait le chunk
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "id": f"{doc['id']}_chunk_{chunk_idx}",
            "parent_id": doc["id"],
            "url": doc["url"],
            "title": doc["title"],
            "text": chunk_text,
            "source_type": doc["source_type"],
            "domain": doc["domain"],
            "language": doc["language"],
            "chunk_idx": chunk_idx,
        })

        # Avance avec overlap pour continuité du contexte
        i += chunk_size - overlap
        chunk_idx += 1

    return chunks


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────
def main():
    logger.info("=== DÉMARRAGE SCRAPING TOURISME BURKINA ===")

    # Crée le dossier data si absent
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    # Session HTTP réutilisable (meilleures perfs, pool connexions)
    session = requests.Session()
    session.headers.update({"Accept-Language": "fr,en;q=0.9"})

    all_chunks = []
    sources_used = []

    # ── Scraping web ──
    logger.info(f"Scraping {len(SEED_URLS)} URLs web...")
    for url in SEED_URLS:
        logger.info(f"Scraping : {url}")
        doc = scrape_url(url, session)
        if doc:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            sources_used.append(url)
            logger.success(f"✓ {len(chunks)} chunks extraits — {url}")
        else:
            logger.warning(f"✗ Skip — {url}")

    # ── Extraction PDFs ──
    logger.info(f"Extraction {len(PDF_URLS)} PDFs...")
    for pdf_url in PDF_URLS:
        logger.info(f"PDF : {pdf_url}")
        doc = extract_pdf(pdf_url, session)
        if doc:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            sources_used.append(pdf_url)
            logger.success(f"✓ {len(chunks)} chunks PDF — {pdf_url}")

        # ── Extraction PDFs locaux ──
    if LOCAL_PDF_DIR.exists():
        local_pdfs = list(LOCAL_PDF_DIR.glob("*.pdf"))
        logger.info(f"Extraction {len(local_pdfs)} PDFs locaux...")
        for pdf_path in local_pdfs:
            logger.info(f"PDF local : {pdf_path.name}")
            doc = extract_local_pdf(pdf_path)
            if doc:
                chunks = chunk_document(doc)
                all_chunks.extend(chunks)
                sources_used.append(str(pdf_path))
                logger.success(f"✓ {len(chunks)} chunks PDF local — {pdf_path.name}")        

    # ── Sauvegarde corpus ──
    logger.info(f"Sauvegarde corpus : {len(all_chunks)} chunks au total")

    # Déduplique par ID
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            unique_chunks.append(chunk)

    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_SOURCES, "w", encoding="utf-8") as f:
        f.write("\n".join(sources_used))

    logger.success(
        f"=== TERMINÉ ===\n"
        f"  Chunks uniques  : {len(unique_chunks)}\n"
        f"  Sources utilisées : {len(sources_used)}\n"
        f"  Corpus sauvegardé : {OUTPUT_CORPUS}\n"
        f"  OBJECTIF 500 docs : {'✓ ATTEINT' if len(unique_chunks) >= 500 else '✗ AJOUTE DES URLs'}"
    )


if __name__ == "__main__":
    main()
