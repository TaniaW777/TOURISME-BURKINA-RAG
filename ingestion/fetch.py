# -*- coding: utf-8 -*-
"""
ingestion/fetch.py
==================
Récupération HTTP respectueuse. Remplace la couche réseau de
data/scraping_script.py (qui reste intact tant que la migration
n'est pas validée).

Politique de sécurité — différences assumées avec l'ancien script :
  - SSL TOUJOURS vérifié (l'ancien script désactivait la vérification
    globalement via ssl._create_unverified_context : supprimé ici).
  - robots.txt respecté pour TOUTES les sources, sans liste blanche
    de contournement et sans `or True` (l'ancien can_fetch pouvait
    toujours renvoyer True).
  - Timeout et limite de taille sur chaque téléchargement.
  - Un échec de récupération est EXPLICITE : status != "ok", contenu
    vide interdit. Jamais de source silencieusement vide.

User-Agent : honnête (nom du projet + contact), pas un faux Chrome —
condition d'un crawl légitime et auditable.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib import robotparser
from urllib.parse import urlparse

import requests

# ──────────────────────────────────────────────
# LIMITES ET CONSTANTES
# ──────────────────────────────────────────────
REQUEST_TIMEOUT_S   = 45      # timeout par requête (connect+read)
PDF_TIMEOUT_S       = 60
MAX_HTML_BYTES      = 5 * 1024 * 1024    # 5 MB de HTML suffit largement
MAX_PDF_BYTES       = 25 * 1024 * 1024   # 25 MB max pour un PDF source
MAX_PDF_PAGES       = 120                # pages extraites au maximum
MIN_TEXT_CHARS      = 150                # en dessous : contenu jugé vide/inutile
REQUEST_DELAY_S     = 1.0                # délai entre requêtes, même hôte
ROBOTS_TIMEOUT_S    = 10

USER_AGENT = (
    "TourismeBurkinaRAG/2.0 (+projet éducatif open source ; "
    "respect robots.txt ; contact via https://github.com/TaniaW777/TOURISME-BURKINA-RAG)"
)


# ──────────────────────────────────────────────
# RÉSULTAT DE RÉCUPÉRATION — explicite, jamais silencieusement vide
# ──────────────────────────────────────────────
@dataclass
class FetchResult:
    url: str                      # URL demandée
    status: str                   # ok | blocked_robots | http_error |
                                 # too_large | extract_failed | invalid_content
    error: str = ""               # diagnostic lisible (jamais vide si status != ok)
    canonical_url: str = ""       # URL finale après redirections (provenance)
    content_type: str = ""
    raw_text: str = ""            # texte extrait, NON normalisé
    title: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ──────────────────────────────────────────────
# ROBOTS.TXT — mis en cache par origine, STRICT
# ──────────────────────────────────────────────
class RobotsCache:
    """Cache de robots.txt par origine.

    Politique stricte :
      - robots.txt lisible et interdit  →  blocked (aucune exception)
      - robots.txt inexistant (404)     →  autorisé (pratique standard)
      - robots.txt illisible (réseau)   →  bloqué (prudence)
    """

    def __init__(self, timeout_s: int = ROBOTS_TIMEOUT_S):
        self._cache: dict[str, robotparser.RobotFileParser | None] = {}
        self._timeout_s = timeout_s

    def allowed(self, url: str) -> tuple[bool, str]:
        origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._fetch_robots(origin)
        rp = self._cache[origin]
        if rp is None:
            return False, f"robots.txt illisible pour {origin} — accès prudent refusé"
        if not rp.can_fetch(USER_AGENT, url):
            return False, f"interdit par robots.txt de {origin}"
        return True, ""

    def _fetch_robots(self, origin: str) -> Optional[robotparser.RobotFileParser]:
        """Récupère et parse robots.txt avec NOTRE User-Agent honnête.

        On n'utilise PAS RobotFileParser.read() : il envoie l'UA par défaut
        "Python-urllib" que plusieurs sites (dont Wikipedia) répondent en
        403 — read() interprète alors 401/403 comme "tout interdire", ce qui
        ne reflète pas la politique réelle du site.
        """
        try:
            resp = requests.get(
                f"{origin}/robots.txt",
                timeout=self._timeout_s,
                headers={"User-Agent": USER_AGENT},
                verify=True,   # SSL vérifié, toujours
            )
        except Exception:
            return None  # robots injoignable (réseau/SSL) → prudent : refus

        rp = robotparser.RobotFileParser()
        if resp.status_code in (401, 403):
            # politique explicite du site : accès automatisé refusé
            rp.disallow_all = True
            return rp
        if resp.status_code >= 400:
            # robots absent (404...) : pas de restriction déclarée
            rp.allow_all = True
            return rp
        rp.parse(resp.text.splitlines())
        return rp


# ──────────────────────────────────────────────
# TÉLÉCHARGEMENT BORNÉ
# ──────────────────────────────────────────────
def _download_bounded(
    url: str,
    session: requests.Session,
    max_bytes: int,
    timeout_s: int,
) -> tuple[Optional[bytes], str, str, str]:
    """Télécharge en stream avec limite de taille.

    Retourne (contenu | None, erreur, content_type, canonical_url).
    SSL vérifié par défaut (verify=True implicite).
    """
    try:
        resp = session.get(
            url,
            timeout=timeout_s,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "fr,en;q=0.9"},
            stream=True,
            verify=True,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        size = int(resp.headers.get("Content-Length", 0) or 0)
        if size > max_bytes:
            return None, f"trop volumineux ({size} > {max_bytes} octets, Content-Length)", content_type, str(resp.url)

        buf = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            buf.write(chunk)
            if buf.tell() > max_bytes:
                return None, f"trop volumineux (> {max_bytes} octets en stream)", content_type, str(resp.url)
        return buf.getvalue(), "", content_type, str(resp.url)

    except requests.Timeout:
        return None, f"timeout après {timeout_s}s", "", url
    except requests.RequestException as e:
        return None, f"erreur HTTP : {e.__class__.__name__}: {e}", "", url


# ──────────────────────────────────────────────
# EXTRACTION WEB (trafilatura, fallback désactivé)
# ──────────────────────────────────────────────
def fetch_web(url: str, session: requests.Session, robots: RobotsCache) -> FetchResult:
    """Récupère et extrait une page web.

    L'ancien fallback BeautifulSoup (texte de page entière, navigation
    comprise) est supprimé : trafilatura fait déjà du fallback interne,
    et une page non extractible est une source ratée, pas du bruit.
    """
    allowed, reason = robots.allowed(url)
    if not allowed:
        return FetchResult(url=url, status="blocked_robots", error=reason)

    time.sleep(REQUEST_DELAY_S)  # politesse inter-requêtes
    body, err, ctype, canonical = _download_bounded(url, session, MAX_HTML_BYTES, REQUEST_TIMEOUT_S)
    if body is None:
        return FetchResult(url=url, status="http_error", error=err,
                           canonical_url=canonical, content_type=ctype)

    import trafilatura
    from bs4 import BeautifulSoup

    html = body.decode(resp_encoding_fallback(body, ctype), errors="replace")
    text = trafilatura.extract(html, include_comments=False, include_tables=True)

    # Titre depuis le HTML (provenance lisible pour les citations)
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()[:200]
    except Exception:
        pass

    if not text or len(text.strip()) < MIN_TEXT_CHARS:
        return FetchResult(
            url=url, status="extract_failed",
            error=f"extraction vide ou < {MIN_TEXT_CHARS} caractères",
            canonical_url=canonical, content_type=ctype, title=title,
        )
    return FetchResult(url=url, status="ok", canonical_url=canonical or url,
                       content_type=ctype, raw_text=text, title=title or url)


def resp_encoding_fallback(body: bytes, content_type: str) -> str:
    """Détermine l'encodage : header Content-Type, sinon détection UTF-8."""
    ctype_low = (content_type or "").lower()
    if "charset=" in ctype_low:
        return ctype_low.split("charset=")[-1].split(";")[0].strip()
    try:
        body.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"
