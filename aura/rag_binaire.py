"""RAG binaire local : retrouver au lieu de deviner, en quelques ms.

Cascade des sources de savoir (l'IA ne parle QUE depuis une source) :
  1. NIVEAU 0 (cache de reponses)      ~0 ms
  2. RAG BINAIRE (cette module)        ~1-5 ms   <- nouveau maillon
  3. GRAPHE DE FAITS (MiniRAG-lite)    ~1 ms
  4. WEB « Knuckles Go » (DuckDuckGo)  ~1-2 s
  5. Abstention honnete

Pourquoi binaire : un JSONL de questions/reponses doit etre parse ligne
par ligne a chaque question (et son loader `gguf` pese 100+ Mo). Ici,
tout est precompile dans `rag.bin` : en-tete + signatures MinHash 64 octets
par document + un bloc LZMA de documents. Chargement = lecture d'un flux
d'octets et comparaisons de bits. Recherche = LSH : des signatures qui
partagent un bandeau votent ensemble -> ~5 ms pour des dizaines de milliers
de documents, ~10 Mo d'empreintes en RAM pour 100 000 documents.

Zero dependance : MinHash et le hachage sont codes ici (pas de datasketch).
"""
import json
import logging
import lzma
import os
import struct
import threading
import time
import unicodedata
from collections import defaultdict
from hashlib import blake2b

LOG = logging.getLogger("aura.rag")

MAGIQUE = b"RAG1"
VERSION = 1
NB_BANDS = 16          # bandeaux LSH (64 octets -> 16 x 4 octets)
NB_PERM = NB_BANDS * 4  # 64 fonctions de hachage = 4 par bandeau

_STOPWORDS = frozenset(
    "le la les un une des de du au aux et ou est sont que qui quoi dont ni "
    "mais car donc pour par sur dans avec sans sous entre vers chez je tu il "
    "elle on nous vous ils elles me te se lui leur y en mon ton son ma ta sa "
    "mes tes ses ce cet cette ces quel quelle combien comment pourquoi quand "
    "etre avoir a ai as suis es plus moins tres peu tout tous toute meme "
    "aussi alors alors voila voici ici dis moi dites peux tu peux-tu stp "
    "svp merci faire fais fait etre".split()
)
_MOTS = None  # compile paresseusement


def _regex_mots():
    global _MOTS
    if _MOTS is None:
        _MOTS = __import__("re").compile(r"[a-z0-9àâäéèêëïîôöùûüç]{2,}")
    return _MOTS


def tokeniser(texte: str) -> frozenset[str]:
    """Mots normalises (sans accents), stopwords exclus."""
    t = unicodedata.normalize("NFKD", texte.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return frozenset(m for m in _regex_mots().findall(t)
                     if m not in _STOPWORDS and len(m) >= 2)


def _h64(donnees: bytes, graine: int) -> int:
    """Blake2b 64 bits, sale par la graine du permutation i."""
    return int.from_bytes(
        blake2b(donnees + graine.to_bytes(2, "little"),
                digest_size=8).digest(), "little")


def signature_minhash(tokens: frozenset[str]) -> bytes:
    """64 octets = 16 bandeaux de 4 octets (LSH)."""
    if not tokens:
        return bytes(64)
    # dizaines de milliards de combinaisons de 50+ tokens : on tire la
    # graine du token lui-meme (PAS du hachage du token + indice i)
    # -> reproductible, et chaque token reste minuscule a hacher.
    sig = bytearray(NB_PERM * 8)
    for tok in tokens:
        hb = blake2b(tok.encode("utf-8"), digest_size=8).digest()
        base = int.from_bytes(hb, "little")
        for i in range(NB_PERM):
            v = base ^ (i * 0x9E3779B97F4A7C15)      # golden ratio scrambling
            v = (v ^ (v >> 29)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
            v = (v ^ (v >> 32)) & 0xFFFFFFFFFFFFFFFF
            off = i * 8
            old = int.from_bytes(sig[off:off + 8], "little")
            if v < old or old == 0:
                sig[off:off + 8] = v.to_bytes(8, "little")
    # reduction : 64 octets -> 16 bandeaux de 4 octets (min de 4 perms)
    bandes = bytearray(NB_BANDS * 4)
    for b in range(NB_BANDS):
        m = min(int.from_bytes(sig[b * 4 * 8 + j * 8: b * 4 * 8 + j * 8 + 8],
                               "little") & 0xFFFFFFFF
                for j in range(4))
        bandes[b * 4:b * 4 + 4] = m.to_bytes(4, "little")
    return bytes(bandes)


class IndexRAG:
    """Index binaire en memoire : signatures + documents (LZMA)."""

    def __init__(self) -> None:
        self._docs: list[dict] = []           # {q, a, src, tag}
        self._sigs: list[bytes] = []
        self._lsh: dict[bytes, list[int]] = defaultdict(list)
        self._verrou = threading.Lock()
        self._charge = False

    # -- chargement ---------------------------------------------------------

    def charger_bytes(self, brut: bytes) -> None:
        with self._verrou:
            self._docs.clear()
            self._sigs.clear()
            self._lsh.clear()
            if brut[:4] != MAGIQUE:
                raise ValueError("rag.bin : signature invalide")
            version, n_docs, off_docs = struct.unpack_from("<HII", brut, 4)
            if version != VERSION:
                raise ValueError(f"rag.bin version {version} != {VERSION}")
            bloc_docs = lzma.decompress(brut[off_docs:])
            docs = json.loads(bloc_docs.decode("utf-8"))
            curseur = 4 + struct.calcsize("<HII")
            for i in range(n_docs):
                self._sigs.append(brut[curseur + i * 64:
                                       curseur + (i + 1) * 64])
            self._docs = docs
            for i, sig in enumerate(self._sigs):
                for b in range(NB_BANDS):
                    self._lsh[sig[b * 4:b * 4 + 4]].append(i)
            self._charge = True
            LOG.info("[rag] %d documents charges", len(self._docs))

    def _assurer(self) -> None:
        if self._charge:
            return
        # 1) paquet embedded (forge dans le code) — marche partout, meme .aef
        try:
            from . import _rag_data          # type: ignore[attr-defined]
            self.charger_bytes(_rag_data.BINAIRE)
            return
        except Exception:
            pass
        # 2) rag.bin a cote du paquet
        chemin = os.path.join(os.path.dirname(__file__), "rag.bin")
        if os.path.exists(chemin):
            with open(chemin, "rb") as f:
                self.charger_bytes(f.read())

    # -- recherche ----------------------------------------------------------

    def chercher(self, question: str, seuil: float = 0.35,
                 top: int = 1) -> str | None:
        """Meilleure reponse si similarite >= seuil, sinon None (-> cascade)."""
        self._assurer()
        if not self._charge or not self._docs:
            return None
        debut = time.perf_counter()
        sig_q = signature_minhash(tokeniser(question))
        # votes LSH : les docs qui partagent au moins un bandeau
        candidats: set[int] = set()
        for b in range(NB_BANDS):
            candidats.update(self._lsh.get(sig_q[b * 4:b * 4 + 4], ()))
        if not candidats:
            LOG.info("[rag] miss LSH (%.1f ms)",
                     (time.perf_counter() - debut) * 1000)
            return None
        # score reel sur les candidats seulement (Jaccard approxime)
        toks_q = tokeniser(question)
        meilleur, meilleur_score = None, 0.0
        for i in candidats:
            toks_d = tokeniser(self._docs[i].get("q", ""))
            if not toks_d or not toks_q:
                continue
            inter = len(toks_q & toks_d)
            union = len(toks_q | toks_d)
            score = inter / union if union else 0.0
            if score > meilleur_score:
                meilleur, meilleur_score = i, score
        LOG.info("[rag] %d candidats, meilleur %.2f (%.1f ms)",
                 len(candidats), meilleur_score,
                 (time.perf_counter() - debut) * 1000)
        if meilleur is None or meilleur_score < seuil:
            return None
        doc = self._docs[meilleur]
        return doc.get("a", "")


INDEX = IndexRAG()


def chercher(question: str) -> str | None:
    """API module : reponse extraite si confiance suffisante, sinon None."""
    try:
        return INDEX.chercher(question)
    except Exception as e:
        LOG.warning("[rag] erreur : %s", e)
        return None


def statistiques() -> dict:
    """Pour le diagnostic : taille et remplissage de l'index."""
    try:
        INDEX._assurer()
    except Exception:
        pass
    return {"charge": INDEX._charge, "documents": len(INDEX._docs)}
