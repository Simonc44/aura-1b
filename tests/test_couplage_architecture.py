"""Le couplage LoRA <-> architecture du cerveau est DECLARE et MESURE.

Cet item n'est pas un bug : c'est un CHOIX d'architecture. Les 4 adaptateurs
sont des deltas de poids calcules pour Llama-3.2-1B ; ils interdisent de
changer de cerveau sans les re-entrainer.

Deux choses etaient corrigeables, et sont corrigees ici :

1. Le couplage etait INVISIBLE — llvm.cpp ne leve aucune erreur quand on
   monte un adaptateur sur une autre architecture : il produit du bruit.
2. Le CONTROLE etait faux : il comparait des chaines de caracteres dans le
   nom du fichier (« llama-3.2-1b » in nom). Un fichier renomme passait donc
   le controle, et un cerveau du bon type range autrement etait refuse.
   Les nombres qui comptent sont DANS le GGUF : le verdict est desormais une
   MESURE (`aura/gguf_meta.py`).

Constat de la bascule du 9 oct. 2026 :
    Llama-3.2-1B : 16 couches, dim 2048, 32/8 tetes
    Gemma-3-1B   : 26 couches, dim 1152,  4/1 tetes
Les 4 adaptateurs du disque sont donc INCOMPATIBLES avec le cerveau par
defaut, et c'est ce que ces tests verrouillent — au lieu de le decouvrir
sous forme de sorties degradees sans message d'erreur.
"""
import json
import struct
import urllib.request
from pathlib import Path

import pytest

from aura import adaptateurs, gguf_meta

RACINE = Path(__file__).resolve().parent.parent
LLAMA = RACINE / "modeles" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
GEMMA = (RACINE / "modeles"
         / "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_Q4_k_m.gguf")


class TestDeclaration:
    def test_architecture_cible_est_le_cerveau_actuel(self):
        """Les adaptateurs re-entraines visent Gemma, comme le cerveau.

        Avant la bascule, ARCHITECTURE_CIBLE decrivait les adaptateurs LLAMA
        du disque. Depuis le re-entrainement, la cible EST le cerveau : les
        deux declarations doivent donc concorder (sinon la garde refuserait
        les adaptateurs qu'on vient de produire).
        """
        assert adaptateurs.ARCHITECTURE_CIBLE["famille"] == "gemma3"
        assert adaptateurs.ARCHITECTURE_CIBLE["parametres"] == "1B"
        assert adaptateurs.ARCHITECTURE_CIBLE["block_count"] == 26
        assert adaptateurs.ARCHITECTURE_CIBLE["embedding_length"] == 1152

    def test_architecture_du_cerveau_actuel_declaree(self):
        assert adaptateurs.ARCHITECTURE_CERVEAU["famille"] == "gemma3"
        assert adaptateurs.ARCHITECTURE_CERVEAU["block_count"] == 26
        assert adaptateurs.ARCHITECTURE_CERVEAU["embedding_length"] == 1152


class TestVerdictMesure:
    def test_cerveau_configure_compatible_avec_les_adaptateurs(self):
        """Le cerveau PAR DEFAUT (Gemma) accepte les adaptateurs du disque.

        Avant le re-entrainement, les .gguf du disque etaient des Llama et ce
        test portait sur Llama-3.2 : il verifiait l'ANCIEN etat. Depuis le
        10 oct. 2026 les 4 adaptateurs sont Gemma, donc le cerveau a verifier
        est celui que le projet charge reellement (llama_cerveau.chemin_gguf).
        """
        from aura import llama_cerveau
        cerveau = llama_cerveau.chemin_gguf()
        if not Path(cerveau).is_file():
            pytest.skip("GGUF du cerveau configure absent : rien a mesurer")
        if not adaptateurs.adaptateurs_installes():
            pytest.skip("aucun adaptateur sur le disque")
        ok, pourquoi = adaptateurs.verifier_cerveau(str(cerveau))
        assert ok, pourquoi

    def test_bascule_effective_llama_refuse_desormais(self):
        """Le cerveau Llama, reste sur le disque, doit REFUSER les adaptateurs

        Gemma : c'est la preuve que la bascule a bien eu lieu, la ou l'ancien
        test se contentait de constater l'etat precedent.
        """
        if not LLAMA.is_file():
            pytest.skip("GGUF Llama absent : rien a mesurer")
        if not adaptateurs.adaptateurs_installes():
            pytest.skip("aucun adaptateur sur le disque")
        if "gemma3" not in adaptateurs.architecture_adaptateurs().get(
                "familles", []):
            pytest.skip("les adaptateurs du disque ne sont pas des Gemma")
        ok, pourquoi = adaptateurs.verifier_cerveau(str(LLAMA))
        assert not ok
        assert "famille" in pourquoi and "gemma3" in pourquoi

    def test_fichier_renomme_reste_reconnu(self):
        """LE defaut corrige : le nom ne prouve rien, le contenu si.

        Une copie de Llama portant un nom quelconque doit etre ACCEPTEE
        (c'est la meme architecture) — l'ancien controle la refusait.
        """
        if not LLAMA.is_file():
            pytest.skip("GGUF Llama absent")
        arc = gguf_meta.architecture(str(LLAMA))
        assert arc["block_count"] == 16          # lu, pas deduit du nom
        # le verdict ne depend d'aucune sous-chaine du nom de fichier
        assert not adaptateurs.ARCHITECTURE_CIBLE.get("motif_nom")

    def test_gemma_refuse_les_adaptateurs_llama_du_disque(self):
        """Tant que les .gguf du disque sont des deltas Llama, Gemma les refuse.

        Le verdict se lit sur les FICHIERS : apres le re-entrainement Gemma,
        ce cas disparait tout seul (cf. TestMontageAuto). On saute donc quand
        les adaptateurs ne sont plus des Llama plutot que d'exiger un etat
        perime.
        """
        if not GEMMA.is_file():
            pytest.skip("GGUF Gemma absent : rien a mesurer")
        arc = adaptateurs.architecture_adaptateurs()
        if "llama" not in arc.get("familles", []):
            pytest.skip("les adaptateurs du disque ne sont plus des Llama")
        ok, pourquoi = adaptateurs.verifier_cerveau(str(GEMMA))
        assert not ok
        # le message doit DIRE pourquoi : les deux familles et le remede
        assert "llama" in pourquoi and "gemma3" in pourquoi
        assert "re-entrainement" in pourquoi

    def test_fichier_absent_ne_leve_jamais(self):
        """Un garde-fou qui plante serait pire que le silence qu'il remplace."""
        for entree in ("", "?", "/chemin/inexistant.gguf", "a/b/c.gguf"):
            ok, pourquoi = adaptateurs.verifier_cerveau(entree)
            assert ok is False and isinstance(pourquoi, str)


def _ecrire_gguf_minimal(chemin: Path, kv: dict) -> None:
    """Un en-tete GGUF valide a 0 tenseur : assez pour lire l'architecture.

    C'est ce qu'un LoRA GGUF expose reellement (mesure sur les fichiers du
    projet) : `general.architecture` + `general.type`/`adapter.type`, mais
    PAS `block_count`/`embedding_length`. On reproduit donc cet en-tete.
    """
    with open(chemin, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))        # version
        f.write(struct.pack("<Q", 0))        # n_tensors
        f.write(struct.pack("<Q", len(kv)))  # n_kv
        for k, v in kv.items():
            kb = k.encode()
            f.write(struct.pack("<Q", len(kb)))
            f.write(kb)
            if isinstance(v, str):
                vb = v.encode()
                f.write(struct.pack("<I", 8))
                f.write(struct.pack("<Q", len(vb)))
                f.write(vb)
            elif isinstance(v, float):
                f.write(struct.pack("<I", 6))
                f.write(struct.pack("<f", v))
            else:
                f.write(struct.pack("<I", 4))
                f.write(struct.pack("<I", int(v)))


def _poser_adaptateurs(monkeypatch, tmp_path: Path, famille: str) -> Path:
    """Ecrit 4 faux adaptateurs de la famille voulue et pointe `dossier()`."""
    d = tmp_path / "adaptateurs"
    d.mkdir(parents=True, exist_ok=True)
    for nom in adaptateurs._ADAPTATEURS.values():
        _ecrire_gguf_minimal(d / f"{nom}.gguf", {
            "general.architecture": famille,
            "general.type": "adapter",
            "adapter.type": "lora",
        })
    monkeypatch.setattr(adaptateurs, "dossier", lambda: d)
    return d


class TestMontageAuto:
    """Le montage suit l'ARCHITECTURE REELLE des fichiers, pas une constante.

    Amelioration demandee : remplacer les 4 .gguf par les versions Gemma doit
    activer le montage sans toucher une ligne de code (et les Llama doivent
    rester refuses). On le verifie avec des en-tetes GGUF synthetiques, donc
    sans dependre des fichiers du disque.
    """

    def test_adaptateurs_gemma_acceptes_sur_cerveau_gemma(self, tmp_path, monkeypatch):
        if not GEMMA.is_file():
            pytest.skip("GGUF Gemma absent : rien a mesurer")
        _poser_adaptateurs(monkeypatch, tmp_path, "gemma3")
        ok, pourquoi = adaptateurs.verifier_cerveau(str(GEMMA))
        assert ok, pourquoi

    def test_adaptateurs_llama_refuses_sur_cerveau_gemma(self, tmp_path, monkeypatch):
        if not GEMMA.is_file():
            pytest.skip("GGUF Gemma absent")
        _poser_adaptateurs(monkeypatch, tmp_path, "llama")
        ok, pourquoi = adaptateurs.verifier_cerveau(str(GEMMA))
        assert not ok
        assert "famille" in pourquoi and "llama" in pourquoi

    def test_adaptateurs_gemma_refuses_sur_cerveau_llama(self, tmp_path, monkeypatch):
        if not LLAMA.is_file():
            pytest.skip("GGUF Llama absent")
        _poser_adaptateurs(monkeypatch, tmp_path, "gemma3")
        ok, _ = adaptateurs.verifier_cerveau(str(LLAMA))
        assert not ok

    def test_aucun_adaptateur_aucune_garde(self, tmp_path, monkeypatch):
        """Sans adaptateur sur disque, rien a proteger : on accepte."""
        d = tmp_path / "adaptateurs"
        d.mkdir()
        monkeypatch.setattr(adaptateurs, "dossier", lambda: d)
        ok, _ = adaptateurs.verifier_cerveau(str(GEMMA))
        assert ok is True

    def test_en_tete_illisible_ne_bloque_pas_le_montage(self, tmp_path, monkeypatch):
        """Un .bin non-GGUF ne doit pas faire refuser les autres."""
        if not GEMMA.is_file():
            pytest.skip("GGUF Gemma absent")
        d = tmp_path / "adaptateurs"
        d.mkdir(parents=True)
        (d / "aura-code.gguf").write_bytes(b"pas un gguf")
        _ecrire_gguf_minimal(d / "aura-math.gguf", {
            "general.architecture": "gemma3", "general.type": "adapter"})
        monkeypatch.setattr(adaptateurs, "dossier", lambda: d)
        ok, pourquoi = adaptateurs.verifier_cerveau(str(GEMMA))
        assert ok, pourquoi

    def test_adaptateurs_installes_liste_les_quatre(self, tmp_path, monkeypatch):
        _poser_adaptateurs(monkeypatch, tmp_path, "gemma3")
        trouves = adaptateurs.adaptateurs_installes()
        assert len(trouves) == 4
        assert all(p.suffix == ".gguf" for p in trouves)


class TestBootServeur:
    def test_adaptateurs_refuses_si_cerveau_incompatible(self):
        """Le garde-fou est BRANCHE dans le boot, pas seulement defini."""
        from aura import serveur_lora as sl
        source = Path(sl.__file__).read_text(encoding="utf-8")
        assert "verifier_cerveau" in source, (
            "le garde-fou d'architecture doit etre appele au boot du serveur "
            "— sinon les adaptateurs seraient montes sur un cerveau "
            "incompatible, en silence")

    def test_serveur_utilise_le_meme_cerveau_que_le_reste(self):
        """Une seule source pour le chemin du GGUF.

        serveur_lora avait sa propre liste de noms de fichiers : le jour ou
        l'on change de cerveau, un oubli ne produit pas d'erreur mais un
        serveur qui tourne sur l'ANCIEN modele.
        """
        from aura import llama_cerveau, serveur_lora
        assert serveur_lora._gguf_cerveau() == Path(llama_cerveau.chemin_gguf())


class TestReutilisationServeur:
    """Un serveur DEJA en place n'est repris que s'il sert le bon cerveau.

    Defaut reel, constate le 9 oct. 2026 : `_existe_serveur()` ne lisait que
    `/health`. Un llama-server reste d'un lancement precedent (Llama-3.2 + ses
    4 adaptateurs) repond « ok » exactement comme le bon — RontoGPT l'adoptaiat
    donc en silence et repondait avec l'ANCIEN cerveau pendant que tout le
    reste annoncait le nouveau. Meme remede que pour les adaptateurs : le
    verdict est une MESURE d'architecture, pas une comparaison de noms.
    """

    def test_meme_cerveau_accepte_le_cerveau_configure(self):
        from aura import llama_cerveau, serveur_lora
        cerveau = llama_cerveau.chemin_gguf()
        if not Path(cerveau).is_file():
            pytest.skip("GGUF du cerveau configure absent : rien a mesurer")
        assert serveur_lora._meme_cerveau(cerveau) is True

    def test_autre_cerveau_refuse(self):
        """Le cerveau par defaut est Gemma : Llama-3.2 doit etre refuse.

        On teste l'ecart sur les deux fichiers REELS du disque quand ils sont
        la — c'est exactement le cas qui s'est produit.
        """
        from aura import llama_cerveau, serveur_lora
        cerveau = llama_cerveau.chemin_gguf()
        if not (LLAMA.is_file() and GEMMA.is_file() and Path(cerveau).is_file()):
            pytest.skip("les deux GGUF ne sont pas sur le disque")
        if Path(cerveau).name != GEMMA.name:
            pytest.skip("le cerveau configure n'est pas Gemma")
        assert serveur_lora._meme_cerveau(str(LLAMA)) is False

    def test_sans_information_on_ne_refuse_pas(self):
        """On ne refuse jamais ce qu'on n'a pas su mesurer : None, un chemin
        inexistant ou un en-tete illisible ne doivent pas couper le serveur."""
        from aura import serveur_lora
        assert serveur_lora._meme_cerveau(None) is True
        assert serveur_lora._meme_cerveau("") is True
        assert serveur_lora._meme_cerveau("/chemin/inexistant.gguf") is True

    def _urlopen_factice(self, charge, monkeypatch):
        class _Reponse:
            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def read(self):
                return json.dumps(charge).encode()

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Reponse())

    def test_modele_servi_lit_les_deux_formes(self, monkeypatch):
        """llama-server nomme la liste `models` ou `data` selon la version."""
        from aura import serveur_lora
        self._urlopen_factice({"models": [{"model": "a.gguf"}]}, monkeypatch)
        assert serveur_lora._modele_servi() == "a.gguf"
        self._urlopen_factice({"data": [{"id": "b.gguf"}]}, monkeypatch)
        assert serveur_lora._modele_servi() == "b.gguf"

    def test_modele_servi_illisible_donne_none(self, monkeypatch):
        from aura import serveur_lora
        for charge in ({}, {"models": []}, {"models": "pas une liste"},
                       {"models": ["pas un dict"]}, {"models": [{}]}):
            self._urlopen_factice(charge, monkeypatch)
            assert serveur_lora._modele_servi() is None, charge

        def _explose(*_a, **_k):
            raise OSError("port ferme")

        monkeypatch.setattr(urllib.request, "urlopen", _explose)
        assert serveur_lora._modele_servi() is None

    def test_conflit_detecte_et_message_actionnable(self, monkeypatch):
        """Le verdict doit NOMMER le cerveau configure — c'est ce qui rend
        l'ecart visible pour quelqu'un qui lit les journaux."""
        from aura import llama_cerveau, serveur_lora
        monkeypatch.setattr(serveur_lora, "_existe_serveur", lambda *a, **k: True)
        monkeypatch.setattr(serveur_lora, "_modele_servi", lambda *a, **k: str(LLAMA))
        monkeypatch.setattr(serveur_lora, "_meme_cerveau", lambda *a, **k: False)
        message = serveur_lora._conflit_cerveau()
        assert message is not None
        assert llama_cerveau.cle_cerveau() in message
        assert "in-process" in message

    def test_pas_de_conflit_quand_le_cerveau_correspond(self, monkeypatch):
        from aura import serveur_lora
        monkeypatch.setattr(serveur_lora, "_existe_serveur", lambda *a, **k: True)
        monkeypatch.setattr(serveur_lora, "_meme_cerveau", lambda *a, **k: True)
        assert serveur_lora._conflit_cerveau() is None

    def test_pas_de_serveur_pas_de_conflit(self, monkeypatch):
        from aura import serveur_lora
        monkeypatch.setattr(serveur_lora, "_existe_serveur", lambda *a, **k: False)
        assert serveur_lora._conflit_cerveau() is None

    def test_assurer_refuse_un_serveur_etranger(self, monkeypatch):
        """BRANCHE, pas seulement defini : `_assurer()` doit dire non, ce qui
        fait retomber `chat()` sur le chemin in-process (le bon cerveau)."""
        from aura import serveur_lora
        monkeypatch.setenv("AURA_SERVEUR", "1")
        monkeypatch.setattr(serveur_lora, "_existe_serveur", lambda *a, **k: True)
        monkeypatch.setattr(serveur_lora, "_conflit_cerveau", lambda: "autre cerveau")
        monkeypatch.setattr(serveur_lora, "_conflit_signale", False)
        assert serveur_lora._assurer() is False

    def test_assurer_reprend_quand_le_cerveau_correspond(self, monkeypatch):
        from aura import serveur_lora
        monkeypatch.setenv("AURA_SERVEUR", "1")
        monkeypatch.setattr(serveur_lora, "_existe_serveur", lambda *a, **k: True)
        monkeypatch.setattr(serveur_lora, "_conflit_cerveau", lambda: None)
        assert serveur_lora._assurer() is True
