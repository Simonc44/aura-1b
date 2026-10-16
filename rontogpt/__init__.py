"""RontoGPT : nom public du paquet.

Le code vit dans `aura/` (nom historique) ; ce paquet existe pour que la
commande tapee corresponde au nom du produit :

    python -m rontogpt                 # interface terminal (defaut)
    python -m rontogpt "question"
    python -m rontogpt --console

`import rontogpt` reste volontairement leger : rien n'est charge avant
d'en avoir besoin (l'import de `aura` tire l'orchestrateur, gplearn et
llama_cpp).
"""

__all__ = ["__version__"]


def __getattr__(nom: str) -> str:
    """`rontogpt.__version__` sans charger tout `aura` (PEP 562)."""
    if nom == "__version__":
        from aura import __version__
        return __version__
    raise AttributeError(f"module 'rontogpt' has no attribute {nom!r}")
