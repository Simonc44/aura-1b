"""Harnais d'intervention TransformerLens 4 sur gemma-3-1b-it (cache local).

TransformerLens 4 a retire HookedTransformer : on passe par
TransformerBridge.boot_transformers + enable_compatibility_mode.

1. Sanity : un hook identite ne change pas les logits.
2. Ablation moyenne (remplace le courant residuel par sa moyenne sur les
   positions) et attenuation graduee (x0.90) : mesure couche par couche.
   La mise a zero totale effondre le modele et n'est pas exploitable.
3. Test de la langue : on injecte alpha * (E[' the'] - E[' le']) dans le
   courant residuel de la couche 1 et on genere de facon gloutonne. Un
   changement de langue controle (sans <pad>) prouve la direction.

Usage : PYTHONPATH=. python scripts/intervention_tl.py
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import TransformerBridge

MODELE_HF = "unsloth/gemma-3-1b-it"
DOSSIER_LOCAL = os.path.expanduser("~/.cache/aura_models/gemma-3-1b-it")
PROMPT_ANGLAIS = "Q: What is the capital of Australia?\nA:"
PROMPT_FR = "Q: Quelle est la capitale de l'Australie ?\nR:"


def charger():
    if not os.path.exists(os.path.join(DOSSIER_LOCAL, "model.safetensors")):
        raise FileNotFoundError(f"poids absents : {DOSSIER_LOCAL}")
    tok = AutoTokenizer.from_pretrained(DOSSIER_LOCAL)
    hf = AutoModelForCausalLM.from_pretrained(DOSSIER_LOCAL, dtype=torch.float32)
    modele = TransformerBridge.boot_transformers(
        MODELE_HF, hf_model=hf, tokenizer=tok, device="cpu", dtype=torch.float32)
    modele.enable_compatibility_mode()
    modele.eval()
    return modele, tok


def logits_dernier(modele, tok, prompt: str, hooks=()):
    entree = tok(prompt, return_tensors="pt")
    with torch.no_grad(), modele.hooks(fwd_hooks=list(hooks)):
        sortie = modele(entree["input_ids"])
    return sortie[0, -1]


def generer(modele, tok, prompt: str, n: int = 20, hooks=()) -> str:
    """Generation gloutonne ; les hooks restent actifs a chaque pas."""
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad(), modele.hooks(fwd_hooks=list(hooks)):
        for _ in range(n):
            suivant = modele(ids)[0, -1].argmax()
            ids = torch.cat([ids, suivant.view(1, 1)], dim=1)
    return tok.decode(ids[0, -n:], skip_special_tokens=True)


def ablation_moyenne(x, hook):
    return x.mean(dim=1, keepdim=True).expand_as(x)


def attenuation(x, hook):
    return x * 0.90


def sanity_et_ablations(modele, tok) -> None:
    base = logits_dernier(modele, tok, PROMPT_FR)
    print(f"reference : {tok.decode([int(base.argmax())])!r}")

    def identite(x, hook):
        return x

    ident = logits_dernier(modele, tok, PROMPT_FR,
                           [("blocks.10.hook_resid_post", identite)])
    diff = float((base - ident).abs().max())
    print(f"sanity identite : ecart max = {diff:.2e}")
    assert diff < 1e-4, "un hook identite modifie les logits"

    for nom, fonction in (("moyenne", ablation_moyenne),
                          ("attenuation x0.90", attenuation)):
        for couche in range(0, modele.cfg.n_layers, 4):
            z = logits_dernier(modele, tok, PROMPT_FR,
                               [(f"blocks.{couche}.hook_resid_post", fonction)])
            top = tok.decode([int(z.argmax())])
            ecart = float((base - z).abs().max())
            print(f"{nom:18s} couche {couche:2d} : top={top!r} ecart={ecart:.2f}")


def test_langue(modele, tok) -> None:
    """Direction ' the' - ' le' injectee en couche 1 (tous les emplacements)."""
    emb = modele.W_E
    id_the = tok.encode(" the", add_special_tokens=False)[0]
    id_le = tok.encode(" le", add_special_tokens=False)[0]
    delta = emb[id_the] - emb[id_le]
    print(f"norme de delta : {float(delta.norm()):.3f}")

    print("sans intervention :", repr(generer(modele, tok, PROMPT_ANGLAIS)))
    for alpha in (0.5, 1.0, 2.0, 4.0):
        def injecter(x, hook, a=alpha):
            return x + a * delta.to(x.dtype)
        sortie = generer(modele, tok, PROMPT_ANGLAIS,
                         hooks=[("blocks.1.hook_resid_post", injecter)])
        print(f"alpha={alpha}: {sortie!r}")


PAIRES_CONCEPT = [
    ("Q: What is the capital of Australia?\nA:", "Q: Quelle est la capitale de l'Australie ?\nR:"),
    ("Q: What is the largest planet?\nA:", "Q: Quelle est la plus grande planete ?\nR:"),
    ("Q: Who wrote Hamlet?\nA:", "Q: Qui a ecrit Hamlet ?\nR:"),
    ("Q: What is the boiling point of water?\nA:", "Q: Quel est le point d'ebullition de l'eau ?\nR:"),
    ("Q: How many legs does a spider have?\nA:", "Q: Combien de pattes a une araignee ?\nR:"),
]
PROMPTS_EVAL = [
    "Q: What is the capital of Germany?\nA:",
    "Q: Who painted the Mona Lisa?\nA:",
]
MOTS_FR = {"le", "la", "les", "de", "des", "est", "une", "un", "et", "qui", "que", "du", "pas"}
MOTS_EN = {"the", "is", "of", "and", "a", "an", "what", "who", "how", "many", "that", "it", "not"}


def residu_dernier(modele, tok, prompt: str, couche: int):
    """Courant residuel du dernier token, couche donnee."""
    capture = {}

    def garder(x, hook):
        capture["h"] = x[0, -1].detach().clone()
        return x

    logits_dernier(modele, tok, prompt,
                   [(f"blocks.{couche}.hook_resid_post", garder)])
    return capture["h"]


def vecteur_concept(modele, tok, couche: int) -> torch.Tensor:
    """Moyenne des deltas (FR - EN) sur les paires : le bruit des mots s'annule."""
    deltas = []
    for en, fr in PAIRES_CONCEPT:
        deltas.append(residu_dernier(modele, tok, fr, couche)
                      - residu_dernier(modele, tok, en, couche))
    return torch.stack(deltas).mean(dim=0)


def score_langue(texte: str) -> float:
    """Part de mots fonctionnels francais parmi les mots fonctionnels reconnus."""
    mots = [m.strip(".,:?!\"'").lower() for m in texte.split()]
    fr = sum(m in MOTS_FR for m in mots)
    en = sum(m in MOTS_EN for m in mots)
    return fr / (fr + en) if fr + en else 0.5


def test_concept(modele, tok) -> None:
    """Balayage des couches 8 a 16 avec le vecteur de concept moyen."""
    for prompt in PROMPTS_EVAL:
        print("sans intervention :", repr(generer(modele, tok, prompt)))
    for couche in range(8, 17, 2):
        vec = vecteur_concept(modele, tok, couche)
        print(f"couche {couche} norme={float(vec.norm()):.2f}")
        for alpha in (1.0, 2.0, 4.0):
            def injecter(x, hook, a=alpha, v=vec):
                return x + a * v.to(x.dtype)
            for prompt in PROMPTS_EVAL:
                sortie = generer(modele, tok, prompt,
                                 hooks=[(f"blocks.{couche}.hook_resid_post", injecter)])
                print(f"  alpha={alpha} fr={score_langue(sortie):.2f} {sortie!r}")


def test_controle(modele, tok) -> None:
    """Controle : vecteur aleatoire de meme norme que le concept (seed fixe)."""
    for couche in (12, 16):
        vec = vecteur_concept(modele, tok, couche)
        torch.manual_seed(0)
        alea = torch.randn_like(vec)
        alea = alea * (vec.norm() / alea.norm())
        for nom, v in (("concept", vec), ("aleatoire", alea)):
            for alpha in (1.0, 2.0):
                for prompt in PROMPTS_EVAL:
                    def injecter(x, hook, a=alpha, w=v):
                        return x + a * w.to(x.dtype)
                    sortie = generer(modele, tok, prompt,
                                     hooks=[(f"blocks.{couche}.hook_resid_post", injecter)])
                    print(f"couche {couche} {nom:9s} alpha={alpha} fr={score_langue(sortie):.2f} {sortie!r}")


def main() -> int:
    modele, tok = charger()
    test_controle(modele, tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
