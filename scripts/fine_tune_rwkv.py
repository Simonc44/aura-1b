"""Fine-tuning de RWKV-4 430M sur des données françaises d'instruction.

Entrée : un fichier JSONL avec des paires {"user": "...", "assistant": "..."}
Sortie : un modèle RWKV fine-tuné qui parle français comme Qwen instruct.

Utilise le package `rwkv` (pur PyTorch, pas de CUDA requis).

Usage :
    python scripts/fine_tune_rwkv.py --data donnees_francais.jsonl --epochs 3
"""
import argparse
import json
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("fine_tune")


def charger_donnees(chemin: str, max_exemples: int = 1000) -> list[dict]:
    """Charge les paires user/assistant depuis un JSONL."""
    donnees = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                entree = json.loads(ligne)
                if "user" in entree and "assistant" in entree:
                    donnees.append(entree)
                    if len(donnees) >= max_exemples:
                        break
            except json.JSONDecodeError:
                continue
    LOG.info("charge %d exemples depuis %s", len(donnees), chemin)
    return donnees


def formater_prompt(exemple: dict) -> str:
    """Formate un exemple en prompt d'instruction."""
    return (
        f"### Human: {exemple['user']}\n"
        f"### Assistant: {exemple['assistant']}\n\n"
    )


def fine_tuner(chemin_modele: str, donnees: list[dict],
               epochs: int = 3, lr: float = 1e-4, batch_size: int = 4):
    """Fine-tune le modèle RWKV sur les données d'instruction."""
    import torch
    from torch.optim import AdamW
    from rwkv.model import RWKV

    LOG.info("chargement du modèle %s ...", chemin_modele)
    model = RWKV(model=chemin_modele, strategy="cpu fp32")

    # prepérer les données
    prompts = [formater_prompt(d) for d in donnees]

    # paramètres d'entraînement
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    LOG.info("debut du fine-tuning : %d epochs, %d exemples", epochs, len(donnees))
    t0 = time.time()

    for epoch in range(epochs):
        loss_total = 0.0
        n_batch = 0
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            for prompt in batch:
                # tokeniser
                tokens = model.tokenizer.encode(prompt) if hasattr(model, 'tokenizer') else [0]
                if len(tokens) < 2:
                    continue

                # forward pass (simplifié : prediction du token suivant)
                x = torch.tensor([tokens[:-1]]).long()
                y = torch.tensor([tokens[1:]]).long()
                out, _ = model.forward(x, None)
                loss = torch.nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
                loss.backward()
                loss_total += loss.item()
                n_batch += 1

            optimizer.step()
            optimizer.zero_grad()

            if n_batch % 10 == 0:
                LOG.info("  epoch %d/%d, batch %d, loss=%.4f",
                         epoch + 1, epochs, n_batch, loss_total / max(n_batch, 1))

        LOG.info("epoch %d terminee, loss moyenne = %.4f",
                 epoch + 1, loss_total / max(n_batch, 1))

    LOG.info("fine-tuning termine en %.0fs", time.time() - t0)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="fichier JSONL user/assistant")
    p.add_argument("--modele", default=None, help="chemin vers le .pth RWKV")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max", type=int, default=1000)
    p.add_argument("--output", default="RWKV-4-430M-finetuned.pth")
    args = p.parse_args()

    # chercher le modèle par défaut
    if args.modele is None:
        from huggingface_hub import hf_hub_download
        args.modele = hf_hub_download(
            "BlinkDL/rwkv-4-pile-430m",
            "RWKV-4-Pile-430M-20220808-8066.pth")
        LOG.info("modele telecharge : %s", args.modele)

    donnees = charger_donnees(args.data, args.max)
    if not donnees:
        LOG.error("aucune donnee valide dans %s", args.data)
        sys.exit(1)

    model = fine_tuner(args.modele, donnees, args.epochs, args.lr)

    # sauvegarder
    torch.save(model.state_dict(), args.output)
    LOG.info("modelee fine-tune sauvegarde : %s", args.output)
    LOG.info("pour utiliser : mettez-le dans le dossier du modele et chargez-le")


if __name__ == "__main__":
    main()
