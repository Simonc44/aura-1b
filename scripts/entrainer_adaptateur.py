"""Entrainement d'un adaptateur LoRA pour Aura (Colab T4 gratuit).

Produit un adapter GGUF (~10-40 Mo) a deposer dans adaptateurs/ :
  adaptateurs/aura-web.gguf      specialise faits/actualite
  adaptateurs/aura-math.gguf     specialise raisonnement mathematique
  adaptateurs/aura-code.gguf     specialise code
  adaptateurs/aura-general.gguf  specialise redaction FR

Usage Colab :
  !pip -q install peft transformers accelerate bitsandbytes
  !python entrainer_adaptateur.py --categorie web --dataset web.jsonl
  (telecharge ensuite adaptateurs/aura-web.gguf sur ton PC)

Format dataset (JSONL, >= 200 exemples recommandes) :
  {"question": "...", "reponse": "..."}
  {"question": "...", "reponse": "...", "contexte": "faits utiles"}  (option)

Le template de chat est EXACTEMENT celui d'Aura (llama_cerveau) : l'adap-
tateur apprend dans les memes conditions que l'inference locale.
"""
import argparse
import json
from pathlib import Path

NOMS = {"web": "aura-web", "math": "aura-math",
        "code": "aura-code", "general": "aura-general"}

# template identique a aura/llama_cerveau.py (ChatML Llama 3)
TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{systeme}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{contenu}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{reponse}"
)


def charger_dataset(chemin: str, categorie: str) -> list[dict]:
    """Lit le JSONL et applique la personnalite de la categorie."""
    from aura_personnalites import PROMPT_PAR_CATEGORIE
    systeme = PROMPT_PAR_CATEGORIE[categorie]
    exemples = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            e = json.loads(ligne)
            contenu = e["question"]
            if e.get("contexte"):
                contenu = f"FAITS WEB :\n{e['contexte']}\n\n{contenu}"
            exemples.append({"texte": TEMPLATE.format(
                systeme=systeme, contenu=contenu,
                reponse=e["reponse"] + "<|eot_id|>")})
    return exemples


def entrainer(categorie: str, dataset: str, epochs: int = 3,
              sortie: str = "adaptateurs") -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)

    # miroir public (unsloth) : memes poids que meta-llama, sans token HF
    base = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")

    # LoRA : ~10-40 Mo une fois exporte, cible les projections d'attention
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()          # ~0.5-1 % des parametres

    exemples = charger_dataset(dataset, categorie)
    assert len(exemples) >= 50, "dataset trop petit (>= 50 exemples)"
    ds = (tok([e["texte"] for e in exemples], truncation=True,
              max_length=1024, padding=True, return_tensors="pt")
          .data)

    args = TrainingArguments(
        output_dir=f"./tmp-lora-{categorie}",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True, logging_steps=10,
        save_strategy="no", report_to=[])

    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
            ).train()

    # export : adapter bfloat16 -> GGUF (llama.cpp attends f32/f16)
    dossier = Path(sortie)
    dossier.mkdir(exist_ok=True)
    chemin = dossier / f"{NOMS[categorie]}.gguf"
    model.save_pretrained(f"./tmp-lora-{categorie}/final")
    print(f"[OK] adapter PEFT : ./tmp-lora-{categorie}/final")
    print(f"[SUITE] conversion GGUF (le merge est deja applique dans les "
          f"poids PEFT) :\n"
          f"  python convert_lora_to_gguf.py ./tmp-lora-{categorie}/final "
          f"--outfile {chemin} --base meta-llama/Llama-3.2-1B-Instruct")
    print(f"[FIN] deposer {chemin.name} dans adaptateurs/ sur ton PC, "
          f"puis lancer Aura avec AURA_ADAPTATEURS=1")
    return chemin


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--categorie", required=True, choices=list(NOMS))
    p.add_argument("--dataset", required=True, help="JSONL question/reponse")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    entrainer(args.categorie, args.dataset, args.epochs)
