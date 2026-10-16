/**
 * Vérifie à l'écran la ligne d'attente d'un tour : le compteur de morceaux
 * reçus apparaît dès le premier morceau, et seulement alors. Sort en code 1
 * au premier échec.
 *
 * Usage : tui\node_modules\bin\bun.cmd run src/smoke_attente.ts
 */
import { TextRenderable } from "@opentui/core";
import { createTestRenderer } from "@opentui/core/testing";
import { libelleAttente } from "./affichage";

let echecs = 0;
function controle(nom: string, ok: boolean, detail: string): void {
  if (!ok) echecs += 1;
  console.log(`${ok ? "OK  " : "ECHEC"} ${nom}${detail ? ` — ${detail}` : ""}`);
}

const ecran = await createTestRenderer({ width: 90, height: 6 });
const ligne = new TextRenderable(ecran.renderer, { content: "" });
ecran.renderer.root.add(ligne);

async function peindre(texte: string): Promise<string> {
  ligne.content = texte;
  await ecran.flush();
  return ecran.captureCharFrame();
}

// 1. avant tout morceau : pas de compteur
let frame = await peindre(libelleAttente("Réflexion", 12.7, 0));
controle("sans morceau, pas de compteur à l'écran",
  frame.includes("Réflexion (12 s • Échap pour interrompre)")
  && !frame.includes("morceaux"), frame.split("\n")[0]?.trim() ?? "");

// 2. dès le premier morceau : le compte réel est affiché
frame = await peindre(libelleAttente("Recherche sur le web…", 3, 1));
controle("au premier morceau, le compte réel est affiché",
  frame.includes("Recherche sur le web… (3 s • 1 morceau • Échap pour interrompre)"),
  frame.split("\n")[0]?.trim() ?? "");

// 3. le compte progresse avec le temps
frame = await peindre(libelleAttente("Réflexion", 9, 42));
controle("le compte progresse (42 morceaux à 9 s)",
  frame.includes("Réflexion (9 s • 42 morceaux • Échap pour interrompre)"),
  frame.split("\n")[0]?.trim() ?? "");

ecran.renderer.destroy();
if (echecs > 0) {
  console.log(`\n${echecs} échec(s)`);
  process.exit(1);
}
console.log("\ntous les contrôles d'attente passent");
