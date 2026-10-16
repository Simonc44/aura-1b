/**
 * Harnais commun aux scripts de fumée du TUI (`smoke.ts`, `smoke_attente.ts`,
 * `smoke_design.ts`).
 *
 * Ces trois entrées font exactement la même chose en tête et en queue :
 * compter les contrôles, imprimer une ligne `OK`/`ECHEC` lisible, puis sortir
 * en code 1 dès qu'un contrôle a échoué — c'est ce qui les rend utilisables
 * tels quels dans une CI. Ce harnais était RECOPIÉ dans les trois ; il vit
 * ici une fois. Les messages finaux restent paramétrables pour que la sortie
 * de chaque script ne change pas (elle est citée dans `tui/README.md`).
 */

let nbEchecs = 0;

/** Un contrôle : `nom`, son verdict, et un détail libre affiché après `—`. */
export function controle(nom: string, ok: boolean, detail = ""): void {
  if (!ok) {
    nbEchecs += 1;
  }
  console.log(`${ok ? "OK  " : "ECHEC"} ${nom}${detail ? ` — ${detail}` : ""}`);
}

/**
 * Conclut le script : code de sortie 0 si tous les contrôles passent, 1 sinon.
 * `succes` est le message imprimé dans le cas vert ; `echec` est le mot qui
 * suit le nombre dans le cas rouge (`3 controle(s) en echec`).
 */
export function terminer(succes: string, echec = "controle(s) en echec"): never {
  console.log(`\n${nbEchecs === 0 ? succes : `${nbEchecs} ${echec}`}`);
  process.exit(nbEchecs === 0 ? 0 : 1);
}
