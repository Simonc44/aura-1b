/**
 * RontoGPT — interface terminal (OpenTUI), habillage « Codex ».
 *
 * Ce fichier ne fait QUE l'affichage et la saisie : le cerveau vit dans le
 * processus Python (`aura/api.py`) et on lui parle en HTTP/SSE sur
 * 127.0.0.1:8760. Aucun import de `aura.*` ici, aucune duplication de logique.
 *
 * Langage visuel (celui de l'écran d'accueil de Codex, avec l'emblème RontoGPT) :
 *   - fond quasi noir, aucun cadre autour de la conversation ;
 *   - en-tête `>_ RontoGPT (vX.Y.Z)` et dossier courant en dessous ;
 *   - emblème RontoGPT très discret, centré, tant que le fil est vide ;
 *   - ta question : bande grise pleine largeur, précédée de `›` ;
 *   - la réponse : texte nu précédé de `•`, un cadre par bloc de code, une
 *     section « 🔗 Sources », puis une ligne de méta discrète (src/affichage.ts) ;
 *   - attente : puce clignotante + « Réflexion (12 s • Échap pour interrompre) » ;
 *   - barre de saisie grise de 3 lignes, et deux lignes de pied de page.
 */
import { homedir } from "node:os";
import {
  BoxRenderable,
  InputRenderable,
  InputRenderableEvents,
  ScrollBoxRenderable,
  TextRenderable,
  bold,
  createCliRenderer,
  fg,
  t,
} from "@opentui/core";
import { BASE, etat, formaterDuree, reinitialiser, repondre, sante } from "./client";
import type { Evenement, InfoEtat } from "./client";
import { LOGO, LOGO_HAUTEUR, LOGO_LARGEUR, SLOGAN } from "./logo";
import { TH, creerCorps, creerStyleCode, libelleAttente } from "./affichage";
import type { Corps, OptionsPeinture } from "./affichage";

/** Dossier courant, avec ~ pour le dossier personnel (comme Codex). */
function dossier(): string {
  const cwd = process.cwd();
  const home = homedir();
  return home && cwd.toLowerCase().startsWith(home.toLowerCase())
    ? `~${cwd.slice(home.length)}`
    : cwd;
}

// -- squelette de l'écran ---------------------------------------------------
const renderer = await createCliRenderer({
  exitOnCtrlC: true,
  backgroundColor: TH.fond,
});

// Le style de coloration des blocs de code : créé une fois, lié à cet écran.
const styleCode = creerStyleCode(renderer);

const racine = new BoxRenderable(renderer, {
  width: "100%",
  height: "100%",
  flexDirection: "column",
  backgroundColor: TH.fond,
});
renderer.root.add(racine);

// En-tête : `>_ RontoGPT (v0.1.0)` puis le dossier courant.
const entete = new BoxRenderable(renderer, {
  width: "100%",
  height: 3,
  flexDirection: "column",
  paddingLeft: 1,
  paddingTop: 1,
  backgroundColor: TH.fond,
});
const enteteTitre = new TextRenderable(renderer, { content: "", fg: TH.texte });
const enteteDossier = new TextRenderable(renderer, { content: "", fg: TH.discret });

/** Dossier courant, tronqué à gauche s'il dépasse la largeur (jamais sur 2 lignes). */
function majDossier(): void {
  const plein = dossier();
  const max = Math.max(10, renderer.width - 6);
  enteteDossier.content = `   ${plein.length > max ? `…${plein.slice(plein.length - max + 1)}` : plein}`;
}
majDossier();
renderer.on("resize", majDossier);
entete.add(enteteTitre);
entete.add(enteteDossier);
racine.add(entete);

function majTitre(version: string | null): void {
  enteteTitre.content = version
    ? t`${fg(TH.accent)(">_")} ${bold(fg("#ffffff")("RontoGPT"))} ${fg(TH.discret)(`(v${version})`)}`
    : t`${fg(TH.accent)(">_")} ${bold(fg("#ffffff")("RontoGPT"))}`;
}
majTitre(null);

// Écran d'accueil : l'emblème, tant que le fil est vide.
const accueil = new BoxRenderable(renderer, {
  width: "100%",
  flexGrow: 1,
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
  gap: 1,
  backgroundColor: TH.fond,
});
const logo = new TextRenderable(renderer, { content: LOGO.join("\n"), fg: TH.logo });
const slogan = new TextRenderable(renderer, { content: SLOGAN, fg: TH.discret });
accueil.add(logo);
accueil.add(slogan);
racine.add(accueil);

/** L'emblème ne s'affiche que si l'écran est assez grand pour le contenir. */
function ajusterAccueil(): void {
  logo.visible = renderer.height >= LOGO_HAUTEUR + 12 && renderer.width >= LOGO_LARGEUR + 4;
}
ajusterAccueil();
renderer.on("resize", ajusterAccueil);

// Le fil de conversation (masqué à l'accueil), sans bordure ni cadre.
const fil = new ScrollBoxRenderable(renderer, {
  width: "100%",
  flexGrow: 1,
  stickyScroll: true,
  stickyStart: "bottom",
  backgroundColor: TH.fond,
  paddingBottom: 1,
});
fil.visible = false;
racine.add(fil);

// Barre de saisie : bande grise de 3 lignes, `›` à gauche, texte sur la ligne du milieu.
const composer = new BoxRenderable(renderer, {
  width: "100%",
  height: 3,
  flexDirection: "row",
  alignItems: "center",
  backgroundColor: TH.bande,
  paddingLeft: 1,
});
const invite = new TextRenderable(renderer, { content: "›", fg: TH.discret, width: 2 });
const saisie = new InputRenderable(renderer, {
  id: "question",
  width: "100%",
  placeholder: "Demandez n'importe quoi à RontoGPT",
  backgroundColor: TH.bande,
  focusedBackgroundColor: TH.bande,
  textColor: TH.texte,
  placeholderColor: TH.discret,
  cursorColor: TH.texte,
});
saisie.flexGrow = 1;
composer.add(invite);
composer.add(saisie);
racine.add(composer);
saisie.focus();

// Pied de page : ligne « modèle » puis raccourcis.
const pied = new BoxRenderable(renderer, {
  width: "100%",
  height: 2,
  flexDirection: "column",
  paddingLeft: 3,
  backgroundColor: TH.fond,
});
const piedEtat = new TextRenderable(renderer, { content: "RontoGPT · connexion…", fg: TH.modele });
const piedAide = new TextRenderable(renderer, {
  content: t`${fg(TH.doux)("/")} ${fg(TH.discret)("commandes")} ${fg(TH.trait)("·")} ${fg(TH.doux)("Échap")} ${fg(TH.discret)("interrompre")} ${fg(TH.trait)("·")} ${fg(TH.doux)("Ctrl+C")} ${fg(TH.discret)("quitter")}`,
  fg: TH.discret,
});
pied.add(piedEtat);
pied.add(piedAide);
racine.add(pied);

// -- helpers d'affichage ----------------------------------------------------
function descendre(): void {
  const hauteur = (fil as unknown as { scrollHeight?: number }).scrollHeight;
  if (typeof hauteur === "number") fil.scrollTo({ x: 0, y: hauteur });
}

/** Passe de l'écran d'accueil au fil dès qu'il y a quelque chose à montrer. */
function montrerFil(): void {
  if (!fil.visible) {
    accueil.visible = false;
    fil.visible = true;
  }
}

function montrerAccueil(): void {
  fil.visible = false;
  accueil.visible = true;
}

/**
 * Un bloc « puce + texte » : la puce est posée en absolu dans la marge, le
 * texte garde donc une retraite fixe et se replie proprement (comme Codex).
 */
function blocAvecPuce(
  glyphe: string,
  couleur: string,
  fond?: string,
): { bloc: BoxRenderable; puce: TextRenderable } {
  // marginTop explicite : le `gap` d'un ScrollBox ne sépare pas ses enfants.
  const bloc = new BoxRenderable(renderer, {
    width: "100%",
    flexDirection: "column",
    marginTop: 1,
    paddingLeft: 3,
    paddingRight: 1,
    ...(fond ? { backgroundColor: fond } : {}),
  });
  const puce = new TextRenderable(renderer, {
    content: glyphe,
    fg: couleur,
    position: "absolute",
    left: 1,
    top: 0,
  });
  bloc.add(puce);
  return { bloc, puce };
}

/** Note hors conversation (aide, état, erreur) : même mise en page qu'une réponse. */
function note(texte: string, couleur: string): TextRenderable {
  montrerFil();
  const { bloc } = blocAvecPuce("•", couleur);
  const corps = new TextRenderable(renderer, {
    content: texte,
    fg: couleur,
    wrapMode: "word",
    width: "100%",
  });
  bloc.add(corps);
  fil.add(bloc);
  descendre();
  return corps;
}

const AIDE = [
  "Commandes",
  "  /aide      cet écran                     /effacer   vide le fil affiché",
  "  /oubli     le cerveau oublie la conversation (reset)",
  "  /etat      profil, llama.cpp, budget du cerveau",
  "  /quitter   sortir",
  "  Échap      interrompt la réponse en cours    Ctrl+C  quitter",
].join("\n");

// -- en-tête / pied de page -------------------------------------------------
function majEntete(info: InfoEtat | null, nota = ""): void {
  if (info === null) {
    majTitre(null);
    piedEtat.content = `RontoGPT hors ligne${nota}`;
    piedEtat.fg = TH.erreur;
    return;
  }
  majTitre(info.version);
  const cerveau = info.prete ? "cerveau prêt"
    : info.chargement ? "chargement du cerveau…" : "cerveau à froid";
  piedEtat.content = `RontoGPT ${info.profil} · ${cerveau}${nota}`;
  piedEtat.fg = info.prete ? TH.modele : TH.alerte;
}

// -- un tour de conversation ------------------------------------------------
let enVol: false | AbortController = false;

interface Tour {
  puce: TextRenderable;
  attente: TextRenderable;
  corps: Corps;
  meta: TextRenderable;
}

function tour(question: string): Tour {
  montrerFil();

  // Question : bande grise pleine largeur, `›` dans la marge.
  const utilisateur = blocAvecPuce("›", TH.doux, TH.bande);
  utilisateur.bloc.add(new TextRenderable(renderer, {
    content: question,
    fg: TH.texte,
    wrapMode: "word",
    width: "100%",
  }));
  fil.add(utilisateur.bloc);

  // Réponse : `•` dans la marge (il clignote pendant l'attente).
  const reponse = blocAvecPuce("•", TH.discret);
  const attente = new TextRenderable(renderer, { content: "", fg: TH.doux });
  const corps = creerCorps(renderer, styleCode);
  const meta = new TextRenderable(renderer, { content: "", fg: TH.trait, width: "100%" });
  corps.bloc.visible = false;  // pas de ligne vide tant qu'il n'y a rien à dire
  meta.visible = false;
  reponse.bloc.add(attente);
  reponse.bloc.add(corps.bloc);
  reponse.bloc.add(meta);
  fil.add(reponse.bloc);
  descendre();
  return { puce: reponse.puce, attente, corps, meta };
}

/** Ligne de méta : `─ niveau0-instantane · 0.0 s · experts : … ─────`. */
function ligneMeta(e: Extract<Evenement, { type: "final" }>, morceaux: number): string {
  const parts = [e.cerveau, `${e.latence.toFixed(1)} s`];
  if (e.experts.length > 0) parts.push(`experts : ${e.experts.join(", ")}`);
  if (morceaux > 0) parts.push(`${morceaux} morceaux`);
  return habillerMeta(parts.join(" · "));
}

function habillerMeta(texte: string): string {
  const largeur = Math.max(12, renderer.width - 10);
  return `─ ${texte} `.padEnd(largeur, "─");
}

// -- envoi ------------------------------------------------------------------
async function envoyer(question: string): Promise<string> {
  if (enVol !== false) {
    note("Un tour est déjà en cours — Échap pour l'interrompre.", TH.alerte);
    return "";
  }
  const { puce, attente, corps, meta } = tour(question);
  const controleur = new AbortController();
  enVol = controleur;

  const debut = Date.now();
  let phase = "Réflexion";
  let texte = "";        // morceaux cumulés (flux du cerveau)
  let affiche = "";      // ce qui est réellement à l'écran
  let morceaux = 0;
  // NB : `content` d'un TextRenderable est un StyledText, pas une chaîne : on
  // ne l'ADDITIONNE jamais (`meta.content += …` affichait « [object Object] »),
  // on l'affecte toujours avec un texte déjà complet.

  attente.content = `${phase} (0 s • Échap pour interrompre)`;
  const tic = setInterval(() => {
    const s = (Date.now() - debut) / 1000;
    // Pas de total connu : on affiche le compte de morceaux reçus (indicateur
    // de progression réel) plutôt qu'une barre dont la fin serait inventée.
    attente.content = libelleAttente(phase, s, morceaux);
    attente.fg = TH.doux;
    puce.content = Math.floor(s * 2) % 2 === 0 ? "•" : "◦";   // puce clignotante
    puce.fg = TH.alerte;
  }, 250);

  const finirAttente = (): void => {
    clearInterval(tic);
    attente.content = "";
    attente.visible = false;
    puce.content = "•";
    puce.fg = TH.discret;
  };
  const poser = (contenu: string, options: OptionsPeinture = {}): void => {
    corps.peindre(contenu, options);        // riche par défaut : cadres + sources
  };
  const poserMeta = (contenu: string, couleur: string): void => {
    meta.content = contenu;
    meta.fg = couleur;
    meta.visible = true;
  };

  try {
    for await (const e of repondre(question, { signal: controleur.signal })) {
      if (e.type === "etape") {
        phase = e.libelle;                  // annonce du cerveau : recherche, lecture…
      } else if (e.type === "morceau") {
        morceaux += 1;
        phase = e.phase === "reflexion" ? "Réflexion" : "Rédaction";
        texte += e.texte;
        affiche = texte;
        poser(affiche, { riche: false });   // le design complet arrive au `final`
        descendre();
      } else if (e.type === "final") {
        finirAttente();
        let couleur: string = TH.texte;
        if (e.reponse && e.reponse.trim() !== "") {
          affiche = e.reponse;
        } else if (affiche === "") {
          affiche = "(réponse vide)";
          couleur = TH.alerte;
        }
        if (e.clarification) {
          affiche += `\n\nÀ préciser : ${e.clarification}`;
        }
        poser(affiche, { couleur });
        poserMeta(ligneMeta(e, morceaux), TH.trait);
        if (!e.ok && e.erreur) note(`Le cerveau signale : ${e.erreur}`, TH.erreur);
      } else if (e.type === "erreur") {
        finirAttente();
        affiche = e.erreur;
        poser(affiche, { couleur: TH.erreur });
        poserMeta(habillerMeta(`échec après ${formaterDuree((Date.now() - debut) / 1000)}`), TH.erreur);
      }
    }
  } catch (e) {
    const interrompu = (e as Error)?.name === "AbortError";
    const raison = interrompu ? "interrompu" : (e as Error).message;
    if (affiche === "") {
      affiche = `(${raison})`;
      if (!interrompu) poser(affiche);      // une interruption ne se dit qu'une fois (méta)
    } else {
      poser(affiche);                       // le flux s'arrête : la réponse prend sa forme
    }
    poserMeta(habillerMeta(raison), TH.alerte);
  } finally {
    finirAttente();
    enVol = false;
    descendre();
  }
  return affiche;
}

// -- saisie -----------------------------------------------------------------
function normaliser(brut: string): string {
  return brut.trim().replace(/^\//, "").toLowerCase();
}

async function traiter(brut: string): Promise<void> {
  const texte = brut.trim();
  if (texte === "") return;
  const cmd = normaliser(texte);

  // Les mots nus marchent aussi : `clear`, `reset`, `exit` (l'ancien CLI les
  // acceptait, et `clear` partait sinon au cerveau comme une vraie question).
  if (["aide", "help", "?"].includes(cmd)) {
    note(AIDE, TH.doux);
    return;
  }
  if (["effacer", "clear", "cls"].includes(cmd)) {
    for (const enfant of [...fil.getChildren()]) enfant.destroy();
    montrerAccueil();
    slogan.content = "Fil effacé (la mémoire du cerveau est intacte — /oubli pour l'oublier)";
    slogan.fg = TH.discret;
    return;
  }
  if (["oubli", "reset", "oublier"].includes(cmd)) {
    const r = await reinitialiser();
    note(r.ok ? `Cerveau : ${r.detail}` : `Impossible d'oublier : ${r.detail}`,
         r.ok ? TH.ok : TH.erreur);
    return;
  }
  if (["etat", "status", "statut"].includes(cmd)) {
    const info = await etat();
    majEntete(info);
    if (info === null) {
      note(`API injoignable sur ${BASE}`, TH.erreur);
      return;
    }
    note(
      `version ${info.version} · profil ${info.profil} · cerveau ${info.prete ? "prêt" : "à froid"}`
      + `\nllama.cpp ${info.serveur_llama ? "actif" : "arrêté"} · GPU ${info.gpu} · budget ${info.budget_s} s`
      + (info.erreur ? `\nerreur : ${info.erreur}` : ""),
      TH.doux,
    );
    return;
  }
  if (["quitter", "exit", "quit", "q", "bye"].includes(cmd)) {
    renderer.destroy();
    process.exit(0);
  }

  await envoyer(texte);
}

saisie.on(InputRenderableEvents.ENTER, (valeur: string) => {
  saisie.value = "";
  void traiter(valeur);
});

renderer.keyInput.on("keypress", (key) => {
  if (key.name === "escape" && enVol !== false) {
    enVol.abort();
  }
});

// -- démarrage --------------------------------------------------------------
const santeInfo = await sante();
if (santeInfo === null) {
  majEntete(null);
  slogan.content = `API injoignable sur ${BASE} — lance :  uv run rontogpt`;
  slogan.fg = TH.erreur;
} else {
  majEntete(await etat());
}

// Le pied de page suit l'état du cerveau (chargement au premier appel, llama.cpp…).
setInterval(() => { void etat().then((info) => majEntete(info)); }, 5000);

// -- mode --smoke : parcours complet sans humain ----------------------------
// Monte vraiment l'écran, joue une question, puis ressort sur le code 0.
// C'est ce qui rend la mise en page vérifiable en machine (voir README).
if (process.argv.includes("--smoke")) {
  const question = process.env.AURA_TUI_SMOKE_Q ?? "Combien font 6 * 7 ?";
  const reponse = await envoyer(question);
  await new Promise((r) => setTimeout(r, 1500));   // laisse partir la frame finale
  const blocs = fil.getChildren().length;
  renderer.destroy();
  console.log(`[smoke] écran monté · ${blocs} bloc(s) dans le fil · question : ${question}`
              + ` · réponse affichée : ${JSON.stringify(reponse.slice(0, 120))}`);
  process.exit(blocs >= 2 && reponse.trim() !== "" ? 0 : 1);
}
