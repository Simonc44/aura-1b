/**
 * Client HTTP du cerveau RontoGPT — `aura/api.py`, streaming SSE.
 *
 * VOLONTAIREMENT sans aucune dependance a OpenTUI : ce module se teste seul
 * (`bun src/smoke.ts`), donc la partie risquee (le protocole) est verifiee
 * sans avoir besoin d'un vrai terminal.
 *
 * Contrat reel de `/v1/ask/stream` (mesure sur cette machine) :
 *   - `morceau` {texte, phase} : uniquement quand le cerveau GENERE des
 *     tokens. La plupart des questions partent aux couches deterministes
 *     (niveau0-instantane, rag-binaire, extraction-web) et arrivent donc
 *     d'un bloc, sans aucun morceau ;
 *   - `final`  {reponse, experts, cerveau, latence, en_attente, clarification} ;
 *   - `erreur` {erreur} ;
 *   - `fin`, puis la ligne sentinelle `data: [DONE]`.
 */
export const BASE = process.env.AURA_API_URL ?? "http://127.0.0.1:8760";
const CLE = process.env.AURA_API_CLE ?? "";

function entetes(json = false): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (CLE) h["X-RontoGPT-Cle"] = CLE;      // meme cle que la GUI (api.py)
  return h;
}

export type Phase = string;                // "reflexion" | "reponse" | ...

export interface Morceau { type: "morceau"; texte: string; phase: Phase }
export interface Final {
  type: "final";
  ok: boolean;
  reponse: string;
  experts: string[];
  cerveau: string;
  latence: number;
  en_attente: boolean;
  clarification: string | null;
  erreur?: string;
}
export interface ErreurFlux { type: "erreur"; erreur: string }
export interface Fin { type: "fin" }
export interface Etape { type: "etape"; libelle: string }
export type Evenement = Morceau | Etape | Final | ErreurFlux | Fin;

export interface InfoEtat {
  version: string;
  profil: string;
  prete: boolean;
  chargement: boolean;
  erreur: string | null;
  serveur_llama: boolean;
  gpu: string;
  budget_s: string;
}

async function lire<T>(chemin: string, init: RequestInit = {},
                       delai = 5000): Promise<T | null> {
  try {
    const r = await fetch(`${BASE}${chemin}`, {
      ...init,
      headers: { ...entetes(), ...(init.headers ?? {}) },
      signal: AbortSignal.timeout(delai),
    });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;                                    // API absente : cas normal au demarrage
  }
}

export const sante = () => lire<{ status: string; version: string }>("/v1/health");
export const etat = () => lire<InfoEtat>("/v1/status");

/** Oublie la conversation cote cerveau (equivalent du `reset` en CLI). */
export async function reinitialiser(): Promise<{ ok: boolean; detail: string }> {
  const r = await lire<{ ok: boolean; oublie?: boolean; raison?: string; erreur?: string }>(
    "/v1/reset", { method: "POST", headers: entetes(true), body: "{}" }, 20000);
  if (r === null) return { ok: false, detail: "API injoignable (ou route /v1/reset absente)" };
  if (!r.ok) return { ok: false, detail: r.erreur ?? "echec" };
  return { ok: true, detail: r.raison ?? "conversation oubliee" };
}

/** Repond en flux. Rend les `morceau` au fil de l'eau puis un `final`. */
export async function* repondre(
  question: string,
  opts: { signal?: AbortSignal } = {},
): AsyncGenerator<Evenement> {
  let r: Response;
  try {
    r = await fetch(`${BASE}/v1/ask/stream`, {
      method: "POST",
      headers: entetes(true),
      body: JSON.stringify({ question }),
      signal: opts.signal,
    });
  } catch (e) {
    yield { type: "erreur", erreur: `API injoignable (${BASE}) : ${(e as Error).message}` };
    return;
  }
  if (!r.ok || r.body === null) {
    let detail = `HTTP ${r.status}`;
    try {
      const corps = (await r.json()) as { erreur?: string };
      detail = corps.erreur ?? detail;          // ex. 409 « un flux est deja en cours »
    } catch { /* corps non JSON : on garde le code */ }
    yield { type: "erreur", erreur: detail };
    return;
  }

  const lecteur = r.body.getReader();
  const decodeur = new TextDecoder();
  let tampon = "";
  try {
    while (true) {
      const { done, value } = await lecteur.read();
      if (done) break;
      tampon += decodeur.decode(value, { stream: true });
      let coupe: number;
      while ((coupe = tampon.indexOf("\n\n")) !== -1) {
        const trame = tampon.slice(0, coupe);
        tampon = tampon.slice(coupe + 2);
        for (const ligne of trame.split("\n")) {
          if (!ligne.startsWith("data:")) continue;
          const brut = ligne.slice(5).trim();
          if (brut === "") continue;
          if (brut === "[DONE]") { yield { type: "fin" }; return; }
          try {
            yield JSON.parse(brut) as Evenement;
          } catch { /* trame tronquee : on l'ignore, le final arrive */ }
        }
      }
    }
  } finally {
    lecteur.releaseLock();
  }
}

export function formaterDuree(secondes: number): string {
  if (secondes < 60) return `${secondes.toFixed(1)} s`;
  const minutes = Math.floor(secondes / 60);
  return `${minutes} min ${Math.round(secondes - minutes * 60)} s`;
}
