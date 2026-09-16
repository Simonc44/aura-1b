"""Genere un dataset bilingue (francais + anglais) pour fine-tuning RWKV.

Principe : on melange des exemples francais (R1-U300) et anglais (templates)
pour que le modele apprenne les deux langues simultanement.

Sortie : donnees_bilingues.jsonl
"""
import json
import os

# exemples anglais d'instruction (templates de base)
EXEMPLES_ANGLAIS = [
    {"user": "What is the capital of France?", "assistant": "The capital of France is Paris."},
    {"user": "What is 2 + 2?", "assistant": "2 + 2 = 4."},
    {"user": "Explain gravity", "assistant": "Gravity is the force that attracts objects toward each other. The more massive an object, the stronger its gravitational pull."},
    {"user": "What is Python?", "assistant": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and AI."},
    {"user": "How do you make coffee?", "assistant": "To make coffee: boil water, add ground coffee beans to a filter or press, pour hot water over them, wait 3-4 minutes, then serve."},
    {"user": "What is the speed of light?", "assistant": "The speed of light in vacuum is approximately 299,792,458 meters per second (about 300,000 km/s)."},
    {"user": "Tell me a joke", "assistant": "Why do programmers prefer dark mode? Because light attracts bugs!"},
    {"user": "What is machine learning?", "assistant": "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without being explicitly programmed."},
    {"user": "How old is the Earth?", "assistant": "The Earth is approximately 4.54 billion years old, based on radiometric dating of meteorite samples."},
    {"user": "What is DNA?", "assistant": "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions for life. It's shaped like a double helix and made of nucleotide bases: A, T, G, and C."},
    {"user": "Explain photosynthesis", "assistant": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in chloroplasts using chlorophyll."},
    {"user": "What is the Pythagorean theorem?", "assistant": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c²."},
    {"user": "How do computers work?", "assistant": "Computers work by processing binary data (0s and 1s) through circuits. The CPU executes instructions, memory stores data, and output devices display results."},
    {"user": "What is climate change?", "assistant": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities like burning fossil fuels that increase greenhouse gases."},
    {"user": "What is electricity?", "assistant": "Electricity is the flow of electric charge through a conductor. It's caused by the movement of electrons and can be used to power devices and transmit energy."},
    {"user": "Explain evolution", "assistant": "Evolution is the process by which species change over time through natural selection. Organisms with favorable traits survive and reproduce more, passing those traits to offspring."},
    {"user": "What is the internet?", "assistant": "The internet is a global network of interconnected computers that communicate using standardized protocols. It enables sharing of information, email, and web browsing."},
    {"user": "How does music work?", "assistant": "Music works through sound waves created by vibrations. Different frequencies produce different pitches, and combinations of pitches create harmony and melody."},
    {"user": "What is chemistry?", "assistant": "Chemistry is the study of matter, its properties, composition, and the changes it undergoes. It explores how atoms combine to form molecules and compounds."},
    {"user": "Explain black holes", "assistant": "Black holes are regions of space where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycle."},
]


def generer(chemin_r1: str = None, chemin_sortie: str = "donnees_bilingues.jsonl"):
    """Genere le dataset bilingue."""
    exemples = []

    # ajouter les exemples anglais
    exemples.extend(EXEMPLES_ANGLAIS)
    print(f"anglais : {len(EXEMPLES_ANGLAIS)} exemples")

    # ajouter les exemples francais depuis R1-U300 si disponible
    if chemin_r1 and os.path.exists(chemin_r1):
        with open(chemin_r1, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    d = json.loads(ligne)
                    if "user" in d and "assistant" in d:
                        exemples.append(d)
                except:
                    pass
        print(f"francais : {len(exemples) - len(EXEMPLES_ANGLAIS)} exemples depuis R1-U300")

    # melanger
    import random
    random.shuffle(exemples)

    # ecrire
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        for e in exemples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"total : {len(exemples)} exemples bilingues -> {chemin_sortie}")
    return exemples


if __name__ == "__main__":
    import sys
    r1 = sys.argv[1] if len(sys.argv) > 1 else None
    generer(r1)
