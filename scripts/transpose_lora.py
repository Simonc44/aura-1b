"""Transpose les tenseurs lora_a/lora_b d'un GGUF LoRA (conventions anciennes↔nouvelles).

Usage : python transpose_lora.py ENTREE.gguf SORTIE.gguf
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter


def main(entree: str, sortie: str) -> int:
    reader = GGUFReader(entree)
    writer = GGUFWriter(sortie, arch="llama")

    # copier toutes les metadonnees (scalaires + arrays)
    for name, field in reader.fields.items():
        if name == "general.architecture":
            continue  # regenere par GGUFWriter selon arch="llama"
        parts = field.parts
        data = quons = None
        try:
            if len(parts) > -1:
                data = b"".join(p.tobytes() for p in parts[-4:]) if False else None
        except Exception:
            pass
        # voie generique : add_key_value attend (key, valeur, type)
        try:
            import gguf
            val = field.parts[-1]
            # decoder selon le type
            t = field.types[0]
            if t == gguf.GGUFValueType.UINT32:
                writer.add_uint32(name, val[0])
            elif t == gguf.GGUFValueType.INT32:
                writer.add_int32(name, val[0])
            elif t == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(name, val[0])
            elif t == gguf.GGUFValueType.BOOL:
                writer.add_bool(name, bool(val[0]))
            elif t == gguf.GGUFValueType.STRING:
                writer.add_string(name, bytes(val).decode("utf-8", "replace"))
            elif t == gguf.GGUFValueType.UINT64:
                writer.add_uint64(name, val[0])
            elif t == gguf.GGUFValueType.INT64:
                writer.add_int64(name, val[0])
            elif t == gguf.GGUFValueType.FLOAT64:
                writer.add_float64(name, val[0])
            elif t == gguf.GGUFValueType.ARRAY:
                # arrays de strings (tags) et autres : re-emettre tels quels
                elems = field.data
                subtype = field.types[1]
                if subtype == gguf.GGUFValueType.STRING:
                    writer.add_array(name, [bytes(s).decode("utf-8", "replace") for s in elems])
                else:
                    writer.add_array(name, list(elems), vtype=subtype)
        except Exception as e:
            print(f"  [!] KV {name}: {e}")

    for t in reader.tensors:
        data = np.asarray(t.data)
        if t.name.endswith(".lora_a") or t.name.endswith(".lora_b"):
            data = data.T.copy()  # (in,r)->(r,in) ou inverse
        writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"OK {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
