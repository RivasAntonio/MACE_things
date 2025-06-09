import sys

def extraer_elementos_xyz(ruta_archivo):
    elementos = set()

    with open(ruta_archivo, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            # Ignorar líneas con metadatos
            if "Lattice=" in linea or "Properties=" in linea or "Energy=" in linea:
                continue

            palabras = linea.split()
            if not palabras:
                continue

            primera = palabras[0]
            # Asegurarse de que parezca un símbolo de elemento (ej. O, Si, H, Al, etc.)
            if primera.isalpha() and primera[0].isupper():
                elementos.add(primera)

    return sorted(elementos)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python extraer_elementos.py archivo.xyz")
        sys.exit(1)

    archivo = sys.argv[1]
    elementos = extraer_elementos_xyz(archivo)
    print("Elementos encontrados:", elementos)
