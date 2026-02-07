# AgroFly Backend

Vytvárajte ortofotomapy z leteckých/dronových snímkov.

Dva režimy:

- **Lokálny (predvolený)** – OpenCV Stitcher, beží offline, žiadny server
- **WebODM** – potrebuje spustené WebODM

## Predpoklady

1. **Python 3.10+**
2. **Závislosti** – `pip install -r requirements.txt`

Pre lokálny režim stačí Python a závislosti. WebODM je potrebné len pri použití `--webodm`.

## Nastavenie

```bash
cd Backend
pip install -r requirements.txt
```

Umiestnite fotografie do `data/`. Podporované formáty: JPG, JPEG, PNG. Potrebujete min. 2 obrázky.

## Použitie

### Lokálny režim (predvolený) – OpenCV, offline

```bash
python src/create_orthomap.py
```

Načíta obrázky z `Backend/data/` a uloží ortofotomapu do `Backend/data/orthomap.png`. Žiadny server nie je potrebný.

```bash
python src/create_orthomap.py --local
python src/create_orthomap.py ./moje_fotky ./vystup/orthomap.png
```

### WebODM režim – potrebuje spustené WebODM

```bash
python src/create_orthomap.py --webodm
```

## Možnosti


| Možnosť              | Popis                                                                              |
| -------------------- | ---------------------------------------------------------------------------------- |
| `--local`            | OpenCV Stitcher (offline). Predvolené.                                             |
| `--webodm`           | WebODM API namiesto lokálneho OpenCV.                                              |
| `-m`, `--mode`       | Režim pre OpenCV: `panorama` alebo `scans` (lepšie pre nadhlavové dronové zábery). |
| `-q`, `--quiet`      | Minimálny výstup.                                                                  |
| `--url`              | WebODM URL (len pri `--webodm`).                                                   |
| `--user`             | WebODM používateľ (len pri `--webodm`).                                            |
| `--password`         | WebODM heslo (len pri `--webodm`).                                                 |
| `-r`, `--resolution` | Rozlíšenie ortofotomapy v cm/pixel (len pri `--webodm`).                           |
| `--no-cleanup`       | Ponechať projekt v WebODM (len pri `--webodm`).                                    |


Príklady:

```bash
python src/create_orthomap.py -m scans
```

## Riešenie problémov

- **"Need at least 2 images"** – Uistite sa, že `data/` obsahuje aspoň 2 súbory JPG/PNG.
- **"Stitching failed"** – Skúste `-m scans` alebo overte prekrytie snímkov (30–70 % odporúčané).
- **WebODM chyby** – Pri `--webodm` skontrolujte, či WebODM beží na danej URL.

