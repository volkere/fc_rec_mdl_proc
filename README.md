# Gesichtserkennung & Altersschätzung mit KI

Dieses Projekt nutzt **dlib**, **OpenCV** und **TensorFlow**, um Gesichter aus Bildern zu erkennen und ihr Alter vorherzusagen. Die erkannten Gesichter werden sortiert und in einem Ausgabeordner gespeichert.

## Features
- **Gesichtserkennung** mit `dlib`
- **Altersschätzung** mit einem trainierten `TensorFlow`-Modell
- **Automatische Sortierung** der Gesichter nach geschätztem Alter
- **Unterstützung für mehrere Bildformate** (`.jpg`, `.png`, `.jpeg`)

---

## Verzeichnisstruktur
```
project_root/
│-- analog_portraits/    # Eingangsordner mit Bildern
│-- sorted_faces/        # Ausgabeordner mit sortierten Gesichtern
│-- age_model.keras      # Trainiertes KI-Modell zur Altersschätzung
│-- allto1.py            # Hauptskript zur Verarbeitung
│-- train_model.py       # Skript zum Trainieren eines neuen Modells
│-- README.md            # Dokumentation
```

---

## Installation
### **Erforderliche Abhängigkeiten installieren**
```sh
pip install opencv-python dlib tensorflow numpy
```
Falls `dlib` Probleme macht, installiere es mit:
```sh
pip install --no-cache-dir dlib
```
Für Apple Silicon (`M1/M2/M3`):
```sh
brew install dlib
```

### **Projektstruktur einrichten**
Erstelle die Eingangs- und Ausgangsordner:
```sh
mkdir analog_portraits sorted_faces
```

---

## Nutzung
### Gesichter erkennen & Alter schätzen**
Lege deine Bilder in `analog_portraits/` ab und starte das Skript:
```sh
python allto1.py
```
Erkannte Gesichter werden im `sorted_faces/` Ordner gespeichert.

### Neues Modell trainieren (optional)**
Falls du ein neues Modell trainieren möchtest:
```sh
python train_model.py
```
Nach dem Training wird das Modell als `age_model.keras` gespeichert.

## Lizenz
MIT License

---

## Mitwirkende
- **@blaubaer** – Hauptentwickler
- **Contributors willkommen!** Stelle gerne Pull-Requests mit Verbesserungen ein.

---

## Unterstützung
Wenn dir das Projekt gefällt, ⭐️ es auf GitHub! 😊

