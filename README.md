# 🚆 Bahngleiserfassung – Video Frame Analyzer (Streamlit + HF + Ontologie)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](#)
[![Status](https://img.shields.io/badge/Build-ok-brightgreen.svg)](#)

Kurzerklärung: Dieses Projekt extrahiert Videoframes (Bahnsteigkamera) und bewertet die Szene **ontologie-basiert** – z. B. „**Person befindet sich im Gleis**“ ⇒ **kritische** Meldung.  
Die Bewertung ist **erklärbar** (Regeln + RDF-artige Tripel).

> 📘 **Detailseite (Extra-Tab):** [Ontologie & Regeln – Deep Dive](docs/ONTOLOGIE.md)

---

## 🧭 Inhalt
- [🚆 Bahngleiserfassung – Video Frame Analyzer (Streamlit + HF + Ontologie)](#-bahngleiserfassung--video-frame-analyzer-streamlit--hf--ontologie)
  - [🧭 Inhalt](#-inhalt)
  - [✨ Features](#-features)
  - [🏗️ Projektstruktur](#️-projektstruktur)
  - [🚀 Schnellstart](#-schnellstart)
  - [⚙️ Konfiguration](#️-konfiguration)
  - [🖥️ Nutzung](#️-nutzung)
  - [🧠 Was ist eine Ontologie?](#-was-ist-eine-ontologie)
  - [🧪 Tests \& Tripel-Export](#-tests--tripel-export)
  - [🛡️ Sicherheit (Secrets)](#️-sicherheit-secrets)
  - [🧰 Troubleshooting](#-troubleshooting)
  - [📄 Lizenz](#-lizenz)
  - [🔌 Integration](#-integration)
  - [🧩 Erweiterung](#-erweiterung)

---

## ✨ Features
- 📼 **Video-Upload** (MP4/AVI/MOV/MKV), auto-Frame-Extraktion  
- 🤖 **HF-Modelle** (Vision/Language) zur Szeneninterpretation  
- 🧩 **Ontologie-Bewertung** (Regeln wie `befindetSichIn(Gleis)`)  
- 🧾 **Erklärungen** (welche Regeln ausgelöst haben)  
- 🧷 **Tripel-Export** (Turtle-ähnlich) zur Weiterverarbeitung

---

## 🏗️ Projektstruktur
```

.
├─ app.py                      # Streamlit-App (UI ohne Freitext-Prompts)
├─ ontology\_eval.py            # Regeln + Ontologie-Tripel-Export
├─ test\_ontology\_triples.py    # Mini-Test + Turtle-Ausgabe
├─ detect\_person\_on\_tracks.py  # Beispiel-Analyse (einbinden/erweitern)
├─ requirements.txt
├─ settings.json.example
├─ .env.example
└─ docs/
└─ ONTOLOGIE.md             # Detaildoku (Extra-Tab)

````

---

## 🚀 Schnellstart
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
````

---

## ⚙️ Konfiguration

**Hugging Face Token** (nicht committen):

```powershell
# Windows (Session)
$env:HF_TOKEN="hf_xxx_dein_token"
# Optional dauerhaft:
setx HF_TOKEN "hf_xxx_dein_token"
```

Optionale App-Settings:

```bash
cp settings.json.example settings.json
```

---

## 🖥️ Nutzung

```bash
streamlit run app.py
```

* 📤 Video hochladen → **Analyse startet automatisch** (keine Freitext-Prompts).
* 🧯 Ergebnis als **Meldung** mit Icon:

  * ✅ NONE / 🟢 LOW / 🟠 MEDIUM / ⚠️ HIGH / 🚨 CRITICAL
* 🔗 Button „Details im neuen Fenster“ öffnet die Ergebnisansicht in **neuem Tab**.

---

## 🧠 Was ist eine Ontologie?

Eine **Ontologie** beschreibt die Domäne formal (Klassen/Beziehungen/Eigenschaften), z. B.:

* **Klassen:** `Person`, `Gleis`, `Bahnsteig`, `Zug`, `Gefahr`, `Sensor`, `Videoüberwachung`, `Alarmsystem`, `Maßnahme`, `Ereignis`, `Objekt`
* **Objekt-Properties:** `befindetSichIn`, `erkennt`, `stehtAuf`, `beobachtet`, `überwacht`, `löstAus`, `führtZu`, `meldet`
* **Daten-Properties:** `hatKonfidenz (xsd:float)`, `hatZeitstempel (xsd:dateTime)`, `hatPosition (xsd:string)`, `hatBeschreibung (xsd:string)`

So wird aus ML-Signalen **bedeutungsvolle** Logik:
„Person im Gleis“ ⇒ **Gefahr** ⇒ **Alarm** ⇒ **Maßnahme**.

> 🔎 Mehr Details inkl. Regeln (R1–R6) und Turtle-Beispielen:
> **[docs/ONTOLOGIE.md](docs/ONTOLOGIE.md)**

---

## 🧪 Tests & Tripel-Export

Schneller Test der Bewertung & Tripel:

```bash
python test_ontology_triples.py
```

Ausgabe: Severity/Score/Labels/Erklärungen + Turtle-Tripel.

---

## 🛡️ Sicherheit (Secrets)

* **Keine** Tokens/Passwörter in Code/Repo einchecken.
* `.env` ist ignoriert (`.gitignore`).
* Bei Leak: **Token sofort revoken/rotieren** (HF-Settings).

---

## 🧰 Troubleshooting

* ❗ **Kein HF\_TOKEN gefunden** → Token setzen (s. o.).
* 🧩 **FFmpeg fehlt** → installieren und zum `PATH` hinzufügen (für robuste Video-Extraktion).
* 🔁 **Zeilenende-Warnungen (CRLF/LF)** → harmlos; ggf. `git config --global core.autocrlf true`.

---

## 📄 Lizenz

tbd (z. B. MIT)

````

---

### `docs/ONTOLOGIE.md`

```markdown
# 📘 Ontologie & Regeln – Deep Dive

Diese Seite beschreibt die Ontologie, das Regelwerk und die erzeugten Tripel.

## 🧠 Ontologie (Auszug)
**Klassen:** `Person`, `Gleis`, `Bahnsteig`, `Zug`, `Gefahr`, `Sensor`, `Videoüberwachung`, `Alarmsystem`, `Maßnahme`, `Ereignis`, `Objekt`  
**Objekt-Properties:** `befindetSichIn`, `erkennt`, `überwacht`, `beobachtet`, `stehtAuf`, `löstAus`, `führtZu`, `meldet`  
**Daten-Properties:** `hatKonfidenz (xsd:float)`, `hatZeitstempel (xsd:dateTime)`, `hatPosition (xsd:string)`, `hatBeschreibung (xsd:string)`

## ⚖️ Bewertungslogik (R1–R6)
- **R1 – Person im Gleis** → `CRITICAL`  
  `on_track_person ≥ Schwelle` ⇒ Tripel: `ex:person ex:befindetSichIn ex:gleis`
- **R2 – Nahe Kante + Zug** → `HIGH`  
  `distance_to_edge ≤ 0.5m` ∧ `train_approaching ≥ Schwelle`
- **R3 – Gestürzte Person nahe Kante/auf Gleis** → `HIGH/CRITICAL`
- **R4 – Objekt im Gleis** → `MEDIUM`
- **R5 – Rauch/Feuer** → `HIGH`
- **R6 – Menschenmenge im Gleisbereich** → `CRITICAL`

**Recall-Bias:** Standard-Schwelle `0.35` (Sicherheitsdomäne → lieber einmal zu viel melden).

## 🧾 Tripel-Export (Turtle-ähnlich)
Beispielauszug:
```turtle
ex:person42 rdf:type ex:Person .
ex:gleis_3   rdf:type ex:Gleis .
ex:person42  ex:befindetSichIn ex:gleis_3 .
ex:gef1      rdf:type ex:Gefahr .
ex:gef1      ex:löstAus ex:alarm_4711 .
ex:alarm_4711 ex:führtZu ex:massnahme_stop .
ex:event_utc ex:hatZeitstempel "2025-09-06T14:32:10Z"^^xsd:dateTime .
````

## 🔌 Integration

Im Code (Beispiel):

```python
from ontology_eval import Observation, evaluate, OntologyContext, decision_to_triples, triples_to_turtle

obs = Observation(on_track_person=0.88, distance_to_edge_m=0.3, train_approaching=0.9)
dec = evaluate(obs)

ctx = OntologyContext(person_id="person42", track_id="gleis_3", platform_id="bahnsteig_3")
print(triples_to_turtle(decision_to_triples(dec, obs, ctx)))
```

## 🧩 Erweiterung

* Neue Klassen/Properties ergänzen (z. B. `Kinderwagen`, `Warnweste`)
* Weitere Regeln (z. B. „Sperrbereich aktiv“ ⇒ höhere Schwere)
* Export als **TTL/JSON-LD/CSV** für Downstream-Systeme


test