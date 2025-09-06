# ontology_eval.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ============================================================================
#  ONTOLOGIE-ANBINDUNG (an die in deiner Grafik gezeigten Klassen/Properties)
#  --------------------------------------------------------------------------
#  Klassen (Auszug):  ex:Person, ex:Gleis, ex:Bahnsteig, ex:Zug, ex:Gefahr,
#                     ex:Videoüberwachung, ex:Sensor, ex:Alarmsystem, ex:Maßnahme
#  Objekt-Properties: ex:befindetSichIn, ex:erkennt, ex:stehtAuf, ex:löstAus,
#                     ex:überwacht, ex:beobachtet, ex:meldet, ex:führtZu
#  Daten-Properties : ex:hatKonfidenz (xsd:float), ex:hatZeitstempel (xsd:dateTime),
#                     ex:hatPosition (xsd:string), ex:hatBeschreibung (xsd:string)
# ============================================================================

EX = "ex:"  # einfacher Prefix (du kannst z.B. "http://example.org/rail#" verwenden)

class Severity(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class HazardLabel(str, Enum):
    PERSON_ON_TRACK = "PersonOnTrack"            # ex:Person befindetSichIn ex:Gleis
    NEAR_EDGE_TRAIN = "NearEdgeWithTrain"        # ex:Person stehtAuf ex:BahnsteigKante  ∧  ex:Zug in Szene
    FALLEN_PERSON = "FallenPersonNearTrack"      # ex:Person liegt/gestürzt nahe Gleis
    OBJECT_ON_TRACK = "ObjectOnTrack"            # ex:Objekt befindetSichIn ex:Gleis
    SMOKE_FIRE = "SmokeOrFire"                   # ex:Rauch/Feuer als Gefahr
    CROWD_OVERFLOW = "CrowdOverflowOnTrack"      # ex:Menschenmenge im Gleisbereich

@dataclass
class Observation:
    """Beobachtungen/Signale für eine Szene (alle Werte ∈ [0,1] sind Konfidenzen)."""
    # Kontext-Geometrie
    distance_to_edge_m: Optional[float] = None
    train_approaching: float = 0.0
    # Detektor-Konfidenzen
    on_track_person: float = 0.0
    fallen_person: float = 0.0
    object_on_track: float = 0.0
    smoke_or_fire: float = 0.0
    crowd_on_track: float = 0.0
    # Bias (Recall-Priorisierung)
    class_threshold_recall_bias: float = 0.35
    # Zusatzinfos
    notes: Dict[str, float] = field(default_factory=dict)

@dataclass
class HazardDecision:
    severity: Severity
    score_0_100: int
    labels: List[HazardLabel]
    explanations: List[str]
    fired_rules: List[str]

# --------------------------- REGELWERK ---------------------------------------

def _passes(p: float, thr: float) -> bool:
    return p >= thr

def evaluate(ob: Observation) -> HazardDecision:
    """Regelbasierte Bewertung mit erklärbarer Ausgabe (Ontologie-gedacht)."""
    thr = ob.class_threshold_recall_bias
    labels: List[HazardLabel] = []
    explains: List[str] = []
    fired: List[str] = []
    score_terms: List[Tuple[Severity, float]] = []

    # R1 — ex:Person ex:befindetSichIn ex:Gleis  → Kritisch
    if _passes(ob.on_track_person, thr):
        labels.append(HazardLabel.PERSON_ON_TRACK)
        fired.append("R1_befindetSichIn_Gleis")
        explains.append(f"R1: Person im Gleis erkannt (p={ob.on_track_person:.2f}).")
        score_terms.append((Severity.CRITICAL, 0.85 + 0.15 * ob.on_track_person))

    # R2 — Nahe Kante + Zug → Hoch
    if (ob.distance_to_edge_m is not None and ob.distance_to_edge_m <= 0.5) and _passes(ob.train_approaching, thr):
        labels.append(HazardLabel.NEAR_EDGE_TRAIN)
        fired.append("R2_stehtAuf_Bahnsteigkante_und_Zug")
        explains.append(
            f"R2: ≤0.5 m zur Kante (d={ob.distance_to_edge_m:.2f} m) & Zug (p={ob.train_approaching:.2f})."
        )
        score_terms.append((Severity.HIGH, 0.75 + 0.25 * ob.train_approaching))

    # R3 — Gestürzte Person nahe Kante/auf Gleis → Hoch/Kritisch
    if _passes(ob.fallen_person, thr):
        if (ob.distance_to_edge_m is not None and ob.distance_to_edge_m <= 1.0) or _passes(ob.on_track_person, thr):
            labels.append(HazardLabel.FALLEN_PERSON)
            fired.append("R3_fallenPerson_in_Gefahrenzone")
            explains.append(f"R3: Gestürzte Person (p={ob.fallen_person:.2f}).")
            sev = Severity.CRITICAL if _passes(ob.on_track_person, thr) else Severity.HIGH
            base = 0.80 if sev is Severity.CRITICAL else 0.70
            score_terms.append((sev, base + 0.20 * ob.fallen_person))

    # R4 — Objekt im Gleis → Mittel
    if _passes(ob.object_on_track, thr):
        labels.append(HazardLabel.OBJECT_ON_TRACK)
        fired.append("R4_Objekt_im_Gleis")
        explains.append(f"R4: Objekt im Gleis (p={ob.object_on_track:.2f}).")
        score_terms.append((Severity.MEDIUM, 0.60 + 0.30 * ob.object_on_track))

    # R5 — Rauch/Feuer → Hoch
    if _passes(ob.smoke_or_fire, thr):
        labels.append(HazardLabel.SMOKE_FIRE)
        fired.append("R5_Rauch_oder_Feuer")
        explains.append(f"R5: Rauch/Feuer (p={ob.smoke_or_fire:.2f}).")
        score_terms.append((Severity.HIGH, 0.70 + 0.25 * ob.smoke_or_fire))

    # R6 — Menschenmenge im Gleis → Kritisch
    if _passes(ob.crowd_on_track, thr):
        labels.append(HazardLabel.CROWD_OVERFLOW)
        fired.append("R6_Menschenmenge_im_Gleisbereich")
        explains.append(f"R6: Crowd im Gleis (p={ob.crowd_on_track:.2f}).")
        score_terms.append((Severity.CRITICAL, 0.80 + 0.20 * ob.crowd_on_track))

    if not score_terms:
        return HazardDecision(
            severity=Severity.NONE,
            score_0_100=0,
            labels=[],
            explanations=["Keine Gefahrenrelation erfüllt."],
            fired_rules=[]
        )

    sev_weights = {Severity.NONE:0.0, Severity.LOW:0.25, Severity.MEDIUM:0.55, Severity.HIGH:0.80, Severity.CRITICAL:1.0}
    best = max(score_terms, key=lambda t: sev_weights[t[0]] * t[1])
    best_sev, best_p = best
    final_score = int(round(100 * sev_weights[best_sev] * best_p))

    labels = list(dict.fromkeys(labels))
    return HazardDecision(best_sev, final_score, labels, explains, fired)

# --------------------------- RDF/TRIPLES -------------------------------------
@dataclass
class OntologyContext:
    """IDs/Metadaten für Tripel (du kannst echte IRIs verwenden)."""
    person_id: str = "person1"
    sensor_id: str = "sensor1"
    video_system_id: str = "videoSys1"
    track_id: str = "gleis1"
    platform_id: str = "bahnsteig1"
    alarm_id: str = "alarm1"
    measure_id: str = "massnahme1"
    event_id: str = "event1"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    position: Optional[str] = None                  # z.B. "x=123,y=45,cam=2"

def _lit(value: str, dtype: str) -> str:
    # Turtle-ähnlicher Literal-Renderer
    return f"\"{value}\"^^{dtype}"

def decision_to_triples(dec: HazardDecision, ob: Observation, ctx: OntologyContext) -> List[Tuple[str,str,str]]:
    """
    Erzeugt RDF-ähnliche Tripel basierend auf der Ontologie aus deiner Grafik.
    Nur stdlib; Ausgabe als einfache (s, p, o)-Tupel (Turtle-artig).
    """
    triples: List[Tuple[str,str,str]] = []

    # Typisierungen (rdf:type)
    triples += [
        (EX+ctx.person_id,       "rdf:type", EX+"Person"),
        (EX+ctx.sensor_id,       "rdf:type", EX+"Sensor"),
        (EX+ctx.video_system_id, "rdf:type", EX+"Videoüberwachung"),
        (EX+ctx.track_id,        "rdf:type", EX+"Gleis"),
        (EX+ctx.platform_id,     "rdf:type", EX+"Bahnsteig"),
        (EX+ctx.alarm_id,        "rdf:type", EX+"Alarmsystem"),
        (EX+ctx.measure_id,      "rdf:type", EX+"Maßnahme"),
        (EX+ctx.event_id,        "rdf:type", EX+"Ereignis"),
        (EX+"gef1",              "rdf:type", EX+"Gefahr"),
    ]

    # Überwachung/Beobachtungskette
    triples += [
        (EX+ctx.video_system_id, EX+"überwacht", EX+ctx.platform_id),
        (EX+ctx.sensor_id,       EX+"beobachtet", EX+ctx.platform_id),
        (EX+ctx.sensor_id,       EX+"erkennt",    EX+ctx.person_id),
    ]

    # Daten-Properties
    triples.append((EX+ctx.event_id, EX+"hatZeitstempel", _lit(ctx.timestamp.isoformat(), "xsd:dateTime")))
    if ctx.position:
        triples.append((EX+ctx.person_id, EX+"hatPosition", _lit(ctx.position, "xsd:string")))

    # Konfidenzen (nur wenn gesetzt)
    def add_conf(name: str, val: float):
        triples.append((EX+name, EX+"hatKonfidenz", _lit(f"{val:.3f}", "xsd:float")))

    if ob.on_track_person:  add_conf(ctx.person_id, ob.on_track_person)
    if ob.object_on_track:  triples.append((EX+"obj1", "rdf:type", EX+"Objekt")) or add_conf("obj1", ob.object_on_track)
    if ob.smoke_or_fire:    triples.append((EX+"smk1", "rdf:type", EX+"Unfall"))  or add_conf("smk1", ob.smoke_or_fire)

    # Ontologische Kernaussagen je nach Label
    for lab in dec.labels:
        if lab == HazardLabel.PERSON_ON_TRACK:
            triples.append((EX+ctx.person_id, EX+"befindetSichIn", EX+ctx.track_id))
        elif lab == HazardLabel.NEAR_EDGE_TRAIN:
            # approximiert: Person steht (nahe) auf Bahnsteigkante
            triples.append((EX+ctx.person_id, EX+"stehtAuf", EX+ctx.platform_id))
        elif lab == HazardLabel.OBJECT_ON_TRACK:
            triples.append((EX+"obj1", EX+"befindetSichIn", EX+ctx.track_id))
        elif lab == HazardLabel.CROWD_OVERFLOW:
            triples.append((EX+ctx.platform_id, EX+"istZugaenglich", _lit("false", "xsd:boolean")))

    # Gefahr → löstAus → Alarm; Alarm → führtZu → Maßnahme
    triples += [
        (EX+"gef1", EX+"hatBeschreibung", _lit(f"Severity={dec.severity.name}; Score={dec.score_0_100}", "xsd:string")),
        (EX+"gef1", EX+"löstAus",         EX+ctx.alarm_id),
        (EX+ctx.alarm_id, EX+"führtZu",   EX+ctx.measure_id),
        (EX+ctx.alarm_id, EX+"meldet",    EX+"Polizei"),   # optionaler Meldeweg
    ]

    return triples

def triples_to_turtle(triples: List[Tuple[str,str,str]]) -> str:
    """Kleine Pretty-Printer-Hilfe für Logs/Datei-Export."""
    lines = []
    for s,p,o in triples:
        if not o.startswith(EX) and not o.startswith("\""):
            # Literale sind schon getaggt; ansonsten als Ressourcen belassen
            o = o
        lines.append(f"{s} {p} {o} .")
    return "\n".join(lines)
