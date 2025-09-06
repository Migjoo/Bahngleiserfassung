# test_ontology_triples.py
from ontology_eval import Observation, evaluate, OntologyContext, decision_to_triples, triples_to_turtle

if __name__ == "__main__":
    # Szenario: Person im Gleis + Zug naht -> kritisch
    ob = Observation(
        on_track_person=0.88,
        train_approaching=0.9,
        distance_to_edge_m=0.3
    )

    dec = evaluate(ob)
    print(f"SEVERITY: {dec.severity.name}  SCORE: {dec.score_0_100}")
    for e in dec.explanations:
        print(" -", e)

    ctx = OntologyContext(
        person_id="person42",
        sensor_id="cam02",
        video_system_id="videosysA",
        track_id="gleis_3",
        platform_id="bahnsteig_3",
        alarm_id="alarm_4711",
        measure_id="massnahme_stop",
        event_id="event_2023_001",
        position="x=123.4,y=56.7,cam=2"
    )

    triples = decision_to_triples(dec, ob, ctx)
    print("\n--- Turtle ---")
    print(triples_to_turtle(triples))
