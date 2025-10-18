# neo4j_io.py

import json
from typing import Dict, Any, Iterable, Optional
from neo4j import GraphDatabase

class Neo4jSink:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def ensure_constraints(self):
        cyphers = [
            "CREATE CONSTRAINT crash_id IF NOT EXISTS FOR (c:Crash) REQUIRE c.crash_id IS UNIQUE",
            "CREATE CONSTRAINT vehicle_id IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
            "CREATE CONSTRAINT summary_pk IF NOT EXISTS FOR (s:Summary) REQUIRE s.crash_id IS UNIQUE",
            "CREATE INDEX crash_city IF NOT EXISTS FOR (c:Crash) ON (c.city)",
            "CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.type)",
        ]
        with self._driver.session() as s:
            for q in cyphers:
                s.run(q)

    @staticmethod
    def _is_location(entity: Dict[str, Any]) -> bool:
        lbl = (entity.get("label") or "").lower()
        return "loc" in lbl or lbl == "location" or lbl == "place"

    def upsert_crash_graph(self, crash_graph: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(self._upsert_crash_graph_tx, crash_graph)

    @staticmethod
    def _upsert_crash_graph_tx(tx, crash_graph: Dict[str, Any]):
        crash = crash_graph.get("crash", {})
        entities = crash_graph.get("entities", []) or []
        events = crash_graph.get("events", []) or []
        rels = crash_graph.get("relationships", []) or []

        # Crash
        tx.run(
            """
            MERGE (c:Crash {crash_id: $crash_id})
            SET c.latitude = $latitude,
                c.longitude = $longitude,
                c.crash_date = $crash_date,
                c.day_of_week = $day_of_week,
                c.crash_time = $crash_time,
                c.county = $county,
                c.city = $city,
                c.sae_autonomy_level = $sae_autonomy_level,
                c.crash_severity = $crash_severity
            """,
            crash_id=str(crash.get("crash_id")),
            latitude=crash.get("latitude"),
            longitude=crash.get("longitude"),
            crash_date=crash.get("crash_date"),
            day_of_week=crash.get("day_of_week"),
            crash_time=crash.get("crash_time"),
            county=crash.get("county"),
            city=crash.get("city"),
            sae_autonomy_level=crash.get("sae_autonomy_level"),
            crash_severity=crash.get("crash_severity"),
        )

        # Entities
        for e in entities:
            eid = e.get("id")
            if not eid:
                continue
            if Neo4jSink._is_location(e):
                tx.run(
                    """
                    MERGE (n:Location {id: $id})
                    SET n.name = coalesce($name, n.name),
                        n.road = coalesce($road, n.road),
                        n.block = coalesce($block, n.block),
                        n.city = coalesce($city, n.city)
                    WITH n
                    MATCH (c:Crash {crash_id: $crash_id})
                    MERGE (c)-[:HAS_ENTITY]->(n)
                    """,
                    id=eid, name=e.get("name"), road=e.get("road"), block=e.get("block"),
                    city=e.get("city"), crash_id=str(crash.get("crash_id")),
                )
            else:
                tx.run(
                    """
                    MERGE (n:Vehicle {id: $id})
                    SET n.unit_id = coalesce($unit_id, n.unit_id),
                        n.mention_text = coalesce($mention_text, n.mention_text),
                        n.confidence = coalesce($confidence, n.confidence)
                    WITH n
                    MATCH (c:Crash {crash_id: $crash_id})
                    MERGE (c)-[:HAS_ENTITY]->(n)
                    """,
                    id=eid, unit_id=e.get("unit_id"),
                    mention_text=e.get("mention_text"), confidence=e.get("confidence"),
                    crash_id=str(crash.get("crash_id")),
                )

        # Events
        for ev in events:
            evid = ev.get("id")
            if not evid:
                continue
            # Serialize attributes to JSON string to avoid Map{} error
            attributes = ev.get("attributes") or {}
            serialized_attributes = json.dumps(attributes) if attributes else None
            
            tx.run(
                """
                MERGE (e:Event {id: $id})
                SET e.type = coalesce($type, e.type),
                    e.label = coalesce($label, e.label),
                    e.attributes = coalesce($attributes, e.attributes),
                    e.confidence = coalesce($confidence, e.confidence),
                    e.evidence_span = coalesce($evidence_span, e.evidence_span)
                WITH e
                MATCH (c:Crash {crash_id: $crash_id})
                MERGE (c)-[:HAS_EVENT]->(e)
                """,
                id=evid, type=ev.get("type"), label=ev.get("label"),
                attributes=serialized_attributes, confidence=ev.get("confidence"),
                evidence_span=ev.get("evidence_span"),
                crash_id=str(crash.get("crash_id")),
            )

        # Relationships
        for r in rels:
            start = r.get("start"); end = r.get("end"); rtype = r.get("type")
            props = r.get("properties") or {}
            if not start or not end or not rtype:
                continue

            if rtype == "PARTICIPATED_IN":
                tx.run(
                    """
                    MATCH (a {id: $start})
                    MATCH (b:Event {id: $end})
                    MERGE (a)-[rel:PARTICIPATED_IN]->(b)
                    SET rel.role = coalesce($role, rel.role),
                        rel.unit_id = coalesce($unit_id, rel.unit_id)
                    """,
                    start=start, end=end,
                    role=props.get("role"), unit_id=props.get("unit_id"),
                )
            elif rtype == "CAUSES":
                tx.run(
                    """
                    MATCH (a:Event {id: $start})
                    MATCH (b:Event {id: $end})
                    MERGE (a)-[rel:CAUSES]->(b)
                    SET rel.marked = coalesce($marked, rel.marked),
                        rel.connective = coalesce($connective, rel.connective),
                        rel.evidence_span = coalesce($evidence_span, rel.evidence_span)
                    """,
                    start=start, end=end,
                    marked=props.get("marked"), connective=props.get("connective"),
                    evidence_span=props.get("evidence_span"),
                )
            elif rtype == "HIT":
                tx.run(
                    """
                    MATCH (a:Vehicle {id: $start})
                    MATCH (b:Vehicle {id: $end})
                    MERGE (a)-[rel:HIT]->(b)
                    SET rel.impact_config = coalesce($impact_config, rel.impact_config)
                    """,
                    start=start, end=end, impact_config=props.get("impact_config"),
                )
            else:
                # Serialize complex properties as JSON strings to avoid Map{} error
                serialized_props = {k: json.dumps(v) if isinstance(v, (dict, list)) else v 
                                  for k, v in props.items()}
                tx.run(
                    """
                    MATCH (a {id: $start})
                    MATCH (b {id: $end})
                    MERGE (a)-[rel:`%s`]->(b)
                    SET rel += $props
                    """ % rtype,
                    start=start, end=end, props=serialized_props
                )

    def upsert_summary(self, summary_record: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(self._upsert_summary_tx, summary_record)

    @staticmethod
    def _upsert_summary_tx(tx, record: Dict[str, Any]):
        cid = record["Crash_ID"]
        
        # Serialize metrics to JSON string to avoid Map{} error
        metrics = record.get("metrics") or {}
        serialized_metrics = json.dumps(metrics) if metrics else None
        
        tx.run(
            """
            MERGE (s:Summary {crash_id: $cid})
            SET s.best_summary = $best_summary,
                s.plan_lines = $plan_lines,
                s.metrics = $metrics,
                s.extract_runtime_sec = $extract_runtime_sec,
                s.extract_total_tokens = $extract_total_tokens,
                s.summarizer_runtime_sec = $summarizer_runtime_sec,
                s.summarizer_input_tokens = $summarizer_input_tokens,
                s.summarizer_output_tokens = $summarizer_output_tokens
            WITH s
            MERGE (c:Crash {crash_id: $cid})
            MERGE (c)-[:HAS_SUMMARY]->(s)
            """,
            cid=str(cid),
            best_summary=record.get("best_summary"),
            plan_lines=record.get("plan_lines"),
            metrics=serialized_metrics,
            extract_runtime_sec=record.get("extract_runtime_sec"),
            extract_total_tokens=record.get("extract_total_tokens"),
            summarizer_runtime_sec=record.get("summarizer_runtime_sec"),
            summarizer_input_tokens=record.get("summarizer_input_tokens"),
            summarizer_output_tokens=record.get("summarizer_output_tokens"),
        )
