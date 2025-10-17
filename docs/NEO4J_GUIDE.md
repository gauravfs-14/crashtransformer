# 🧱 Neo4j Integration Guide

Enable Neo4j to store and query the extracted crash graphs and summaries.

## 🔌 Enable

- In `.env` set `ENABLE_NEO4J=true` and provide credentials, or
- Pass `--neo4j_enabled` to the `run` command

```bash
python crashtransformer.py run --csv crashes.csv --neo4j_enabled \
  --neo4j_uri bolt://localhost:7687
```

## 🔐 Credentials

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
```

## 🧩 Nodes and relationships

- Nodes: `Crash`, `Vehicle`, `Location`, `Event`, `Summary`
- Relationships: `HAS_ENTITY`, `HAS_EVENT`, `CAUSES`, `PARTICIPATED_IN`, `HIT`, `HAS_SUMMARY`

Constraints are created for uniqueness where applicable (e.g., `Crash.crash_id`, `Event.id`).

## 🧪 Cypher examples

Top 10 causal chains:

```cypher
MATCH (e1:Event)-[:CAUSES]->(e2:Event)
RETURN e1.type AS cause, e2.type AS effect, count(*) AS n
ORDER BY n DESC LIMIT 10;
```

Best summaries attached to crashes:

```cypher
MATCH (c:Crash)-[:HAS_SUMMARY]->(s:Summary)
RETURN c.crash_id AS crash_id, s.best_summary AS summary
LIMIT 20;
```

Filter by city and severity:

```cypher
MATCH (c:Crash {city: $city})-[:HAS_EVENT]->(e:Event)
WHERE c.crash_severity = $severity
RETURN c.crash_id, e.type, e.attributes
```

## 🛠️ Tips

- Use indexes/constraints for frequent lookup properties.
- Batch inserts are handled by the pipeline; avoid running multiple pipelines concurrently to the same DB without coordination.
- For remote DBs, ensure networking/firewall rules allow Bolt connections.
