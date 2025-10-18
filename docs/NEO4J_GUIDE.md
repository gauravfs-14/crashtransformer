# 🧱 Neo4j Integration Guide

Enable Neo4j to store and query the extracted crash graphs and summaries for advanced analytics and pattern discovery.

## 🔌 Enable Neo4j Integration

### **Environment Configuration**

Set the following in your `.env` file:

```bash
ENABLE_NEO4J=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
```

### **Command Line Usage**

```bash
# Enable Neo4j via CLI
python crashtransformer.py run --csv crashes.csv --neo4j_enabled \
  --neo4j_uri bolt://localhost:7687

# With custom credentials
python crashtransformer.py run --csv crashes.csv --neo4j_enabled \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_password your-password
```

## 🏗️ Database Schema

### **Node Types**

| Node Type | Description | Key Properties |
|-----------|-------------|----------------|
| `Crash` | Crash incident record | `crash_id`, `latitude`, `longitude`, `city`, `crash_severity` |
| `Vehicle` | Vehicle involved in crash | `id`, `unit_id`, `mention_text`, `confidence` |
| `Location` | Geographic location | `id`, `name`, `road`, `block`, `city` |
| `Event` | Crash-related event | `id`, `type`, `label`, `attributes`, `evidence_span` |
| `Summary` | Generated summary | `crash_id`, `best_summary`, `plan_lines`, `metrics` |

### **Relationship Types**

| Relationship | Description | Example |
|--------------|-------------|---------|
| `HAS_ENTITY` | Crash → Vehicle/Location | `(Crash)-[:HAS_ENTITY]->(Vehicle)` |
| `HAS_EVENT` | Crash → Event | `(Crash)-[:HAS_EVENT]->(Event)` |
| `CAUSES` | Event → Event | `(Violation)-[:CAUSES]->(Collision)` |
| `PARTICIPATED_IN` | Vehicle → Event | `(Vehicle)-[:PARTICIPATED_IN]->(Event)` |
| `HIT` | Vehicle → Vehicle | `(Vehicle1)-[:HIT]->(Vehicle2)` |
| `HAS_SUMMARY` | Crash → Summary | `(Crash)-[:HAS_SUMMARY]->(Summary)` |

### **Constraints and Indexes**

The system automatically creates:

```cypher
CREATE CONSTRAINT crash_id IF NOT EXISTS FOR (c:Crash) REQUIRE c.crash_id IS UNIQUE;
CREATE CONSTRAINT vehicle_id IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT summary_pk IF NOT EXISTS FOR (s:Summary) REQUIRE s.crash_id IS UNIQUE;
CREATE INDEX crash_city IF NOT EXISTS FOR (c:Crash) ON (c.city);
CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.type);
```

## 🔍 Comprehensive Cypher Queries

### **1. Basic Crash Analysis**

#### **Get All Crashes with Summaries**

```cypher
MATCH (c:Crash)-[:HAS_SUMMARY]->(s:Summary)
RETURN c.crash_id, c.city, c.crash_severity, s.best_summary
ORDER BY c.crash_id
LIMIT 10;
```

#### **Find Crashes by Location**

```cypher
MATCH (c:Crash {city: "Houston"})
RETURN c.crash_id, c.latitude, c.longitude, c.crash_severity
ORDER BY c.crash_id;
```

#### **Filter by Severity and Date Range**

```cypher
MATCH (c:Crash)
WHERE c.crash_severity = "Injured" 
  AND c.crash_date >= "2024-01-01"
RETURN c.crash_id, c.city, c.crash_date, c.crash_time
ORDER BY c.crash_date DESC;
```

### **2. Causal Chain Analysis**

#### **Most Common Causal Chains**

```cypher
MATCH (e1:Event)-[:CAUSES]->(e2:Event)
RETURN e1.type AS cause_type, e2.type AS effect_type, 
       count(*) AS frequency
ORDER BY frequency DESC
LIMIT 20;
```

#### **Detailed Causal Chain with Evidence**

```cypher
MATCH (e1:Event)-[c:CAUSES]->(e2:Event)
WHERE c.marked = true
RETURN e1.label AS cause, e2.label AS effect, 
       e1.evidence_span AS cause_evidence,
       e2.evidence_span AS effect_evidence,
       c.connective AS connective
ORDER BY e1.id, e2.id;
```

#### **Multi-Step Causal Chains**

```cypher
MATCH path = (e1:Event)-[:CAUSES*2..4]->(e2:Event)
RETURN [node IN nodes(path) | node.type] AS causal_chain,
       length(path) AS chain_length
ORDER BY chain_length DESC
LIMIT 10;
```

### **3. Vehicle and Event Analysis**

#### **Vehicle Involvement Patterns**

```cypher
MATCH (v:Vehicle)-[:PARTICIPATED_IN]->(e:Event)
RETURN v.unit_id, e.type, count(*) AS involvement_count
ORDER BY involvement_count DESC;
```

#### **Event Types by Severity**

```cypher
MATCH (c:Crash)-[:HAS_EVENT]->(e:Event)
WHERE c.crash_severity IS NOT NULL
RETURN c.crash_severity, e.type, count(*) AS event_count
ORDER BY c.crash_severity, event_count DESC;
```

#### **Violation to Collision Analysis**

```cypher
MATCH (violation:Event {type: "VIOLATION"})-[:CAUSES]->(collision:Event {type: "COLLISION"})
MATCH (c:Crash)-[:HAS_EVENT]->(violation)
RETURN c.crash_severity, violation.label AS violation_type, 
       collision.label AS collision_type, count(*) AS frequency
ORDER BY frequency DESC;
```

### **4. Geographic Analysis**

#### **Crash Density by City**

```cypher
MATCH (c:Crash)
WHERE c.city IS NOT NULL
RETURN c.city, count(*) AS crash_count,
       collect(DISTINCT c.crash_severity) AS severities
ORDER BY crash_count DESC;
```

#### **Geographic Clustering**

```cypher
MATCH (c:Crash)
WHERE c.latitude IS NOT NULL AND c.longitude IS NOT NULL
RETURN c.crash_id, c.latitude, c.longitude, c.city, c.crash_severity
ORDER BY c.latitude, c.longitude;
```

#### **Road-Specific Analysis**

```cypher
MATCH (c:Crash)-[:HAS_ENTITY]->(l:Location)
WHERE l.road IS NOT NULL
RETURN l.road, l.city, count(*) AS crash_count
ORDER BY crash_count DESC
LIMIT 20;
```

### **5. Temporal Analysis**

#### **Crash Patterns by Day of Week**

```cypher
MATCH (c:Crash)
WHERE c.day_of_week IS NOT NULL
RETURN c.day_of_week, count(*) AS crash_count,
       collect(DISTINCT c.crash_severity) AS severities
ORDER BY 
  CASE c.day_of_week 
    WHEN 'MON' THEN 1 WHEN 'TUE' THEN 2 WHEN 'WED' THEN 3 
    WHEN 'THU' THEN 4 WHEN 'FRI' THEN 5 WHEN 'SAT' THEN 6 
    WHEN 'SUN' THEN 7 END;
```

#### **Time-Based Crash Analysis**

```cypher
MATCH (c:Crash)
WHERE c.crash_time IS NOT NULL
WITH c, 
     CASE 
       WHEN c.crash_time CONTAINS 'AM' THEN 'Morning'
       WHEN c.crash_time CONTAINS 'PM' THEN 'Evening'
       ELSE 'Unknown'
     END AS time_period
RETURN time_period, count(*) AS crash_count,
       collect(DISTINCT c.crash_severity) AS severities
ORDER BY crash_count DESC;
```

### **6. Summary Quality Analysis**

#### **Summary Metrics Analysis**

```cypher
MATCH (c:Crash)-[:HAS_SUMMARY]->(s:Summary)
WHERE s.metrics IS NOT NULL
RETURN c.crash_id, s.best_summary,
       s.metrics AS quality_metrics
ORDER BY c.crash_id
LIMIT 10;
```

#### **High-Quality Summaries**

```cypher
MATCH (c:Crash)-[:HAS_SUMMARY]->(s:Summary)
WHERE s.metrics CONTAINS '"combined_score":'
  AND s.metrics CONTAINS '0.8'
RETURN c.crash_id, s.best_summary, s.metrics
ORDER BY c.crash_id;
```

### **7. Advanced Pattern Discovery**

#### **Complex Causal Networks**

```cypher
MATCH (c:Crash)-[:HAS_EVENT]->(e1:Event)-[:CAUSES]->(e2:Event)-[:CAUSES]->(e3:Event)
RETURN c.crash_id, c.city, c.crash_severity,
       e1.label AS first_cause, e2.label AS intermediate, e3.label AS final_effect
ORDER BY c.crash_id;
```

#### **Multi-Vehicle Crash Analysis**

```cypher
MATCH (c:Crash)-[:HAS_ENTITY]->(v1:Vehicle)-[:HIT]->(v2:Vehicle)
MATCH (c)-[:HAS_EVENT]->(e:Event)
RETURN c.crash_id, v1.unit_id AS vehicle1, v2.unit_id AS vehicle2,
       collect(DISTINCT e.type) AS event_types
ORDER BY c.crash_id;
```

#### **Autonomous Vehicle Analysis**

```cypher
MATCH (c:Crash)
WHERE c.sae_autonomy_level IS NOT NULL
RETURN c.sae_autonomy_level, count(*) AS crash_count,
       collect(DISTINCT c.crash_severity) AS severities
ORDER BY c.sae_autonomy_level;
```

### **8. Performance and Cost Analysis**

#### **Processing Time Analysis**

```cypher
MATCH (c:Crash)-[:HAS_SUMMARY]->(s:Summary)
WHERE s.extract_runtime_sec IS NOT NULL
RETURN c.crash_id, s.extract_runtime_sec, s.summarizer_runtime_sec,
       s.extract_total_tokens, s.summarizer_input_tokens
ORDER BY s.extract_runtime_sec DESC
LIMIT 20;
```

#### **Token Usage Patterns**

```cypher
MATCH (c:Crash)-[:HAS_SUMMARY]->(s:Summary)
WHERE s.extract_total_tokens IS NOT NULL
RETURN avg(s.extract_total_tokens) AS avg_extract_tokens,
       avg(s.summarizer_input_tokens) AS avg_summarizer_tokens,
       count(*) AS total_crashes;
```

### **9. Data Quality and Validation**

#### **Missing Data Analysis**

```cypher
MATCH (c:Crash)
RETURN 
  count(*) AS total_crashes,
  count(c.latitude) AS with_latitude,
  count(c.longitude) AS with_longitude,
  count(c.city) AS with_city,
  count(c.crash_severity) AS with_severity;
```

#### **Event Coverage Analysis**

```cypher
MATCH (c:Crash)
OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:Event)
RETURN c.crash_id, count(e) AS event_count
ORDER BY event_count ASC
LIMIT 20;
```

### **10. Export and Reporting Queries**

#### **Export for External Analysis**

```cypher
MATCH (c:Crash)-[:HAS_SUMMARY]->(s:Summary)
OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:Event)
OPTIONAL MATCH (c)-[:HAS_ENTITY]->(v:Vehicle)
RETURN c.crash_id, c.city, c.crash_severity, s.best_summary,
       collect(DISTINCT e.type) AS event_types,
       collect(DISTINCT v.unit_id) AS vehicles
ORDER BY c.crash_id;
```

#### **Summary Statistics**

```cypher
MATCH (c:Crash)
RETURN 
  count(*) AS total_crashes,
  count(DISTINCT c.city) AS unique_cities,
  count(DISTINCT c.crash_severity) AS severity_levels,
  collect(DISTINCT c.day_of_week) AS days_of_week;
```

## 🛠️ Best Practices

### **Query Optimization**

1. **Use Indexes**: Leverage the automatically created indexes on `city` and `event_type`
2. **Limit Results**: Always use `LIMIT` for exploratory queries
3. **Use Parameters**: For repeated queries, use parameters instead of hardcoded values

```cypher
// Good: Parameterized query
MATCH (c:Crash {city: $city})
WHERE c.crash_severity = $severity
RETURN c.crash_id, c.latitude, c.longitude;

// Usage: 
// :param city => "Houston"
// :param severity => "Injured"
```

### **Performance Tips**

1. **Batch Operations**: The pipeline handles batch inserts efficiently
2. **Avoid Concurrent Writes**: Don't run multiple pipelines to the same database simultaneously
3. **Monitor Memory**: Large result sets may require pagination

### **Security Considerations**

1. **Network Access**: Ensure firewall rules allow Bolt connections (port 7687)
2. **Authentication**: Use strong passwords and consider SSL/TLS for remote connections
3. **Access Control**: Implement proper user roles and permissions

## 🔧 Troubleshooting

### **Common Issues**

1. **Connection Refused**

   ```bash
   # Check if Neo4j is running
   docker ps | grep neo4j
   # or
   systemctl status neo4j
   ```

2. **Authentication Failed**

   ```bash
   # Reset password
   docker exec -it neo4j-container cypher-shell -u neo4j -p neo4j
   # Then: ALTER USER neo4j SET PASSWORD 'new-password';
   ```

3. **Memory Issues**

   ```bash
   # Increase heap size in neo4j.conf
   dbms.memory.heap.initial_size=2G
   dbms.memory.heap.max_size=4G
   ```

### **Monitoring Queries**

```cypher
// Check database size
CALL apoc.meta.stats() YIELD nodeCount, relCount;

// Check constraint status
SHOW CONSTRAINTS;

// Check indexes
SHOW INDEXES;
```

## 📊 Visualization Examples

### **Causal Chain Visualization**

```cypher
MATCH path = (e1:Event)-[:CAUSES*1..3]->(e2:Event)
RETURN path
LIMIT 50;
```

### **Geographic Distribution**

```cypher
MATCH (c:Crash)
WHERE c.latitude IS NOT NULL AND c.longitude IS NOT NULL
RETURN c.latitude, c.longitude, c.crash_severity, c.city
ORDER BY c.city;
```

This comprehensive guide provides everything you need to effectively use Neo4j with CrashTransformer for advanced crash data analysis and pattern discovery.

---

## 📖 Navigation

**Previous:** [Providers Guide](PROVIDERS_GUIDE.md) - LLM provider setup  
**Next:** [Outputs](OUTPUTS.md) - Output structure and artifacts
