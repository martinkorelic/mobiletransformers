WITH process_activity AS (
 SELECT DISTINCT CAST(ts/1e9 AS INT) as timestamp_sec
 FROM sched_slice
 LEFT JOIN thread USING (utid)  
 LEFT JOIN process USING (upid)
 WHERE process.name = 'com.martinkorelic.ortmobile'
)
SELECT 
 CAST(c.ts/1e9 AS INT) as timestamp_sec,
 t.name as thermal_zone,
 AVG(c.value) / 1000.0 as temperature_celsius
FROM counter c
JOIN track t ON c.track_id = t.id
JOIN process_activity pa ON CAST(c.ts/1e9 AS INT) = pa.timestamp_sec
WHERE t.name LIKE '%BIG%Temperature%' 
  OR t.name LIKE '%MID%Temperature%' 
  OR t.name LIKE '%LITTLE%Temperature%'
GROUP BY timestamp_sec, t.name
ORDER BY timestamp_sec, t.name;