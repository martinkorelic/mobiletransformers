SELECT 
  CAST(c.ts/1e9 AS INT) as timestamp_sec,
  c.value/1024/1024 as ram_usage_mb
FROM counter as c
LEFT JOIN process_counter_track as t ON c.track_id = t.id
LEFT JOIN process as p USING (upid)
WHERE p.name = 'com.martinkorelic.ortmobile'
AND t.name = 'mem.rss'
ORDER BY timestamp_sec;