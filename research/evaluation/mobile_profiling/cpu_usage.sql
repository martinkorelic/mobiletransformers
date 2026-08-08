-- This query retrieves the CPU usage over time for the specified mobile application.

SELECT 
  CAST(ts/1e9 AS INT) as timestamp_sec,
  cpu,
  SUM(dur)/1e6 as cpu_time_ms,
  ROUND((SUM(dur)/1e6 / 1000.0) * 100, 2) as cpu_usage_percent
FROM sched_slice
LEFT JOIN thread USING (utid)  
LEFT JOIN process USING (upid)
WHERE process.name = 'com.martinkorelic.mobiletransformers.app'
GROUP BY timestamp_sec, cpu
ORDER BY timestamp_sec, cpu;