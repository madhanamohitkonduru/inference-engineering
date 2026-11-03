SELECT config, workload, concurrency,
       count(*) AS n,
       round(avg(tpot_ms), 3) AS mean,
       round(quantile_cont(tpot_ms, 0.50), 3) AS p50,
       round(quantile_cont(tpot_ms, 0.95), 3) AS p95,
       round(quantile_cont(tpot_ms, 0.99), 3) AS p99
FROM 'day-10-goodput-analysis/day9_requests.parquet'
GROUP BY config, workload, concurrency
ORDER BY config, workload, concurrency;