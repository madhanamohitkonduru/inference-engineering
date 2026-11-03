SELECT concurrency,
       count(*) AS n,
       round(100.0 * avg(CASE WHEN ttft_ms <= 200 AND tpot_ms <= 10 THEN 1 ELSE 0 END), 2) AS pct_meeting_slo,
       round(quantile_cont(ttft_ms, 0.99), 2) AS ttft_p99,
       round(quantile_cont(tpot_ms, 0.99), 2) AS tpot_p99
FROM 'day-10-goodput-analysis/sweep_requests.parquet'
WHERE cache = 'sweep_cacheoff'
GROUP BY concurrency
ORDER BY concurrency;