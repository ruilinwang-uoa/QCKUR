# MetroPT-3 exploration (auto-generated)

- rows **1,516,948**, span 2020-02-01 00:00:00 .. 2020-09-01 03:59:50
- median cadence 10.0s, gaps>10min: 230 (~862.9h missing), dup ts: 0
- healthy supply: **3,785.3 h** >=72h from any event
- missing signals: analog=[] digital=['DV_electric']

## Event windows (72h pre .. maintenance+24h)

| id | start | rows | coverage | gaps |
|---|---|---|---|---|
| 1 | 2020-04-18 00:00 | 39,973 | 92% | 5 |
| 2 | 2020-05-29 23:30 | 34,760 | 89% | 6 |
| 3 | 2020-06-05 10:00 | 49,217 | 79% | 7 |
| 4 | 2020-07-15 14:30 | 30,296 | 80% | 3 |

## Signature check (event means vs healthy mean)

| signal | healthy | ev1 | ev2 | ev3 | ev4 |
|---|---|---|---|---|---|
| TP2 | 1.244 (±3.148) | 8.518 | 4.796 | 7.584 | 4.894 |
| TP3 | 9.001 (±0.63) | 8.812 | 8.687 | 8.023 | 8.652 |
| H1 | 7.712 (±3.199) | 0.079 | 3.793 | 0.316 | 3.897 |
| DV_pressure | 0.02 (±0.279) | 1.879 | 1.015 | 1.927 | -0.002 |
| Reservoirs | 9.002 (±0.63) | 8.813 | 8.689 | 8.025 | 8.651 |
| Motor_current | 1.985 (±2.274) | 5.604 | 3.828 | 5.364 | 4.238 |
| Oil_temperature | 62.228 (±6.277) | 74.131 | 70.725 | 74.986 | 77.12 |