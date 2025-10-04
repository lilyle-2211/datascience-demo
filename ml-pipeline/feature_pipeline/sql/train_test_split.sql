create or replace table `lily.test.train_test_split` as
WITH stratified AS (
  SELECT
    *,
    CASE WHEN revenue_day1_20 > 0 THEN 'positive' ELSE 'zero' END AS target_group,
    FARM_FINGERPRINT(CAST(user_id AS STRING)) AS hash_val
  FROM
    `lily.test.data_preprocessing`
),
split AS (
  SELECT
    *,
    -- 80% train, 20% test split within each stratum
    CASE
      WHEN MOD(ABS(hash_val), 10) < 8 THEN 'train'
      ELSE 'test'
    END AS split_set
  FROM stratified
)
SELECT *
FROM split
