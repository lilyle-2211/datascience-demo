SELECT
    date,
    user_id,
    SUM(levels_played) as levels_played,
    SUM(levels_completed) as levels_completed,
    SUM(coins_used) as sum_coins_used,
    MAX(max_level_completed) as max_level_completed
  FROM `lilyle-demo.data_source.activity`
   where date >= '2022-06-06'
  GROUP BY 1, 2
 