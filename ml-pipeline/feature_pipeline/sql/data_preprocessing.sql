create or replace table `lily.test.data_preprocessing` as

  -- User installs CTE
WITH user_installs AS (
    SELECT
      user_id,
      DATE(install_date) as install_date,
      is_age_30,
      is_android,
      is_female
    FROM `lily.test.users_view`
    WHERE install_date IS NOT NULL
  ),

  -- First purchase day calculation
  first_purchases AS (
    SELECT
      r.user_id,
      MIN(DATE_DIFF(DATE(r.date), ui.install_date, DAY)) as first_purchase_day
    FROM `lily.test.revenue_view` r
    JOIN user_installs ui ON r.user_id = ui.user_id
    WHERE DATE(r.date) >= ui.install_date
      AND DATE_DIFF(DATE(r.date), ui.install_date, DAY) <= 20
      AND r.transaction_value IS NOT NULL
      AND r.transaction_value > 0
    GROUP BY r.user_id
  ),

  -- Activity metrics by day ranges
  activity_metrics AS (
    SELECT
      ui.user_id,  -- Changed from a.user_id
      ui.install_date,
      -- Day 1 metrics
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) = 0 THEN a.levels_played ELSE 0 END) as sum_cumulative_levels_day1_1,
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) = 0 THEN a.levels_completed ELSE 0 END) as completed_levels_day1_1,
      MAX(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) = 0 THEN a.max_level_completed ELSE 0 END) as max_level_reach_day1_1,

      -- Day 1-3 metrics
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 2 THEN a.levels_played ELSE 0 END) as sum_cumulative_levels_day1_3,
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 2 THEN a.levels_completed ELSE 0 END) as completed_levels_day1_3,
      MAX(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 2 THEN a.max_level_completed ELSE 0 END) as max_level_reach_day1_3,

      -- Day 1-7 metrics
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 6 THEN a.levels_played ELSE 0 END) as sum_cumulative_levels_day1_7,
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 6 THEN a.levels_completed ELSE 0 END) as completed_levels_day1_7,
      MAX(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 6 THEN a.max_level_completed ELSE 0 END) as max_level_reach_day1_7,

      -- Day 1-14 metrics (keeping for feature engineering)
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 13 THEN a.levels_played ELSE 0 END) as sum_cumulative_levels_day1_14,
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 13 THEN a.levels_completed ELSE 0 END) as completed_levels_day1_14,
      MAX(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 13 THEN a.max_level_completed ELSE 0 END) as max_level_reach_day1_14,

      -- Day 1-20 metrics (new target feature window)
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 19 THEN a.levels_played ELSE 0 END) as sum_cumulative_levels_day1_20,
      SUM(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 19 THEN a.levels_completed ELSE 0 END) as completed_levels_day1_20,
      MAX(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 19 THEN a.max_level_completed ELSE 0 END) as max_level_reach_day1_20,

      -- Return next day calculation
      CASE
        WHEN COUNT(CASE WHEN DATE_DIFF(a.date, ui.install_date, DAY) = 1 THEN 1 END) > 0 THEN 1
        ELSE 0
      END as is_return_next_day

    FROM user_installs ui
    LEFT JOIN `lily.test.activity_view` a
      ON ui.user_id = a.user_id
      AND DATE_DIFF(a.date, ui.install_date, DAY) BETWEEN 0 AND 19
    GROUP BY ui.user_id, ui.install_date  -- Changed from a.user_id
  ),

  -- Revenue metrics
  revenue_metrics AS (
    SELECT
      ui.user_id,  -- Changed from r.user_id
      -- Day 1 revenue
      SUM(CASE WHEN DATE_DIFF(DATE(r.date), ui.install_date, DAY) = 0 THEN r.transaction_value ELSE 0 END) as revenue_day1_1,

      -- Day 1-3 revenue
      SUM(CASE WHEN DATE_DIFF(DATE(r.date), ui.install_date, DAY) BETWEEN 0 AND 2 THEN r.transaction_value ELSE 0 END) as revenue_day1_3,

      -- Day 1-7 revenue
      SUM(CASE WHEN DATE_DIFF(DATE(r.date), ui.install_date, DAY) BETWEEN 0 AND 6 THEN r.transaction_value ELSE 0 END) as revenue_day1_7,

      -- Day 1-14 revenue (keeping for feature)
      SUM(CASE WHEN DATE_DIFF(DATE(r.date), ui.install_date, DAY) BETWEEN 0 AND 13 THEN r.transaction_value ELSE 0 END) as revenue_day1_14,

      -- Day 1-20 revenue (TARGET)
      SUM(CASE WHEN DATE_DIFF(DATE(r.date), ui.install_date, DAY) BETWEEN 0 AND 19 THEN r.transaction_value ELSE 0 END) as revenue_day1_20

    FROM user_installs ui
    LEFT JOIN `lily.test.revenue_view` r
      ON ui.user_id = r.user_id
      AND DATE_DIFF(DATE(r.date), ui.install_date, DAY) BETWEEN 0 AND 19
      AND r.transaction_value IS NOT NULL
      AND r.transaction_value > 0
    GROUP BY ui.user_id  -- Changed from r.user_id
  )

  -- Final result
  SELECT
    ui.user_id,
    COALESCE(fp.first_purchase_day, NULL) as first_purchase_day,
    DATE_DIFF(DATE('2022-06-27'), ui.install_date, DAY) as days_since_install,
    ui.install_date,

    -- Levels played metrics
    COALESCE(am.sum_cumulative_levels_day1_1, 0) as sum_cumulative_levels_day1_1,
    COALESCE(am.sum_cumulative_levels_day1_3, 0) as sum_cumulative_levels_day1_3,
    COALESCE(am.sum_cumulative_levels_day1_7, 0) as sum_cumulative_levels_day1_7,
    COALESCE(am.sum_cumulative_levels_day1_14, 0) as sum_cumulative_levels_day1_14,
    COALESCE(am.sum_cumulative_levels_day1_20, 0) as sum_cumulative_levels_day1_20,

    -- Average levels per day
    ROUND(COALESCE(am.sum_cumulative_levels_day1_3, 0) / 3.0, 2) as avg_cumulative_levels_day1_3,
    ROUND(COALESCE(am.sum_cumulative_levels_day1_7, 0) / 7.0, 2) as avg_cumulative_levels_day1_7,
    ROUND(COALESCE(am.sum_cumulative_levels_day1_14, 0) / 14.0, 2) as avg_cumulative_levels_day1_14,
    ROUND(COALESCE(am.sum_cumulative_levels_day1_20, 0) / 20.0, 2) as avg_cumulative_levels_day1_20,

    -- Completion rates
    ROUND(
      CASE WHEN am.sum_cumulative_levels_day1_1 > 0
      THEN am.completed_levels_day1_1 * 100.0 / am.sum_cumulative_levels_day1_1
      ELSE 0 END, 2
    ) as completion_rate_day1_1,

    ROUND(
      CASE WHEN am.sum_cumulative_levels_day1_3 > 0
      THEN am.completed_levels_day1_3 * 100.0 / am.sum_cumulative_levels_day1_3
      ELSE 0 END, 2
    ) as completion_rate_day1_3,

    ROUND(
      CASE WHEN am.sum_cumulative_levels_day1_7 > 0
      THEN am.completed_levels_day1_7 * 100.0 / am.sum_cumulative_levels_day1_7
      ELSE 0 END, 2
    ) as completion_rate_day1_7,

    ROUND(
      CASE WHEN am.sum_cumulative_levels_day1_14 > 0
      THEN am.completed_levels_day1_14 * 100.0 / am.sum_cumulative_levels_day1_14
      ELSE 0 END, 2
    ) as completion_rate_day1_14,

    ROUND(
      CASE WHEN am.sum_cumulative_levels_day1_20 > 0
      THEN am.completed_levels_day1_20 * 100.0 / am.sum_cumulative_levels_day1_20
      ELSE 0 END, 2
    ) as completion_rate_day1_20,

    -- Max level reached
    COALESCE(am.max_level_reach_day1_1, 0) as max_level_reach_day1_1,
    COALESCE(am.max_level_reach_day1_3, 0) as max_level_reach_day1_3,
    COALESCE(am.max_level_reach_day1_7, 0) as max_level_reach_day1_7,
    COALESCE(am.max_level_reach_day1_14, 0) as max_level_reach_day1_14,
    COALESCE(am.max_level_reach_day1_20, 0) as max_level_reach_day1_20,

    -- Revenue metrics
    COALESCE(rm.revenue_day1_1, 0) as revenue_day1_1,
    COALESCE(rm.revenue_day1_3, 0) as revenue_day1_3,
    COALESCE(rm.revenue_day1_7, 0) as revenue_day1_7,
    COALESCE(rm.revenue_day1_14, 0) as revenue_day1_14,
    COALESCE(rm.revenue_day1_20, 0) as revenue_day1_20, -- TARGET

    -- Boolean features
    COALESCE(am.is_return_next_day, 0) as is_return_next_day,
    ui.is_android as is_android_user,
    ui.is_female as is_female,
    ui.is_age_30 as is_age_30

  FROM user_installs ui
  LEFT JOIN first_purchases fp ON ui.user_id = fp.user_id
  LEFT JOIN activity_metrics am ON ui.user_id = am.user_id
  LEFT JOIN revenue_metrics rm ON ui.user_id = rm.user_id
  ORDER BY ui.user_id
