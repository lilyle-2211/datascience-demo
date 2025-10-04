SELECT
cast(REGEXP_EXTRACT(user_id, r'_(\d+)') as int) user_id,
  case when install_date is null then null ELSE DATE(SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*SZ', install_date)) end as install_date,

  case 
    when lower(platform) in ( 'google_android' ,'android' ) then 1
    when lower(platform) in ('apple_ios','ios' ) then 0
    else 0
  end as is_android,
  case when SPLIT(channel_country, '-')[OFFSET(1)] = 'organic' then 1 else 0 end AS is_organic,
  case when gender is null  or gender = 'female'  then 1 else 0 end as is_female,
  CASE 
    
    WHEN SAFE_CAST(age AS INT64) <=30 THEN 1 else 0 end as is_age_30
    ,
  count(DISTINCT user_id) as num_player,
  count(*) as num_device
FROM `lilyle-demo.data_source.users`
WHERE DATE(SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*SZ', install_date)) >= '2022-06-06'
  AND DATE(SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*SZ', install_date)) <= CURRENT_DATE()
GROUP BY 1, 2, 3, 4, 5, 6
ORDER BY 1, 2