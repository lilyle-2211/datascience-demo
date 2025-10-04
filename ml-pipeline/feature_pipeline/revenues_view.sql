SELECT user_id, eventDate as date, sum(transaction_value) as transaction_value
FROM `lilyle-demo.data_source.revenues` 

where eventDate >= '2022-06-06'
    AND eventDate IS NOT NULL
    AND transaction_value IS NOT NULL
    -- Exclude the specific anomaly transaction
    AND NOT (user_id = 21634 AND eventDate = '2022-06-24' AND transaction_value = 10000.0)

    group by 1, 2