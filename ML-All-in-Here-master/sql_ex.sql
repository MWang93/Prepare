--Recent 2 purchase time difference. Solution: lag(), row_number(), subquery
select user_id, unix_tiemstamp - previous_time as delta_secondlastone_lastone
from (
		select user_id, unix_timestamp,
			   lag(unix_timestamp,1) over (partition by user_id order by unix_tiemstamp) as previous_time
			   row_number() over (patition by user_id order by unix_timestamp desc) as order_desc
	    from query_one
	 ) tmp
where order_desc = 1
order by user_id
limit 5;


-- Client ratio by each user. Solution: COUNT() + PARTITION BY
with mobile_user(
	select *, 'mobile' as client
	from mobile_data 
),
with web_user(
	select *, 'web' as client
	from web_data
),
with all_user(
	select *
	from mobile_user

	union 

	select *
	from web_user
)

select user_id, client, count(page) / count(page) OVER (PARTITION BY user_id) as ratio
from all_user
group by user_id, client


SELECT 100*SUM(CASE WHEN m.user_id IS NULL THEN 1 ELSE 0 END)/COUNT(*) as WEB_ONLY,
	   100*SUM(CASE WHEN w.user_id IS NULL THEN 1 ELSE 0 END)/COUNT(*) as MOBILE_ONLY,
	   100*SUM(CASE WHEN m.user_id IS NOT NULL AND w.user_id IS NOT null THEN 1 ELSE 0 END)/COUNT(*) as BOTH
FROM (SELECT distinct user_id FROM query_two_web) w 
	  OUTER JOIN 
	 (SELECT distinct user_id FROM query_two_mobile) m ON m.user_id = w.user_id;


--Group by and row_number(), top Nth 
SELECT user_id, purchasedate
 FROM ( SELECT *, ROW_NUMBER() over(PARTITION BY user_id ORDER BY purchasedate) row_num 
		FROM query_three ) tmp
WHERE row_num = 3
LIMIT 5;


--Union and Group by 
SELECT user_id,
SUM(transaction_amount) as total_amount
FROM
(
SELECT * FROM query_four_march
UNION ALL
SELECT * FROM query_four_april
) tmp
GROUP BY user_id
ORDER BY user_id
LIMIT 5;

--Running Total, Windows_Sum() -- http://www.wagonhq.com/blog/running-totals-sql
SELECT user_id, date,
SUM(amount) over(PARTITION BY user_id ORDER BY date ROWS UNBOUNDED PRECEDING) as total_amount
FROM
(
	SELECT user_id, date, SUM(transaction_amount) as amount
	FROM query_four_march
	GROUP BY user_id, date
	UNION ALL
	SELECT user_id, date, SUM(transaction_amount) as amount
	FROM query_four_april
	GROUP BY user_id, date
) tmp
ORDER BY user_id, date
LIMIT 5;


--Average and Mode 

--Group by Country and Select the 1th and nth Country based on user_id COUNT
select country, user_count
from ( select *, row_number() over(order by user_count) as asc_num, row_number() over(order by user_count DESC) as desc_num
	   from (
	   			select country, count(distinc user_id) as user_count 
	   			from query_six 
	   			group by country
	   		) a
     ) tmp
where asc_num = 1 OR desc_num = 1

--Group by Country and Select 1th and nth user_id COUNT for each Country
SELECT user_id, created_at, country
FROM
(
	SELECT *,
	ROW_NUMBER() OVER (PARTITION BY country ORDER BY created_at) count_asc,
	ROW_NUMBER() OVER (PARTITION BY country ORDER BY created_at desc) count_desc
	FROM query_six
) tmp
WHERE count_asc = 1 or count_desc = 1
LIMIT 5;
