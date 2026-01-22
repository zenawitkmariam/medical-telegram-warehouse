{{ config(materialized='table') }}

with channel_metrics as (
    -- Aggregate statistics per channel from the cleaned staging data
    select
        channel_name,
        min(message_timestamp) as first_post_date,
        max(message_timestamp) as last_post_date,
        count(message_id) as total_posts,
        avg(view_count) as avg_views
    from {{ ref('stg_telegram_messages') }}
    group by channel_name
)

select
    -- 1. Generate Surrogate Key (MD5 hash of the natural key)
    md5(channel_name) as channel_key,
    
    channel_name,

    -- 2. Logic for Channel Type based on common keywords
    case 
        when channel_name ilike '%pharma%' or channel_name ilike '%medicine%' then 'Pharmaceutical'
        when channel_name ilike '%lobelia%' or channel_name ilike '%cosmetic%' then 'Cosmetics'
        else 'Medical'
    end as channel_type,

    -- 3. Metrics
    first_post_date,
    last_post_date,
    total_posts,
    avg_views

from channel_metrics