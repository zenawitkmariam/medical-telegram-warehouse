{{ config(materialized='table') }}

with yolo_data as (
    select *
    from {{ source('raw', 'yolo_image_detections') }}
),

joined as (
    select
        m.message_id,
        c.channel_key,
        m.message_date::date as date_key,
        y.detected_object,
        y.confidence_score,
        y.image_category
    from yolo_data y
    join {{ ref('stg_telegram_messages') }} m
        on y.image_file = m.image_path
    join {{ ref('dim_channels') }} c
        on m.channel_name = c.channel_name
)

select * from joined
