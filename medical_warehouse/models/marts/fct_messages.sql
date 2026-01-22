{{ config(materialized='table') }}

select
    m.message_id,
    md5(m.channel_name) as channel_key,
    to_char(m.message_timestamp, 'YYYYMMDD')::integer as date_key,
    m.message_text,
    m.message_length,
    m.view_count,
    m.forward_count,
    m.has_image
from {{ ref('stg_telegram_messages') }} m