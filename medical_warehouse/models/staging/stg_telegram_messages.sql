/*
   Staging model for Telegram messages.
   Materialized as a view to ensure data is always fresh.
*/

with raw_messages as (
    select * from {{ source('raw_data', 'telegram_messages') }}
)

select
    -- 1. Cast and Rename
    cast(message_id as integer) as message_id,
    cast(channel_name as varchar(255)) as channel_name,
    cast(message_date as timestamp) as message_timestamp,
    
    -- 2. Standardize naming
    coalesce(message_text, '') as message_text,
    cast(coalesce(views, 0) as integer) as view_count,
    cast(coalesce(forwards, 0) as integer) as forward_count,
    
    -- 3. Add Calculated Fields
    length(coalesce(message_text, '')) as message_length,
    case 
        when has_media = true then 1 
        else 0 
    end as has_image,
    
    image_path

from raw_messages
-- 4. Filter Invalid Records
where message_id is not null 
  and message_text is not null 
  and message_text != ''