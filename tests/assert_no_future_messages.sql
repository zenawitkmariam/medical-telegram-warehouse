-- This test passes if it returns 0 rows.
-- It looks for any messages dated after the current timestamp.

select
    message_id,
    message_timestamp
from {{ ref('stg_telegram_messages') }}
-- We compare against current_timestamp to ensure data is realistic
where message_timestamp > current_timestamp