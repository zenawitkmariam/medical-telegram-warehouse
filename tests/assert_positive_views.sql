-- This test passes if it returns 0 rows.
-- It checks for negative view counts which would indicate a data corruption issue.

select
    message_id,
    view_count
from {{ ref('fct_messages') }}
where view_count < 0