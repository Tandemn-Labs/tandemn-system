"""
Lua scripts for atomic Redis chunk queue operations.

Each script runs as a single atomic operation on the Redis server,
preventing race conditions between concurrent workers.
"""

# LEASE_CHUNK
# Atomically pop one chunk from pending and move it to leased.
#
# KEYS[1] = pending_queue (sorted set, score=chunk_index)
# KEYS[2] = leased_set   (sorted set, score=expiry_timestamp)
# ARGV[1] = worker_id
# ARGV[2] = lease_ttl_seconds
# ARGV[3] = current_timestamp
# ARGV[4] = job_id (for building chunk meta key)
#
# Returns: chunk_id (string) or nil if queue is empty
LEASE_CHUNK = """
local chunk = redis.call('ZPOPMIN', KEYS[1])
if #chunk == 0 then
    return nil
end
local chunk_id = chunk[1]
local expiry = tonumber(ARGV[3]) + tonumber(ARGV[2])
redis.call('ZADD', KEYS[2], expiry, chunk_id)
local meta_key = 'job:' .. ARGV[4] .. ':chunk:' .. chunk_id
redis.call('HSET', meta_key, 'worker_id', ARGV[1], 'leased_at', ARGV[3])
redis.call('HINCRBY', meta_key, 'lease_count', 1)
return chunk_id
"""

# COMPLETE_CHUNK
# Atomically move a chunk from leased to completed and bump the counter.
#
# KEYS[1] = leased_set    (sorted set)
# KEYS[2] = completed_set (set)
# KEYS[3] = job_meta      (hash)
# ARGV[1] = chunk_id
# ARGV[2] = worker_id
#
# Returns: 1 on success, 0 if chunk was not in leased set
COMPLETE_CHUNK = """
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then
    return 0
end
redis.call('SADD', KEYS[2], ARGV[1])
redis.call('HINCRBY', KEYS[3], 'completed_count', 1)
return 1
"""

# RELEASE_CHUNK
# Return a leased chunk back to pending (graceful shutdown or failure).
#
# KEYS[1] = leased_set   (sorted set)
# KEYS[2] = pending_queue (sorted set)
# ARGV[1] = chunk_id
# ARGV[2] = chunk_index (original score for re-ordering in pending)
#
# Returns: 1 on success, 0 if chunk was not in leased set
RELEASE_CHUNK = """
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then
    return 0
end
redis.call('ZADD', KEYS[2], ARGV[2], ARGV[1])
return 1
"""

# REAP_EXPIRED_LEASES
# Find all leases past their expiry and re-enqueue or fail them.
#
# KEYS[1] = leased_set   (sorted set, score=expiry_timestamp)
# KEYS[2] = pending_queue (sorted set, score=chunk_index)
# ARGV[1] = current_timestamp
# ARGV[2] = job_id (for chunk meta keys and failed set)
# ARGV[3] = max_retries
#
# Returns: number of chunks reaped
REAP_EXPIRED_LEASES = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
local reaped = 0
for _, chunk_id in ipairs(expired) do
    local meta_key = 'job:' .. ARGV[2] .. ':chunk:' .. chunk_id
    local lease_count = tonumber(redis.call('HGET', meta_key, 'lease_count') or '0')
    redis.call('ZREM', KEYS[1], chunk_id)
    if lease_count >= tonumber(ARGV[3]) then
        redis.call('SADD', 'job:' .. ARGV[2] .. ':failed', chunk_id)
    else
        local idx = tonumber(redis.call('HGET', meta_key, 'chunk_index') or '0')
        redis.call('ZADD', KEYS[2], idx, chunk_id)
    end
    reaped = reaped + 1
end
return reaped
"""
