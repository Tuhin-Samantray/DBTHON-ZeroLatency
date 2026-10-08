import argparse
import asyncio
import random
import time
import asyncpg

REGIONS = ["us-east", "eu-west", "ap-south"]
NODES = {
    "us-east": ("localhost", 26257),
    "eu-west": ("localhost", 26258),
    "ap-south": ("localhost", 26259),
}

USER_IDS = [101, 102, 103]
ITEM_IDS = [1, 2, 3]

async def execute_transaction(pool, region, mode, max_retries=3):
    start = time.perf_counter()
    for attempt in range(max_retries):
        try:
            async with pool.acquire() as conn:
                if mode == "read":
                    item_id = random.choice(ITEM_IDS)
                    await conn.fetchrow("SELECT * FROM inventory WHERE item_id = $1;", item_id)
                elif mode in ["write", "hotspot"]:
                    user_id = random.choice(USER_IDS)
                    item_id = 1 if mode == "hotspot" else random.choice(ITEM_IDS)
                    quantity = 1
                    
                    async with conn.transaction():
                        row = await conn.fetchrow(
                            "SELECT price, stock FROM inventory WHERE item_id = $1;", item_id
                        )
                        if row and row["stock"] >= quantity:
                            total_amount = float(row["price"]) * quantity
                            await conn.execute(
                                """
                                INSERT INTO orders (user_id, item_id, quantity, total_amount, gateway_region)
                                VALUES ($1, $2, $3, $4, $5);
                                """,
                                user_id, item_id, quantity, total_amount, region
                            )
                            await conn.execute(
                                "UPDATE inventory SET stock = stock - $1 WHERE item_id = $2;",
                                quantity, item_id
                            )
                elapsed = (time.perf_counter() - start) * 1000
                return elapsed, True, None
        except asyncpg.exceptions.SerializationError:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.01 * (2 ** attempt))  # Exponential backoff
                continue
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, False, "Max retries exceeded (SerializationFailure)"
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, False, str(e)

async def init_connection(conn):
    await conn.execute("SET multiple_active_portals_enabled = true;")

async def worker(pool, region, mode, duration, results):
    end_time = time.time() + duration
    while time.time() < end_time:
        latency, success, err = await execute_transaction(pool, region, mode)
        results.append((latency, success, err))
        await asyncio.sleep(0.01)

async def main():
    parser = argparse.ArgumentParser(description="ZeroLatency Load Generator")
    parser.add_argument("--region", choices=REGIONS, default="us-east", help="Gateway node region")
    parser.add_argument("--mode", choices=["read", "write", "hotspot"], default="read", help="Workload profile")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent workers")
    parser.add_argument("--duration", type=int, default=10, help="Test run duration in seconds")
    args = parser.parse_args()

    host, port = NODES[args.region]
    dsn = f"postgresql://root@{host}:{port}/ecommerce"

    print(f"[LoadGen] Connecting to {args.region} ({host}:{port}) | Mode: {args.mode} | Workers: {args.concurrency}")
    
    try:
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=args.concurrency,
            max_size=args.concurrency,
            statement_cache_size=0,
            init=init_connection
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to {dsn}: {e}")
        return

    results = []
    tasks = [
        asyncio.create_task(worker(pool, args.region, args.mode, args.duration, results))
        for _ in range(args.concurrency)
    ]
    await asyncio.gather(*tasks)
    await pool.close()

    latencies = [r[0] for r in results if r[1]]
    failures = [r for r in results if not r[1]]

    if results:
        print(f"\n--- LoadGen Execution Summary ---")
        print(f"Total Transactions: {len(results)} | Successful: {len(latencies)} | Failed: {len(failures)}")
        if failures:
            sample_err = failures[0][2]
            print(f"Sample Failure Error: {sample_err}")
        if latencies:
            latencies.sort()
            n = len(latencies)
            p50 = latencies[min(int(n * 0.50), n - 1)]
            p95 = latencies[min(int(n * 0.95), n - 1)]
            p99 = latencies[min(int(n * 0.99), n - 1)]
            print(f"Latency P50: {p50:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms")

if __name__ == "__main__":
    asyncio.run(main())
