import time
import requests
import psycopg

NODES = {
    1: ("us-east", "http://localhost:8080/_status/vars"),
    2: ("eu-west", "http://localhost:8081/_status/vars"),
    3: ("ap-south", "http://localhost:8082/_status/vars"),
}

DB = "host=localhost port=5432 dbname=metrics user=zerolatency password=zerolatency"


def parse_metrics(text):
    metrics = {}

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) != 2:
            continue

        name = parts[0]
        value = parts[1]

        if "{" in name:
            name = name.split("{", 1)[0]

        try:
            metrics.setdefault(name, []).append(float(value))
        except ValueError:
            pass

    return metrics


def get_value(metrics, name):
    values = metrics.get(name, [])
    return values[0] if values else None


def get_histogram_average_ms(metrics, name):
    total = get_value(metrics, name + "_sum")
    count = get_value(metrics, name + "_count")

    if total is None or count is None or count == 0:
        return None

    return (total / count) / 1000000.0


def collect_node(node_id, region, url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    metrics = parse_metrics(response.text)

    cpu = get_value(metrics, "admission_elastic_cpu_utilization")
    sql_queue = get_value(
        metrics,
        "admission_wait_queue_length_sql_sql_response"
    )
    network_rtt = get_histogram_average_ms(
        metrics,
        "round_trip_latency"
    )
    replication_latency = get_histogram_average_ms(
        metrics,
        "raft_replication_latency"
    )

    return (
        node_id,
        region,
        cpu,
        sql_queue,
        network_rtt,
        replication_latency,
    )


def main():
    conn = psycopg.connect(DB)

    print("COLLECTOR_STARTED")

    try:
        while True:
            for node_id, (region, url) in NODES.items():
                try:
                    row = collect_node(node_id, region, url)

                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO node_metrics
                            (
                                node_id,
                                region,
                                cpu_util,
                                sql_queue,
                                network_rtt_ms,
                                replication_lag_ms
                            )
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            row,
                        )

                    conn.commit()

                    print(
                        f"node={node_id} "
                        f"region={region} "
                        f"cpu={row[2]} "
                        f"queue={row[3]} "
                        f"rtt_ms={row[4]} "
                        f"repl_ms={row[5]}"
                    )

                except Exception as e:
                    conn.rollback()
                    print(f"node={node_id} ERROR: {e}")

            time.sleep(5)

    except KeyboardInterrupt:
        print("COLLECTOR_STOPPED")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
