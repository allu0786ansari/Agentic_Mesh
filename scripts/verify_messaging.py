"""
scripts/verify_messaging.py
Week 1 validation: sends 10 messages to Kafka and 10 to NATS JetStream,
reads them back, and asserts round-trip success for both.
Run: python scripts\verify_messaging.py
Expected final line: "All messaging checks passed"
"""
from __future__ import annotations
import asyncio, json, os, sys, time, uuid
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
NATS_URL        = os.environ.get("NATS_URL", "nats://localhost:4222")
TEST_TOPIC      = "telemetry.raw"
NUM_MESSAGES    = 10
TIMEOUT_S       = 15


def check_kafka() -> tuple[bool, str]:
    try:
        from kafka import KafkaProducer, KafkaConsumer
    except ImportError:
        return False, "kafka-python not installed — run: pip install kafka-python"
    run_id   = str(uuid.uuid4())[:8]
    messages = [json.dumps({"run_id": run_id, "seq": i}).encode() for i in range(NUM_MESSAGES)]
    try:
        producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP, request_timeout_ms=5000)
        for msg in messages:
            producer.send(TEST_TOPIC, value=msg)
        producer.flush(timeout=10)
        producer.close()
    except Exception as exc:
        return False, f"Kafka produce failed: {exc}"
    try:
        consumer = KafkaConsumer(
            TEST_TOPIC, bootstrap_servers=KAFKA_BOOTSTRAP,
            auto_offset_reset="earliest",
            consumer_timeout_ms=TIMEOUT_S * 1000,
            group_id=f"verify-{run_id}",
            value_deserializer=lambda b: json.loads(b.decode()),
        )
        received = []
        for record in consumer:
            if record.value.get("run_id") == run_id:
                received.append(record.value)
            if len(received) >= NUM_MESSAGES:
                break
        consumer.close()
    except Exception as exc:
        return False, f"Kafka consume failed: {exc}"
    if len(received) < NUM_MESSAGES:
        return False, f"Only received {len(received)}/{NUM_MESSAGES}"
    return True, f"Kafka: {NUM_MESSAGES}/{NUM_MESSAGES} messages round-tripped"


async def _nats_check() -> tuple[bool, str]:
    try:
        import nats
        from nats.errors import TimeoutError as NatsTimeout
    except ImportError:
        return False, "nats-py not installed — run: pip install nats-py"
    run_id  = str(uuid.uuid4())[:8]
    subject = f"agmesh.verify.{run_id}"
    nc = None
    try:
        nc = await nats.connect(NATS_URL, connect_timeout=5)
        js = nc.jetstream()
        try:
            await js.add_stream(name="AGMESH_VERIFY", subjects=["agmesh.verify.>"])
        except Exception:
            pass
        for i in range(NUM_MESSAGES):
            await js.publish(subject, json.dumps({"run_id": run_id, "seq": i}).encode())
        sub      = await js.subscribe(subject, durable=f"v-{run_id}")
        received = []
        deadline = time.monotonic() + TIMEOUT_S
        while len(received) < NUM_MESSAGES and time.monotonic() < deadline:
            try:
                msg  = await sub.next_msg(timeout=2.0)
                data = json.loads(msg.data.decode())
                if data.get("run_id") == run_id:
                    received.append(data)
                await msg.ack()
            except NatsTimeout:
                break
        await sub.unsubscribe()
        try:
            await js.delete_stream("AGMESH_VERIFY")
        except Exception:
            pass
    except Exception as exc:
        return False, f"NATS failed: {exc}"
    finally:
        if nc:
            await nc.drain()
    if len(received) < NUM_MESSAGES:
        return False, f"Only received {len(received)}/{NUM_MESSAGES} — is port-forward running?"
    return True, f"NATS: {NUM_MESSAGES}/{NUM_MESSAGES} messages round-tripped"


def check_nats() -> tuple[bool, str]:
    return asyncio.run(_nats_check())


def main() -> int:
    console.rule("[bold cyan]Week 1 — Messaging Verification[/bold cyan]")
    results = []
    console.print(f"\n[yellow]Checking Kafka ({KAFKA_BOOTSTRAP})...[/yellow]")
    results.append(("Kafka", *check_kafka()))
    console.print(f"\n[yellow]Checking NATS JetStream ({NATS_URL})...[/yellow]")
    results.append(("NATS JetStream", *check_nats()))

    table = Table(title="Results", show_header=True)
    table.add_column("Service", style="bold")
    table.add_column("Status")
    table.add_column("Detail")
    all_ok = True
    for service, ok, detail in results:
        table.add_row(service, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
        if not ok:
            all_ok = False
    console.print()
    console.print(table)
    if all_ok:
        console.print("\n[bold green]All messaging checks passed[/bold green]")
        return 0
    console.print("\n[bold red]One or more checks FAILED[/bold red]")
    return 1

if __name__ == "__main__":
    sys.exit(main())