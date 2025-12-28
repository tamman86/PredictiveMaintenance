
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from aiokafka import AIOKafkaProducer
import uvicorn
import json
import sys
import asyncio
from contextlib import asynccontextmanager

# Config
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "sensor_readings"

producer = None

# When server starts
async def lifespan(app: FastAPI):
    # Startup Logic
    global producer
    print(f"Connecting to Kafka at {KAFKA_BROKER}")
    try:
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        await producer.start()
        print("Connected to Kafka!")
    except Exception as e:
        print(f"Failed to connect to Kafka")

    yield

    # Shutdown Logic
    if producer:
        await producer.stop()
        print("Disconnected from Kafka!")

app = FastAPI(lifespan=lifespan)

# Data Model
class SensorReading(BaseModel):
    unit_nr: int
    time_cycles: int
    settings: List[float]
    sensors: List[float]

# Endpoint
@app.post("/ingest")
async def ingest_data(reading: SensorReading):
    # Receive data and send to Kafka

    # 1. Convert Pydantic object to dict
    data_dict = reading.model_dump()

    # 2. Send to Kafka
    if producer:
        try:
            await producer.send_and_wait(KAFKA_TOPIC, data_dict)
            return {"status": "sent_to_kafka"}
        except Exception as e:
            print(f"Kafka Send Error: {e}")
            return {"status": "error", "detail": str(e)}
    else:
        return {"status": "error", "detail": "Kafka Producer not connected"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
import sys
import asyncio

# --- 1. THE WINDOWS FIX ---
# aiokafka does not support the default Windows ProactorEventLoop.
# We must force it to use the SelectorEventLoop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from aiokafka import AIOKafkaProducer
import uvicorn
import json
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "sensor_readings"

# --- 2. THE PYCHARM FIX (Type Hinting) ---
# We tell PyCharm: "This variable can be None OR a Producer"
producer: Optional[AIOKafkaProducer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer
    print(f"🔌 Connecting to Kafka at {KAFKA_BROKER}...")

    try:
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await producer.start()
        print("✅ Connected to Kafka!")

        # --- 3. IMMEDIATE TEST ---
        # Send a test message right now to force topic creation
        print("📤 Sending Test Message...")
        await producer.send_and_wait(KAFKA_TOPIC, {"status": "system_startup_test"})
        print("✅ Test Message Sent! Topic should exist now.")

    except Exception as e:
        print(f"❌ KAFKA CONNECTION FAILED: {e}")
        print("Check if Docker is running and Port 9092 is exposed.")

    yield

    if producer:
        await producer.stop()
        print("🛑 Disconnected from Kafka.")


app = FastAPI(lifespan=lifespan)


class SensorReading(BaseModel):
    unit_nr: int
    time_cycles: int
    settings: List[float]
    sensors: List[float]


@app.post("/ingest")
async def ingest_data(reading: SensorReading):
    data_dict = reading.model_dump()

    # Safety check: If connection failed during startup, don't crash here
    if producer is None:
        return {"status": "error", "detail": "Kafka is not connected."}

    try:
        await producer.send_and_wait(KAFKA_TOPIC, data_dict)
        return {"status": "sent_to_kafka"}
    except Exception as e:
        print(f"⚠️ Kafka Send Error: {e}")
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''