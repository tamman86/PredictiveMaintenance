### api.py

import sys
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from aiokafka import AIOKafkaProducer
import uvicorn
import json
from contextlib import asynccontextmanager

# --- 1. THE WINDOWS FIX ---
# Critical for running aiokafka on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- CONFIGURATION ---
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "sensor_readings"

# Global Producer Variable
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

    if producer is None:
        return {"status": "error", "detail": "Kafka is not connected."}

    try:
        # Fire and forget
        await producer.send(KAFKA_TOPIC, data_dict)
        return {"status": "sent_to_kafka"}
    except Exception as e:
        print(f"⚠️ Kafka Send Error: {e}")
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)