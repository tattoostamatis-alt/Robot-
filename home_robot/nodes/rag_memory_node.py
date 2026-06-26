#!/usr/bin/env python3
"""RAG long-term memory — ChromaDB + Lemonade (embed-gemma-300m-FLM, NPU)
embeddings, queried/fed by llm_bridge_node.py.

Subscribes to `memory/store` (std_msgs/String, raw fact text) and stores it
as an embedded document. Subscribes to `memory/query` (std_msgs/String, raw
question text), runs a similarity search, and publishes the matching facts
as a JSON list on `memory/answer` (std_msgs/String) — same query/answer
topic pattern as vision_node.py's `vision/query`/`vision/answer`.
"""

import json
import os
import time
import uuid

import chromadb
import rclpy
import requests
from chromadb.api.types import EmbeddingFunction
from rclpy.node import Node
from std_msgs.msg import String


class LemonadeEmbeddingFunction(EmbeddingFunction):
    def __init__(self, url, model):
        self.url = url
        self.model = model

    def name(self):
        return f'lemonade-{self.model}'

    def __call__(self, input):
        r = requests.post(f'{self.url}/embeddings',
                           json={'model': self.model, 'input': list(input)}, timeout=30)
        r.raise_for_status()
        data = sorted(r.json()['data'], key=lambda e: e['index'])
        return [e['embedding'] for e in data]


class RagMemoryNode(Node):
    def __init__(self):
        super().__init__('rag_memory_node')

        self.declare_parameter('lemonade_url', 'http://127.0.0.1:13305/api/v1')
        self.declare_parameter('embedding_model', 'embed-gemma-300m-FLM')
        self.declare_parameter('db_path', os.path.expanduser('~/.robot_memory'))
        self.declare_parameter('top_k', 3)
        self.declare_parameter('max_distance', 1.0)

        lemonade_url = self.get_parameter('lemonade_url').value
        embedding_model = self.get_parameter('embedding_model').value
        db_path = self.get_parameter('db_path').value
        self.top_k = self.get_parameter('top_k').value
        self.max_distance = self.get_parameter('max_distance').value

        ef = LemonadeEmbeddingFunction(lemonade_url, embedding_model)
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_or_create_collection('robot_memory', embedding_function=ef)

        self.answer_pub = self.create_publisher(String, 'memory/answer', 10)
        self.create_subscription(String, 'memory/store', self._on_store, 10)
        self.create_subscription(String, 'memory/query', self._on_query, 10)

        self.get_logger().info(
            f'RAG memory node started — db={db_path} '
            f'({self.collection.count()} facts stored), model={embedding_model}')

    def _on_store(self, msg: String):
        fact = msg.data.strip()
        if not fact:
            return
        try:
            self.collection.add(documents=[fact], ids=[str(uuid.uuid4())],
                                 metadatas=[{'stored_at': time.time()}])
            self.get_logger().info(f'Remembered: {fact}')
        except Exception as e:
            self.get_logger().error(f'Failed to store memory: {e}')

    def _on_query(self, msg: String):
        question = msg.data.strip()
        facts = []
        if question and self.collection.count() > 0:
            try:
                res = self.collection.query(
                    query_texts=[question],
                    n_results=min(self.top_k, self.collection.count()))
                for doc, dist in zip(res['documents'][0], res['distances'][0]):
                    if dist <= self.max_distance:
                        facts.append(doc)
            except Exception as e:
                self.get_logger().error(f'Memory query failed: {e}')
        self.answer_pub.publish(String(data=json.dumps(facts, ensure_ascii=False)))

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = RagMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
