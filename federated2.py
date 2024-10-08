import asyncio
import pickle
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import logging
import ssl
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    ip: str
    port: int

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class ModelManager:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, data_loader: torch.utils.data.DataLoader, epochs: int = 1) -> None:
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            logger.info(f"Epoch {epoch+1}/{epochs} completed")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def update_parameters(self, new_params: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(new_params)

class DataManager:
    @staticmethod
    def get_data_loader(data: Dict[str, torch.Tensor]) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(data['inputs'], data['labels'])
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

class CommunicationManager:
    def __init__(self):
        self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.ssl_context.load_cert_chain(certfile="path/to/cert.pem", keyfile="path/to/key.pem")

    async def send_model(self, peer: PeerInfo, model_data: bytes) -> None:
        try:
            reader, writer = await asyncio.open_connection(peer.ip, peer.port, ssl=self.ssl_context)
            writer.write(model_data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"Error sending model to {peer.ip}:{peer.port}: {str(e)}")

    async def receive_model(self, reader: asyncio.StreamReader) -> bytes:
        try:
            data = await reader.read()
            return data
        except Exception as e:
            logger.error(f"Error receiving model: {str(e)}")
            return b''

class AggregationManager:
    @staticmethod
    def aggregate_models(models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        aggregated = {}
        num_models = len(models)
        for key in models[0].keys():
            aggregated[key] = sum(model[key] for model in models) / num_models
        return aggregated

class FederatedClient:
    def __init__(self, model: nn.Module, local_data: Any, peers: List[PeerInfo], local_address: str, local_port: int):
        self.model_manager = ModelManager(model)
        self.data_manager = DataManager()
        self.comm_manager = CommunicationManager()
        self.aggregation_manager = AggregationManager()
        self.local_data = local_data
        self.peers = peers
        self.local_address = local_address
        self.local_port = local_port
        self.server = None
        self.stop_event = asyncio.Event()

    async def run_training_cycle(self, epochs: int = 1) -> None:
        data_loader = self.data_manager.get_data_loader(self.local_data)
        self.model_manager.train(data_loader, epochs)

        local_params = self.model_manager.get_parameters()
        serialized_params = pickle.dumps(local_params)

        send_tasks = [self.comm_manager.send_model(peer, serialized_params) for peer in self.peers]
        await asyncio.gather(*send_tasks)

        received_params = await self.receive_from_peers()
        aggregated_params = self.aggregation_manager.aggregate_models([local_params] + received_params)
        self.model_manager.update_parameters(aggregated_params)

    async def receive_from_peers(self) -> List[Dict[str, torch.Tensor]]:
        self.server = await asyncio.start_server(self.handle_client, self.local_address, self.local_port, ssl=self.comm_manager.ssl_context)
        async with self.server:
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        return self.received_models

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        data = await self.comm_manager.receive_model(reader)
        if data:
            params = pickle.loads(data)
            self.received_models.append(params)
        writer.close()
        await writer.wait_closed()

    async def run(self, num_cycles: int = 5) -> None:
        for cycle in range(num_cycles):
            logger.info(f"Starting training cycle {cycle + 1}")
            await self.run_training_cycle()
            logger.info(f"Completed training cycle {cycle + 1}")

    async def shutdown(self):
        logger.info("Shutting down client...")
        self.stop_event.set()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("Client shutdown complete")

async def main():
    local_address = '0.0.0.0'
    local_port = 8000
    peers = [
        PeerInfo(ip='192.168.1.2', port=8000),
        PeerInfo(ip='192.168.1.3', port=8000),
    ]

    model = NeuralNet()
    local_data = {
        'inputs': torch.randn(1000, 10),
        'labels': torch.randint(0, 2, (1000,))
    }

    client = FederatedClient(
        model=model,
        local_data=local_data,
        peers=peers,
        local_address=local_address,
        local_port=local_port
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(getattr(signal, signame), lambda: asyncio.create_task(client.shutdown()))

    try:
        await client.run()
    finally:
        await client.shutdown()

async def test_federated_learning_simulation():
    num_clients = 3
    initial_model = NeuralNet()
    global_parameters = initial_model.state_dict()

    async def simulate_client(client_id: int):
        local_data = {
            'inputs': torch.randn(1000, 10),
            'labels': torch.randint(0, 2, (1000,))
        }

        model = NeuralNet()
        model.load_state_dict(global_parameters)

        client = FederatedClient(
            model=model,
            local_data=local_data,
            peers=[],
            local_address='127.0.0.1',
            local_port=8000 + client_id
        )

        data_loader = client.data_manager.get_data_loader(local_data)
        client.model_manager.train(data_loader, epochs=1)

        await asyncio.sleep(0.5 + 1.5 * client_id / num_clients)  # Simulated network latency

        return client.model_manager.get_parameters()

    client_tasks = [simulate_client(i) for i in range(num_clients)]
    client_parameters = await asyncio.gather(*client_tasks)

    aggregated_params = AggregationManager.aggregate_models(client_parameters)
    initial_model.load_state_dict(aggregated_params)

    logger.info("Federated learning simulation completed.")

if __name__ == '__main__':
    asyncio.run(main())
    asyncio.run(test_federated_learning_simulation())