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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from contextlib import asynccontextmanager
import json
import os

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
    def get_data_loader(data: torch.utils.data.Subset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

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
    def __init__(self, model: nn.Module, local_data: torch.utils.data.Subset):
        self.model_manager = ModelManager(model)
        self.data_manager = DataManager()
        self.local_data = local_data

    async def run_training_cycle(self, epochs: int = 1) -> Dict[str, torch.Tensor]:
        data_loader = self.data_manager.get_data_loader(self.local_data)
        self.model_manager.train(data_loader, epochs)
        return self.model_manager.get_parameters()

    def update_parameters(self, new_params: Dict[str, torch.Tensor]) -> None:
        self.model_manager.update_parameters(new_params)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

async def test_federated_learning_simulation():
    print("Starting federated learning simulation setup...")
    num_clients = 3
    num_rounds = 5
    batch_size = 64
    test_batch_size = 1000

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    # Split dataset for federated learning
    client_dataset_size = len(full_dataset) // num_clients
    client_datasets = torch.utils.data.random_split(full_dataset, [client_dataset_size] * num_clients)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

    # Function to evaluate model
    def evaluate_model(model, data_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(data_loader.dataset)
        return accuracy

    # Check for cached centralized results
    cache_file = 'centralized_results.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_results = json.load(f)
        print(f"Using cached centralized results: {cached_results}")
        centralized_accuracy = cached_results['final_accuracy']
    else:
        # Centralized training
        logger.info("Starting centralized training...")
        centralized_model = MNISTNet()
        centralized_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        centralized_model_manager = ModelManager(centralized_model)
        
        for round in range(num_rounds):
            centralized_model_manager.train(centralized_loader, epochs=1)
            accuracy = evaluate_model(centralized_model, test_loader)
            logger.info(f"Centralized Round {round + 1}/{num_rounds} - Accuracy: {accuracy:.4f}")
        
        centralized_accuracy = evaluate_model(centralized_model, test_loader)
        logger.info(f"Final Centralized Accuracy: {centralized_accuracy:.4f}")

        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump({
                'num_rounds': num_rounds,
                'batch_size': batch_size,
                'final_accuracy': centralized_accuracy
            }, f)

    # Federated learning simulation
    print("Starting federated learning simulation...")
    async def simulate_client(client_id):
        print(f"Simulating client {client_id}...")
        model = MNISTNet()
        client = FederatedClient(
            model=model,
            local_data=client_datasets[client_id]
        )
        return await client.run_training_cycle(epochs=1)

    federated_model = MNISTNet()
    global_parameters = federated_model.state_dict()

    for round in range(num_rounds):
        print(f"Federated Round {round + 1}/{num_rounds}")
        client_tasks = [simulate_client(i) for i in range(num_clients)]
        client_parameters = await asyncio.gather(*client_tasks)

        print("Aggregating client models...")
        aggregated_params = AggregationManager.aggregate_models(client_parameters)
        federated_model.load_state_dict(aggregated_params)
        global_parameters = aggregated_params

        # Evaluate federated model on test data
        federated_accuracy = evaluate_model(federated_model, test_loader)
        
        print(f"Federated Round {round + 1}/{num_rounds} - Accuracy: {federated_accuracy:.4f}")

    print(f"Final Centralized Accuracy: {centralized_accuracy:.4f}")
    print(f"Final Federated Accuracy: {federated_accuracy:.4f}")
    print("Federated learning simulation completed.")

@asynccontextmanager
async def graceful_shutdown(client):
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        shutdown_event.set()
    
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(getattr(signal, signame), signal_handler)
    
    try:
        yield shutdown_event
    finally:
        for signame in ('SIGINT', 'SIGTERM'):
            loop.remove_signal_handler(getattr(signal, signame))
        await client.shutdown()

async def run_with_graceful_shutdown(client):
    async with graceful_shutdown(client) as shutdown_event:
        run_task = asyncio.create_task(client.run())
        await asyncio.wait([run_task, shutdown_event.wait()], return_when=asyncio.FIRST_COMPLETED)
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

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

    await run_with_graceful_shutdown(client)

if __name__ == '__main__':
    print("Starting the federated learning simulation...")
    asyncio.run(test_federated_learning_simulation())