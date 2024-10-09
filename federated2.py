import asyncio
import pickle
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
import logging
import ssl
import signal
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision import datasets, transforms
from contextlib import asynccontextmanager
import json
import os
import time
import multiprocessing
from functools import partial
import numpy as np
import random
from tqdm import tqdm
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add this constant at the top of the file
CENTRALIZED_RESOURCE_FRACTION = 0.1

@dataclass
class PeerInfo:
    ip: str
    port: int

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelManager:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, data_loader: torch.utils.data.DataLoader, epochs: int = 1, optimizer: optim.Optimizer = None) -> None:
        self.model.train()
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
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

    @staticmethod
    def get_label_based_subsets(dataset, num_clients):
        labels = dataset.targets.numpy()
        label_indices = [np.where(labels == i)[0] for i in range(10)]
        client_subsets = [[] for _ in range(num_clients)]
        for label_idx in label_indices:
            np.random.shuffle(label_idx)
            split_indices = np.array_split(label_idx, num_clients)
            for i, split in enumerate(split_indices):
                client_subsets[i].extend(split)
        return [torch.utils.data.Subset(dataset, indices) for indices in client_subsets]

    @staticmethod
    def get_iid_subsets(dataset, num_clients):
        num_items = len(dataset)
        indices = list(range(num_items))
        random.shuffle(indices)
        return [torch.utils.data.Subset(dataset, indices[i::num_clients]) for i in range(num_clients)]

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
        self.optimizer = optim.SGD(self.model_manager.model.parameters(), lr=0.01, momentum=0.9)
        self.training_time = 0.0
        self.network_delay = 0.0

    async def run_training_cycle(self, epochs: int = 1) -> Dict[str, torch.Tensor]:
        start_time = time.time()
        data_loader = self.data_manager.get_data_loader(self.local_data)
        self.model_manager.train(data_loader, epochs, self.optimizer)
        self.training_time = time.time() - start_time
        self.network_delay = simulate_network_delay()
        return self.model_manager.get_parameters()

    def update_parameters(self, new_params: Dict[str, torch.Tensor]) -> None:
        self.model_manager.update_parameters(new_params)

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = correct / total
    f1 = f1_score(all_targets, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return accuracy, f1, conf_matrix

@dataclass
class ClientStats:
    client_id: int
    training_time: float
    network_delay: float

@dataclass
class RoundStats:
    round_number: int
    client_stats: List[ClientStats] = field(default_factory=list)
    total_round_time: float = 0.0

def simulate_network_delay():
    return random.uniform(0.1, 0.5)  # Simulated delay between 100ms and 500ms

def run_client_training(model, local_data, optimizer, epochs, device, client_id):
    start_time = time.time()
    model.train()
    data_loader = DataManager.get_data_loader(local_data)
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Print progress every epoch with ongoing time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print(f"Client {client_id}: Epoch {epoch+1}/{epochs} completed. Epoch time: {epoch_time:.2f}s, Total time: {total_time:.2f}s")
    
    training_time = time.time() - start_time
    network_delay = simulate_network_delay()
    print(f"Client {client_id}: Training time: {training_time:.2f}s, Network delay: {network_delay:.2f}s")
    
    return model.state_dict(), ClientStats(client_id, training_time, network_delay)

def save_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

async def federated_learning_simulation(num_clients, num_rounds, local_epochs, train_dataset, test_loader, device, iid=False):
    cache_file = f'federated_cache_{"iid" if iid else "non_iid"}_{num_clients}_{num_rounds}_{local_epochs}.pkl'
    cached_data = load_cache(cache_file)
    
    if cached_data:
        print(f"Loading {'IID' if iid else 'non-IID'} federated learning results from cache...")
        return cached_data

    if iid:
        client_datasets = DataManager.get_iid_subsets(train_dataset, num_clients)
    else:
        client_datasets = DataManager.get_label_based_subsets(train_dataset, num_clients)
    
    global_model = MNISTNet().to(device)
    clients = [FederatedClient(MNISTNet().to(device), client_data) for client_data in client_datasets]
    
    early_stopping = EarlyStopping(patience=5)
    all_round_stats: List[RoundStats] = []
    
    max_round_times = []
    total_round_times = []

    for round in tqdm(range(num_rounds), desc="Federated Learning Rounds"):
        round_start_time = time.time()
        
        # Simulate parallel training
        with multiprocessing.Pool(processes=num_clients) as pool:
            results = pool.starmap(
                run_client_training,
                [(client.model_manager.model, client.local_data, client.optimizer, local_epochs, device, client_id) 
                 for client_id, client in enumerate(clients)]
            )
        
        client_states, client_stats = zip(*results)
        
        # Calculate max and total round times
        max_round_time = max(stat.training_time + stat.network_delay for stat in client_stats)
        total_round_time = sum(stat.training_time + stat.network_delay for stat in client_stats)
        
        max_round_times.append(max_round_time)
        total_round_times.append(total_round_time)
        
        # Aggregate models
        aggregated_state = AggregationManager.aggregate_models(client_states)
        global_model.load_state_dict(aggregated_state)
        
        # Update client models with the aggregated global model
        for client in clients:
            client.update_parameters(aggregated_state)
        
        # Evaluate global model
        accuracy, f1, conf_matrix = evaluate_model(global_model, test_loader, device)
        
        round_end_time = time.time()
        round_time = round_end_time - round_start_time
        
        # Store round statistics
        round_stats = RoundStats(round + 1, list(client_stats), round_time)
        all_round_stats.append(round_stats)
        
        early_stopping(1 - accuracy)  # Using accuracy as the metric to monitor
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    results = (global_model, accuracy, f1, conf_matrix, all_round_stats, max_round_times, total_round_times)
    save_cache(results, cache_file)
    return results

async def centralized_learning_simulation(model, train_loader, test_loader, device, num_epochs):
    cache_file = f'centralized_cache_{num_epochs}.pkl'
    cached_data = load_cache(cache_file)
    
    if cached_data:
        print("Loading centralized learning results from cache...")
        return cached_data

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    early_stopping = EarlyStopping(patience=5)
    
    # Limit CPU usage
    process = psutil.Process()
    cpu_count = psutil.cpu_count()
    process.cpu_affinity([i for i in range(int(cpu_count * CENTRALIZED_RESOURCE_FRACTION))])

    total_time = 0
    for epoch in tqdm(range(num_epochs), desc="Centralized Learning Epochs"):
        epoch_start_time = time.time()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        
        accuracy, f1, conf_matrix = evaluate_model(model, test_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        
        early_stopping(1 - accuracy)  # Using accuracy as the metric to monitor
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    results = (model, accuracy, f1, conf_matrix, total_time)
    save_cache(results, cache_file)
    return results

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    num_clients = 10
    num_rounds = 5
    local_epochs = 5
    
    try:
        print("Starting Non-IID Federated Learning Simulation...")
        fed_start_time = time.time()
        fed_model, fed_accuracy, fed_f1, fed_conf_matrix, round_stats, max_round_times, total_round_times = await federated_learning_simulation(
            num_clients, num_rounds, local_epochs, full_dataset, test_loader, device, iid=False
        )
        fed_end_time = time.time()

        print("\nStarting IID Federated Learning Simulation...")
        iid_fed_start_time = time.time()
        iid_fed_model, iid_fed_accuracy, iid_fed_f1, iid_fed_conf_matrix, iid_round_stats, iid_max_round_times, iid_total_round_times = await federated_learning_simulation(
            num_clients, num_rounds, local_epochs, full_dataset, test_loader, device, iid=True
        )
        iid_fed_end_time = time.time()

        print("\nStarting Centralized Learning Simulation...")
        cent_model = MNISTNet().to(device)
        cent_start_time = time.time()
        cent_model, cent_accuracy, cent_f1, cent_conf_matrix, cent_total_time = await centralized_learning_simulation(
            cent_model, train_loader, test_loader, device, num_epochs=5
        )
        cent_end_time = time.time()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("\nFinal Results:")
        if 'fed_accuracy' in locals():
            print("Non-IID Federated Learning:")
            print(f"  Accuracy: {fed_accuracy:.4f}, F1: {fed_f1:.4f}")
            print(f"  Total Time: {fed_end_time - fed_start_time:.2f}s")
            print(f"  Max Path Time: {sum(max_round_times):.2f}s")
            print(f"  Sum Total Time: {sum(total_round_times):.2f}s")
        else:
            print("Non-IID Federated Learning - Incomplete")

        if 'iid_fed_accuracy' in locals():
            print("\nIID Federated Learning:")
            print(f"  Accuracy: {iid_fed_accuracy:.4f}, F1: {iid_fed_f1:.4f}")
            print(f"  Total Time: {iid_fed_end_time - iid_fed_start_time:.2f}s")
            print(f"  Max Path Time: {sum(iid_max_round_times):.2f}s")
            print(f"  Sum Total Time: {sum(iid_total_round_times):.2f}s")
        else:
            print("IID Federated Learning - Incomplete")
        
        if 'cent_accuracy' in locals():
            print("\nCentralized Learning:")
            print(f"  Accuracy: {cent_accuracy:.4f}, F1: {cent_f1:.4f}")
            print(f"  Total Time: {cent_end_time - cent_start_time:.2f}s")
            print(f"  Actual Training Time: {cent_total_time:.2f}s")
        else:
            print("Centralized Learning - Incomplete")

        if 'fed_conf_matrix' in locals() and 'iid_fed_conf_matrix' in locals() and 'cent_conf_matrix' in locals():
            print("\nConfusion Matrices:")
            print("Non-IID Federated Learning:")
            print(fed_conf_matrix)
            print("\nIID Federated Learning:")
            print(iid_fed_conf_matrix)
            print("\nCentralized Learning:")
            print(cent_conf_matrix)

        if 'round_stats' in locals():
            print("\nRound Statistics:")
            for round_stat in round_stats:
                print(f"Round {round_stat.round_number}:")
                print(f"  Total Round Time: {round_stat.total_round_time:.2f}s")
                for client_stat in round_stat.client_stats:
                    print(f"  Client {client_stat.client_id}: Training Time: {client_stat.training_time:.2f}s, Network Delay: {client_stat.network_delay:.2f}s")
                print()

if __name__ == '__main__':
    asyncio.run(main())