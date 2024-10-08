import threading
import socket
import pickle
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model (can be replaced with any model)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer = nn.Linear(10, 2)  # Example dimensions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

# Federated Client class to handle local training and communication
class FederatedClient:
    def __init__(self, 
                 model: nn.Module, 
                 local_data: Any, 
                 peers: List[Dict[str, Any]], 
                 local_address: str, 
                 local_port: int):
        """
        Initialize the Federated Client.

        :param model: The local PyTorch model.
        :param local_data: Local dataset for training.
        :param peers: List of peer addresses [{'ip': str, 'port': int}, ...].
        :param local_address: IP address of the local machine.
        :param local_port: Port number for the local server.
        """
        self.model = model
        self.local_data = local_data
        self.peers = peers
        self.local_address = local_address
        self.local_port = local_port
        self.global_model_params = None  # Aggregated model parameters
        self.lock = threading.Lock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.stop_event = threading.Event()

    def train_local_model(self, epochs: int = 1):
        """
        Train the local model on local data.

        :param epochs: Number of training epochs.
        """
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        data_loader = self.get_data_loader(self.local_data)

        for epoch in range(epochs):
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def get_data_loader(self, data: Any) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for the local dataset.

        :param data: Local dataset.
        :return: DataLoader object.
        """
        # Replace with actual DataLoader creation code
        dataset = torch.utils.data.TensorDataset(data['inputs'], data['labels'])
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    def send_model(self, peer_ip: str, peer_port: int):
        """
        Send the local model parameters to a peer.

        :param peer_ip: IP address of the peer.
        :param peer_port: Port number of the peer.
        """
        model_data = self.serialize_model()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((peer_ip, peer_port))
            sock.sendall(model_data)

    def receive_model(self, connection: socket.socket):
        """
        Receive model parameters from a peer and aggregate them.

        :param connection: Socket connection object.
        """
        data = b''
        while True:
            packet = connection.recv(4096)
            if not packet:
                break
            data += packet

        peer_params = self.deserialize_model(data)
        with self.lock:
            self.aggregate_models(peer_params)

    def serialize_model(self) -> bytes:
        """
        Serialize the local model parameters.

        :return: Serialized model parameters.
        """
        return pickle.dumps(self.model.state_dict())

    def deserialize_model(self, data: bytes) -> Dict[str, torch.Tensor]:
        """
        Deserialize model parameters.

        :param data: Serialized model parameters.
        :return: Model parameters dictionary.
        """
        return pickle.loads(data)

    def aggregate_models(self, peer_params: Dict[str, torch.Tensor]):
        """
        Aggregate received model parameters with local parameters.

        :param peer_params: Parameters received from a peer.
        """
        if self.global_model_params is None:
            self.global_model_params = {k: v.clone() for k, v in peer_params.items()}
        else:
            for k in self.global_model_params.keys():
                self.global_model_params[k] += peer_params[k]

    def update_local_model(self):
        """
        Update the local model with aggregated global parameters.
        """
        if self.global_model_params is not None:
            num_models = len(self.peers) + 1  # Including local model
            averaged_params = {k: v / num_models for k, v in self.global_model_params.items()}
            self.model.load_state_dict(averaged_params)
            self.global_model_params = None

    def start_server(self):
        """
        Start a server to listen for incoming model updates from peers.
        """
        server_thread = threading.Thread(target=self.server_loop)
        server_thread.start()

    def server_loop(self):
        """
        Server loop to handle incoming connections.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.bind((self.local_address, self.local_port))
            server_sock.listen()
            server_sock.settimeout(1.0)  # Non-blocking accept with timeout
            while not self.stop_event.is_set():
                try:
                    conn, addr = server_sock.accept()
                    threading.Thread(target=self.handle_client, args=(conn,)).start()
                except socket.timeout:
                    continue  # Check for stop_event

    def handle_client(self, conn: socket.socket):
        """
        Handle a client connection.

        :param conn: Socket connection object.
        """
        with conn:
            self.receive_model(conn)

    def run_training_cycle(self, epochs: int = 1):
        """
        Run a full training cycle: local training, sending model, receiving models, updating local model.

        :param epochs: Number of training epochs.
        """
        self.train_local_model(epochs)
        send_threads = []
        for peer in self.peers:
            thread = threading.Thread(target=self.send_model, args=(peer['ip'], peer['port']))
            thread.start()
            send_threads.append(thread)

        # Wait for all send operations to complete
        for thread in send_threads:
            thread.join()

        # Optionally, wait for incoming models or implement a timeout
        # For simplicity, we proceed to update the local model
        self.update_local_model()

    def shutdown(self):
        """
        Shutdown the server and stop all threads.
        """
        self.stop_event.set()

# Main function to initialize and run the client
def main():
    # Configuration parameters
    local_address = '0.0.0.0'  # Local IP address
    local_port = 8000  # Local port to listen on
    peers = [
        {'ip': '192.168.1.2', 'port': 8000},
        {'ip': '192.168.1.3', 'port': 8000},
        # Add more peers as needed
    ]

    # Initialize local model
    model = NeuralNet()

    # Load local dataset
    local_data = {
        'inputs': torch.randn(1000, 10),
        'labels': torch.randint(0, 2, (1000,))
    }

    # Initialize Federated Client
    client = FederatedClient(
        model=model,
        local_data=local_data,
        peers=peers,
        local_address=local_address,
        local_port=local_port
    )

    # Start server to receive models from peers
    client.start_server()

    try:
        # Run multiple training cycles
        for _ in range(5):
            client.run_training_cycle(epochs=1)
    finally:
        # Ensure server is properly shutdown
        client.shutdown()

if __name__ == '__main__':
    main()

def test_federated_learning_simulation():
    """
    Simulate running multiple federated clients with network latency.
    """
    import time
    import random
    import threading

    # Number of simulated clients
    num_clients = 3

    # Shared global model parameters
    initial_model = NeuralNet()
    global_parameters = initial_model.state_dict()

    # Thread-safe lock for updating global parameters
    global_lock = threading.Lock()

    def simulate_client(client_id: int):
        # Simulate local data for each client
        local_data = {
            'inputs': torch.randn(1000, 10),
            'labels': torch.randint(0, 2, (1000,))
        }

        # Initialize client model with global parameters
        model = NeuralNet()
        model.load_state_dict(global_parameters)

        # Create FederatedClient instance without actual networking
        client = FederatedClient(
            model=model,
            local_data=local_data,
            peers=[],  # Peers are simulated
            local_address='127.0.0.1',
            local_port=8000 + client_id
        )

        # Train local model
        client.train_local_model(epochs=1)

        # Simulate network latency
        time.sleep(random.uniform(0.5, 2.0))

        # Collect local model parameters
        local_parameters = client.model.state_dict()

        # Update global parameters with thread safety
        with global_lock:
            for key in global_parameters:
                global_parameters[key] += local_parameters[key]

    # Create and start client threads
    client_threads = []
    for client_id in range(num_clients):
        thread = threading.Thread(target=simulate_client, args=(client_id,))
        thread.start()
        client_threads.append(thread)

    # Wait for all clients to complete
    for thread in client_threads:
        thread.join()

    # Average the aggregated parameters
    for key in global_parameters:
        global_parameters[key] = global_parameters[key] / num_clients

    # Update the initial model with new global parameters
    initial_model.load_state_dict(global_parameters)

    # Evaluate or save the updated model as needed
    print("Federated learning simulation completed.")
