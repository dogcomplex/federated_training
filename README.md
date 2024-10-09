# Federated Machine Learning Training Repository

This repository contains implementations of federated machine learning training algorithms, focusing on simulations for both localhost and remote connections.

## Current Status

The project currently consists of two main implementations:

1. federated3.py: Simulations for localhost connections (remote connections are currently untested).

federated2.py: Multi-process simulations, which is the most mature and functional implementation.

## Intent

The primary goal of this project is to explore and implement federated learning techniques, allowing for distributed machine learning across multiple devices or servers while maintaining data privacy.

## Installation

### Windows

Ensure you have Python 3.7+ installed.

Clone this repository:

```

git clone git@github.com:dogcomplex/federated_training.git

cd federated_training

```

Create a virtual environment:

```

python -m venv venv

venv\Scripts\activate

```

Install required packages:

```

pip install -r requirements.txt

```

### Linux

1. Ensure you have Python 3.7+ installed.

2. Clone this repository:

```

git clone https://github.com/yourusername/federated_training.git

cd federated_training

```

Create a virtual environment:

```

python3 -m venv venv

source venv/bin/activate

```

Install required packages:

```

pip install -r requirements.txt

```
(you may need to do your own CUDA wrangling.  God speed)

## Usage

To run the most mature current implementation:

```

python federated2.py

```

This will execute the multi-process simulation of federated learning.

## Features

Simulated federated learning environment

Multi-process implementation for parallel client simulations

Adaptive aggregation strategies

Performance comparisons between federated and centralized learning

## Future Work

Implement and test remote connections in federated3.py

Enhance security measures for data privacy

Optimize communication efficiency between clients and server

Explore more advanced federated learning algorithms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.