#!/usr/bin/env python3
"""
Flower Federated Learning Simulation
Main entry point for running federated movie recommendation experiments
"""

import os
import sys
import logging
import torch
import flwr as fl
from typing import Dict, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.moviedb import MovieDBDataLayer
from src.models.foundation import FoundationModelManager
from src.models.minimal_foundation import MinimalFoundationManager
from src.models.recommendation import LoRARecommendationModel, RecommendationTrainer
from src.federation.client import create_client_factory
from src.federation.strategy import DomainAwareFedAvg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FederatedSimulation:
    """
    Orchestrates federated learning simulation using Flower.
    """

    def __init__(
        self,
        num_clients: int = 10,
        num_rounds: int = 10,
        use_foundation_model: bool = False,
        model_name: str = "intfloat/e5-large-v2",
        use_domain_aware_strategy: bool = False,
        use_minimal_foundation: bool = False
    ):
        """
        Initialize simulation.

        Args:
            num_clients: Number of federated clients
            num_rounds: Number of federation rounds
            use_foundation_model: Whether to use real foundation model
            model_name: Foundation model to use
            use_domain_aware_strategy: Whether to use domain-aware aggregation
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.use_foundation_model = use_foundation_model
        self.model_name = model_name
        self.use_domain_aware_strategy = use_domain_aware_strategy
        self.use_minimal_foundation = use_minimal_foundation

        logger.info(f"Initializing simulation with {num_clients} clients, {num_rounds} rounds")
        logger.info(f"Foundation model: {model_name if use_foundation_model else 'None (testing mode)'}")

        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("No GPU available, running on CPU")

    def setup_data(self, num_movies: int = 500, num_users: int = 200):
        """
        Setup MovieDB data layer.

        Args:
            num_movies: Number of synthetic movies
            num_users: Number of synthetic users
        """
        logger.info(f"Creating MovieDB dataset with {num_movies} movies, {num_users} users")
        self.data_layer = MovieDBDataLayer(
            num_movies=num_movies,
            num_users=num_users,
            seed=42
        )
        return self.data_layer

    def setup_foundation_model(self):
        """
        Setup foundation model if requested.

        Returns:
            Foundation model manager or None
        """
        if self.use_minimal_foundation:
            logger.info("Loading minimal foundation model (TF-IDF based)")
            manager = MinimalFoundationManager(model_name="minimal-tfidf")
            manager.load_model()
            return manager
        elif self.use_foundation_model:
            logger.info(f"Loading foundation model: {self.model_name}")
            manager = FoundationModelManager(model_name=self.model_name)
            manager.load_model()

            # Estimate client capacity
            capacity = manager.estimate_concurrent_clients(lora_rank=16)
            logger.info(f"Estimated concurrent client capacity: {capacity}")

            if capacity < self.num_clients:
                logger.warning(f"Client count ({self.num_clients}) exceeds capacity ({capacity})")

            return manager
        else:
            logger.info("Skipping foundation model (testing mode)")
            return None

    def create_model_factory(self, foundation_manager: Optional[FoundationModelManager]):
        """
        Create model factory function for clients.

        Args:
            foundation_manager: Foundation model manager

        Returns:
            Model factory function
        """
        # Store model name instead of the model itself to avoid serialization issues
        foundation_model_name = foundation_manager.model_name if foundation_manager else None

        def model_fn():
            """Create model and trainer for a client."""
            # Recreate foundation model in each worker to avoid serialization
            if foundation_model_name:
                # Create a new foundation manager in this worker
                from src.models.foundation import FoundationModelManager
                worker_foundation_manager = FoundationModelManager(foundation_model_name)
                worker_foundation_manager.load_model()

                # Move to CPU since workers don't have CUDA access
                foundation_model = worker_foundation_manager.model.to('cpu')
                embedding_dim = 1024  # e5-large-v2
            else:
                foundation_model = None
                embedding_dim = 128  # Small for testing

            # Create LoRA model on CPU (workers don't have CUDA)
            device = 'cpu'
            model = LoRARecommendationModel(
                foundation_model=foundation_model,
                embedding_dim=embedding_dim,
                lora_rank=16 if foundation_model_name else 4,  # Smaller rank for testing
                num_users=self.data_layer.num_users,
                dropout_rate=0.1
            ).to(device)

            # Create trainer
            trainer = RecommendationTrainer(
                model=model,
                learning_rate=5e-4,
                weight_decay=0.01
            )

            return model, trainer

        return model_fn

    def run_simulation(self):
        """
        Run federated learning simulation using Flower.
        """
        logger.info("Starting federated learning simulation")

        # Setup data
        data_layer = self.setup_data()

        # Setup foundation model
        foundation_manager = self.setup_foundation_model()

        # Create model factory
        model_fn = self.create_model_factory(foundation_manager)

        # Get tokenizer if using foundation model
        tokenizer = foundation_manager.tokenizer if foundation_manager else None

        # Create client factory with per-user LoRA parameter
        client_fn = create_client_factory(model_fn, data_layer, tokenizer, max_users_per_client=20)

        # Configure strategy
        if self.use_domain_aware_strategy:
            logger.info("Using domain-aware federated averaging strategy")
            strategy = DomainAwareFedAvg(
                fraction_fit=0.8,  # 80% of clients participate in training
                fraction_evaluate=0.5,  # 50% participate in evaluation
                min_fit_clients=min(2, self.num_clients),
                min_evaluate_clients=min(2, self.num_clients),
                min_available_clients=min(2, self.num_clients),
                evaluate_metrics_aggregation_fn=self.aggregate_metrics,
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config,
                domain_weight_factor=0.3,
                quality_weight_factor=0.4,
                contribution_weight_factor=0.3
            )
        else:
            logger.info("Using standard FedAvg strategy")
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=0.8,  # 80% of clients participate in training
                fraction_evaluate=0.5,  # 50% participate in evaluation
                min_fit_clients=min(2, self.num_clients),
                min_evaluate_clients=min(2, self.num_clients),
                min_available_clients=min(2, self.num_clients),
                evaluate_metrics_aggregation_fn=self.aggregate_metrics,
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config
            )

        # Run simulation
        logger.info("Launching Flower simulation...")

        # Configure simulation - use CPU for workers to avoid CUDA serialization issues
        # Foundation model stays on GPU in main process
        ray_init_args = {"num_gpus": 0}

        # Start simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
            ray_init_args=ray_init_args
        )

        logger.info("Simulation completed successfully!")

        # Print results
        self.print_results(history)

        return history

    def fit_config(self, server_round: int) -> Dict:
        """
        Configure training for each round.

        Args:
            server_round: Current round number

        Returns:
            Configuration dictionary
        """
        config = {
            "server_round": server_round,
            "local_epochs": 2 if server_round < 5 else 1,  # Fewer epochs in later rounds
            "batch_size": 16
        }
        return config

    def evaluate_config(self, server_round: int) -> Dict:
        """
        Configure evaluation for each round.

        Args:
            server_round: Current round number

        Returns:
            Configuration dictionary
        """
        config = {
            "server_round": server_round,
            "batch_size": 32
        }
        return config

    def aggregate_metrics(self, metrics):
        """
        Aggregate evaluation metrics from clients.

        Args:
            metrics: List of (num_examples, metrics_dict) tuples

        Returns:
            Aggregated metrics dictionary
        """
        if not metrics:
            return {}

        # Weighted average of metrics
        total_examples = sum(num_examples for num_examples, _ in metrics)

        aggregated = {}
        for num_examples, client_metrics in metrics:
            weight = num_examples / total_examples
            for key, value in client_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in aggregated:
                        aggregated[key] = 0
                    aggregated[key] += value * weight

        return aggregated

    def print_results(self, history):
        """
        Print simulation results.

        Args:
            history: Flower history object
        """
        print("\n" + "="*60)
        print("FEDERATED LEARNING SIMULATION RESULTS")
        print("="*60)

        if hasattr(history, "losses_distributed"):
            print("\nDistributed Training Losses by Round:")
            for round_num, loss in enumerate(history.losses_distributed, 1):
                if isinstance(loss, (list, tuple)) and len(loss) == 2:
                    # If loss is a tuple (round, value)
                    print(f"  Round {loss[0]}: {loss[1]:.4f}")
                elif isinstance(loss, (int, float)):
                    print(f"  Round {round_num}: {loss:.4f}")

        if hasattr(history, "metrics_distributed"):
            print("\nDistributed Metrics by Round:")
            for round_num, metrics in history.metrics_distributed.items():
                print(f"  Round {round_num}: {metrics}")

        print("\nSimulation completed successfully!")
        print("="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run federated learning simulation")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--use-foundation", action="store_true", help="Use real foundation model")
    parser.add_argument("--model", type=str, default="intfloat/e5-large-v2", help="Foundation model name")
    parser.add_argument("--domain-aware", action="store_true", help="Use domain-aware aggregation strategy")
    parser.add_argument("--minimal-foundation", action="store_true", help="Use minimal TF-IDF foundation model")

    args = parser.parse_args()

    # Create and run simulation
    simulation = FederatedSimulation(
        num_clients=args.clients,
        num_rounds=args.rounds,
        use_foundation_model=args.use_foundation,
        model_name=args.model,
        use_domain_aware_strategy=args.domain_aware,
        use_minimal_foundation=args.minimal_foundation
    )

    try:
        history = simulation.run_simulation()
        return 0
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
