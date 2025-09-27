"""
Flower Client for Federated Learning
Extends NumPyClient to participate in federated learning
"""

import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class MovieRecommendationClient(fl.client.NumPyClient):
    """
    Flower client for federated movie recommendation learning with per-user LoRA.
    Shares only global model parameters, keeping user-specific LoRA and data local.
    """

    def __init__(
        self,
        client_id: str,
        model,
        trainer,
        data_layer,
        tokenizer=None,
        local_epochs: int = 2,
        batch_size: int = 16,
        max_users: int = 100
    ):
        """
        Initialize Flower client with per-user LoRA support.

        Args:
            client_id: Unique client identifier
            model: LoRA recommendation model
            trainer: Model trainer
            data_layer: MovieDB data layer
            tokenizer: Tokenizer for content (optional)
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            max_users: Maximum number of users supported by this client
        """
        self.client_id = client_id
        self.model = model
        self.trainer = trainer
        self.data_layer = data_layer
        self.tokenizer = tokenizer
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.max_users = max_users

        # Get client-specific data split
        self.client_data = data_layer.get_federated_split(
            client_id=int(client_id.split("_")[1]) if "_" in client_id else 0
        )

        # Get training and validation data
        self.train_data, self.val_data = data_layer.get_training_data(
            client_id=int(client_id.split("_")[1]) if "_" in client_id else 0
        )

        # Map user IDs from training data to local user indices
        self.user_id_mapping = {}
        user_counter = 0
        for item in self.train_data:
            original_user_id_str = item["user_id"]
            # Extract numeric part from user_id (e.g., from "user_123" extract 123)
            try:
                original_user_id = int(original_user_id_str.split("_")[1])
            except:
                original_user_id = hash(original_user_id_str) % 10000  # Fallback
            
            if original_user_id not in self.user_id_mapping:
                self.user_id_mapping[original_user_id] = user_counter
                user_counter += 1
                
            if user_counter >= self.max_users:
                break

        logger.info(f"Client {client_id} initialized:")
        logger.info(f"  Preferred genres: {self.client_data['preferred_genres']}")
        logger.info(f"  Training examples: {len(self.train_data)}")
        logger.info(f"  Validation examples: {len(self.val_data)}")
        logger.info(f"  User mappings: {len(self.user_id_mapping)} users mapped")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return global model parameters as NumPy arrays.
        Only returns shared parameters, not user-specific LoRA.

        Args:
            config: Configuration dictionary from server

        Returns:
            List of parameter arrays
        """
        # Get only global parameters for federated sharing
        # User-specific parameters stay local to preserve personalization
        global_params = self.model.get_lora_parameters()

        # Convert to list of numpy arrays
        parameters = []
        for key in sorted(global_params.keys()):
            param = global_params[key]
            parameters.append(param.detach().cpu().numpy())

        logger.debug(f"Client {self.client_id} sending {len(parameters)} global parameter arrays")

        return parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set model parameters from NumPy arrays.

        Args:
            parameters: List of parameter arrays from server
        """
        # Get current LoRA parameters to maintain key ordering
        lora_params = self.model.get_lora_parameters()
        keys = sorted(lora_params.keys())

        if len(parameters) != len(keys):
            raise ValueError(f"Parameter count mismatch: expected {len(keys)}, got {len(parameters)}")

        # Convert numpy arrays to tensors and update model
        param_dict = {}
        # Use CPU if CUDA is not available (Ray workers may not have GPU access)
        device = next(self.model.parameters()).device
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')
            # Move model to CPU if needed
            self.model = self.model.to(device)

        for key, param_array in zip(keys, parameters):
            param_tensor = torch.tensor(param_array, dtype=torch.float32).to(device)
            param_dict[key] = param_tensor

        self.model.set_lora_parameters(param_dict)

        logger.debug(f"Client {self.client_id} updated {len(param_dict)} parameters")

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally and return updated parameters.

        Args:
            parameters: Global model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id} starting local training")

        # Set global parameters
        self.set_parameters(parameters)

        # Get training configuration
        epochs = config.get("local_epochs", self.local_epochs)
        batch_size = config.get("batch_size", self.batch_size)

        # Local training
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            # Create batches
            for i in range(0, len(self.train_data), batch_size):
                batch_data = self.train_data[i:i + batch_size]

                if len(batch_data) == 0:
                    continue

                # Prepare batch
                batch = self._prepare_batch(batch_data)

                # Training step
                loss = self.trainer.train_step(batch)

                epoch_loss += loss
                epoch_batches += 1

            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                logger.info(f"Client {self.client_id} Epoch {epoch + 1}/{epochs}: loss = {avg_epoch_loss:.4f}")
                total_loss += epoch_loss
                num_batches += epoch_batches

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Get updated parameters
        updated_parameters = self.get_parameters(config)

        # Return metrics
        metrics = {
            "client_id": self.client_id,
            "train_loss": avg_loss,
            "num_examples": len(self.train_data),
            "preferred_genres": self.client_data["preferred_genres"]
        }

        logger.info(f"Client {self.client_id} finished training: loss = {avg_loss:.4f}")

        return updated_parameters, len(self.train_data), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model locally and return metrics.

        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration from server

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id} starting evaluation")

        # Set global parameters
        self.set_parameters(parameters)

        # Evaluate on validation data
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        batch_size = config.get("batch_size", self.batch_size)

        for i in range(0, len(self.val_data), batch_size):
            batch_data = self.val_data[i:i + batch_size]

            if len(batch_data) == 0:
                continue

            # Prepare batch
            batch = self._prepare_batch(batch_data)

            # Evaluate batch
            metrics = self.trainer.evaluate(batch)

            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            num_batches += 1

        # Calculate averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

        # Return metrics
        metrics = {
            "client_id": self.client_id,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "num_examples": len(self.val_data)
        }

        logger.info(f"Client {self.client_id} evaluation: loss = {avg_loss:.4f}, accuracy = {avg_accuracy:.4f}")

        return avg_loss, len(self.val_data), metrics

    def _prepare_batch(self, batch_data: List[Dict]) -> Dict:
        """
        Prepare a batch for training/evaluation with user ID mapping.

        Args:
            batch_data: List of data examples

        Returns:
            Batch dictionary ready for model
        """
        device = next(self.model.parameters()).device

        # Extract fields
        contents = [item["content"] for item in batch_data]
        
        # Map original user IDs to local user indices in this client
        original_user_ids = []
        local_user_ids = []
        for item in batch_data:
            original_user_id_str = item["user_id"]
            try:
                original_user_id = int(original_user_id_str.split("_")[1])
            except:
                original_user_id = hash(original_user_id_str) % 10000  # Fallback
            
            # Map to local user index
            if original_user_id in self.user_id_mapping:
                local_user_id = self.user_id_mapping[original_user_id]
            else:
                # Assign new local user index if not seen before
                local_user_id = len(self.user_id_mapping)
                self.user_id_mapping[original_user_id] = local_user_id
            
            original_user_ids.append(original_user_id)
            local_user_ids.append(local_user_id)
            
        labels = [item["label"] for item in batch_data]

        # Prepare content for foundation model
        if self.tokenizer is not None:
            # Traditional tokenizer approach
            content_tokens = self.tokenizer(
                contents,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
        else:
            # For SentenceTransformer or no tokenizer, pass texts directly
            content_tokens = {
                "texts": contents,
                "device": device,
                # Also include dummy tokens for backward compatibility
                "input_ids": torch.randint(0, 1000, (len(contents), 50)).to(device)
            }

        # Convert to tensors
        user_ids = torch.tensor(local_user_ids, dtype=torch.long).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)

        return {
            "content_tokens": content_tokens,
            "user_ids": user_ids,
            "labels": labels,
            "original_user_ids": original_user_ids  # Keep original IDs for reference
        }


def create_client_factory(model_fn, data_layer, tokenizer=None, max_users_per_client=20):
    """
    Create a client factory function for Flower simulation with per-user LoRA.

    Args:
        model_fn: Function to create model and trainer
        data_layer: MovieDB data layer
        tokenizer: Optional tokenizer
        max_users_per_client: Maximum users per client (for per-user LoRA)

    Returns:
        Client factory function
    """
    def client_fn(cid: str) -> fl.client.Client:
        """Create a Flower client with per-user LoRA support."""
        # Create model and trainer
        model, trainer = model_fn()

        # Create client with per-user LoRA support
        client = MovieRecommendationClient(
            client_id=cid,
            model=model,
            trainer=trainer,
            data_layer=data_layer,
            tokenizer=tokenizer,
            max_users=max_users_per_client
        )

        return client

    return client_fn


def test_flower_client():
    """Test Flower client creation."""
    import logging
    logging.basicConfig(level=logging.INFO)

    # Import dependencies
    from src.data.moviedb import MovieDBDataLayer
    from src.models.recommendation import LoRARecommendationModel, RecommendationTrainer

    # Create data layer
    data_layer = MovieDBDataLayer(num_movies=50, num_users=20)

    # Create model and trainer
    model = LoRARecommendationModel(
        foundation_model=None,  # Testing without foundation model
        embedding_dim=128,
        lora_rank=4,
        num_users=100
    )
    trainer = RecommendationTrainer(model)

    # Create client
    client = MovieRecommendationClient(
        client_id="client_0",
        model=model,
        trainer=trainer,
        data_layer=data_layer
    )

    # Test get_parameters
    params = client.get_parameters({})
    print(f"Number of parameter arrays: {len(params)}")
    print(f"Parameter shapes: {[p.shape for p in params]}")

    # Test set_parameters
    client.set_parameters(params)

    # Test fit (with dummy parameters)
    updated_params, num_examples, metrics = client.fit(params, {"local_epochs": 1})
    print(f"Training completed: {num_examples} examples")
    print(f"Metrics: {metrics}")

    return client


if __name__ == "__main__":
    test_flower_client()