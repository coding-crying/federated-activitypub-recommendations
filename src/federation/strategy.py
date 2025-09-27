"""
Custom Flower Aggregation Strategy
Implements domain-aware aggregation for federated recommendation systems
"""

import flwr as fl
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
import logging

logger = logging.getLogger(__name__)


class DomainAwareFedAvg(fl.server.strategy.FedAvg):
    """
    Domain-aware federated averaging strategy.

    Weights client contributions based on:
    1. Genre/domain similarity
    2. Client data quality metrics
    3. Convergence contribution
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        domain_weight_factor: float = 0.3,
        quality_weight_factor: float = 0.4,
        contribution_weight_factor: float = 0.3
    ):
        """
        Initialize domain-aware strategy.

        Args:
            domain_weight_factor: Weight for domain similarity
            quality_weight_factor: Weight for data quality metrics
            contribution_weight_factor: Weight for convergence contribution
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn
        )

        self.domain_weight_factor = domain_weight_factor
        self.quality_weight_factor = quality_weight_factor
        self.contribution_weight_factor = contribution_weight_factor

        # Track client metrics over time
        self.client_history: Dict[str, Dict] = {}
        self.global_loss_history: List[float] = []

        logger.info(f"DomainAwareFedAvg initialized with weights: "
                   f"domain={domain_weight_factor}, quality={quality_weight_factor}, "
                   f"contribution={contribution_weight_factor}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using domain-aware weighting.
        """
        if not results:
            return None, {}

        logger.info(f"Round {server_round}: Aggregating {len(results)} client results")

        # Extract client metrics and parameters
        client_data = []
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid

            # Extract client metrics
            metrics = fit_res.metrics if fit_res.metrics else {}

            client_data.append({
                'client_id': client_id,
                'parameters': fit_res.parameters,
                'num_examples': fit_res.num_examples,
                'metrics': metrics,
                'client_proxy': client_proxy
            })

        # Calculate domain-aware weights
        weights = self._calculate_domain_aware_weights(client_data, server_round)

        # Perform weighted aggregation
        aggregated_parameters = self._weighted_aggregate(client_data, weights)

        # Calculate aggregated metrics
        aggregated_metrics = self._aggregate_metrics(client_data, weights)

        # Update client history
        self._update_client_history(client_data, server_round)

        logger.info(f"Round {server_round}: Used domain-aware weights: {dict(zip([c['client_id'] for c in client_data], weights))}")

        return aggregated_parameters, aggregated_metrics

    def _calculate_domain_aware_weights(
        self,
        client_data: List[Dict],
        server_round: int
    ) -> List[float]:
        """
        Calculate domain-aware weights for each client.
        """
        num_clients = len(client_data)
        weights = np.ones(num_clients)

        # 1. Data quality weights (based on number of examples and loss)
        quality_weights = self._calculate_quality_weights(client_data)

        # 2. Domain similarity weights (based on genre preferences)
        domain_weights = self._calculate_domain_weights(client_data)

        # 3. Contribution weights (based on improvement contribution)
        contribution_weights = self._calculate_contribution_weights(client_data, server_round)

        # Combine weights
        for i in range(num_clients):
            weights[i] = (
                self.quality_weight_factor * quality_weights[i] +
                self.domain_weight_factor * domain_weights[i] +
                self.contribution_weight_factor * contribution_weights[i]
            )

        # Normalize weights
        weights = weights / np.sum(weights) * num_clients

        return weights.tolist()

    def _calculate_quality_weights(self, client_data: List[Dict]) -> np.ndarray:
        """Calculate weights based on data quality metrics."""
        num_clients = len(client_data)
        quality_weights = np.ones(num_clients)

        # Extract metrics
        example_counts = np.array([c['num_examples'] for c in client_data])
        losses = np.array([c['metrics'].get('train_loss', 1.0) for c in client_data])

        # Normalize example counts (more data = higher weight)
        if example_counts.std() > 0:
            example_weights = (example_counts - example_counts.min()) / example_counts.std()
            example_weights = np.exp(example_weights * 0.5)  # Soft scaling
        else:
            example_weights = np.ones(num_clients)

        # Normalize losses (lower loss = higher weight, but not too extreme)
        if losses.std() > 0:
            loss_weights = np.exp(-(losses - losses.min()) / losses.std() * 0.3)
        else:
            loss_weights = np.ones(num_clients)

        # Combine quality metrics
        quality_weights = example_weights * loss_weights

        # Normalize
        if quality_weights.sum() > 0:
            quality_weights = quality_weights / quality_weights.sum()
        else:
            quality_weights = np.ones(num_clients) / num_clients

        return quality_weights

    def _calculate_domain_weights(self, client_data: List[Dict]) -> np.ndarray:
        """Calculate weights based on domain/genre similarity."""
        num_clients = len(client_data)
        domain_weights = np.ones(num_clients)

        # Extract preferred genres for each client
        client_genres = []
        for client in client_data:
            genres = client['metrics'].get('preferred_genres', [])
            client_genres.append(set(genres) if isinstance(genres, list) else set())

        if not any(client_genres):
            return domain_weights / num_clients

        # Calculate pairwise genre similarity
        similarity_matrix = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    genres_i = client_genres[i]
                    genres_j = client_genres[j]
                    if genres_i and genres_j:
                        # Jaccard similarity
                        intersection = len(genres_i.intersection(genres_j))
                        union = len(genres_i.union(genres_j))
                        similarity_matrix[i][j] = intersection / union if union > 0 else 0

        # Weight based on average similarity to other clients
        # Clients with unique domains get lower weights to prevent overfitting
        avg_similarity = similarity_matrix.mean(axis=1)

        # Balance between specialization and generalization
        # Moderate similarity gets highest weight
        optimal_similarity = 0.3  # Sweet spot for cross-domain learning
        domain_weights = np.exp(-np.abs(avg_similarity - optimal_similarity) * 2)

        # Normalize
        if domain_weights.sum() > 0:
            domain_weights = domain_weights / domain_weights.sum()
        else:
            domain_weights = np.ones(num_clients) / num_clients

        return domain_weights

    def _calculate_contribution_weights(
        self,
        client_data: List[Dict],
        server_round: int
    ) -> np.ndarray:
        """Calculate weights based on historical contribution to convergence."""
        num_clients = len(client_data)
        contribution_weights = np.ones(num_clients)

        if server_round <= 1:
            return contribution_weights / num_clients

        # Look at historical performance improvements
        for i, client in enumerate(client_data):
            client_id = client['client_id']

            if client_id in self.client_history:
                history = self.client_history[client_id]

                # Calculate improvement trend
                recent_losses = history.get('loss_history', [])
                if len(recent_losses) >= 2:
                    # Clients showing consistent improvement get higher weight
                    improvement = recent_losses[-2] - recent_losses[-1]
                    contribution_weights[i] = max(0.1, 1.0 + improvement * 2)

        # Normalize
        if contribution_weights.sum() > 0:
            contribution_weights = contribution_weights / contribution_weights.sum()
        else:
            contribution_weights = np.ones(num_clients) / num_clients

        return contribution_weights

    def _weighted_aggregate(
        self,
        client_data: List[Dict],
        weights: List[float]
    ) -> Optional[Parameters]:
        """Perform weighted parameter aggregation."""
        if not client_data:
            return None

        # Convert parameters to numpy arrays
        param_arrays_list = []
        for client in client_data:
            param_arrays = [np.array(param) for param in client['parameters'].tensors]
            param_arrays_list.append(param_arrays)

        # Weighted average
        num_params = len(param_arrays_list[0])
        aggregated_arrays = []

        for param_idx in range(num_params):
            # Get all client arrays for this parameter
            param_values = [client_params[param_idx] for client_params in param_arrays_list]

            # Weighted average
            weighted_sum = np.zeros_like(param_values[0])
            for client_idx, param_value in enumerate(param_values):
                weighted_sum += weights[client_idx] * param_value

            aggregated_arrays.append(weighted_sum)

        # Convert back to Parameters
        return Parameters(tensors=[param.tobytes() for param in aggregated_arrays], tensor_type="numpy")

    def _aggregate_metrics(
        self,
        client_data: List[Dict],
        weights: List[float]
    ) -> Dict[str, Scalar]:
        """Aggregate client metrics using weights."""
        if not client_data:
            return {}

        aggregated = {}

        # Aggregate numeric metrics
        metric_keys = set()
        for client in client_data:
            metric_keys.update(client['metrics'].keys())

        for key in metric_keys:
            if key in ['preferred_genres', 'client_id']:
                continue

            values = []
            client_weights = []

            for i, client in enumerate(client_data):
                if key in client['metrics']:
                    value = client['metrics'][key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        client_weights.append(weights[i])

            if values:
                weighted_avg = np.average(values, weights=client_weights)
                aggregated[f"aggregated_{key}"] = weighted_avg

        # Add strategy-specific metrics
        aggregated["strategy_type"] = "domain_aware_fedavg"
        aggregated["num_contributing_clients"] = len(client_data)

        return aggregated

    def _update_client_history(
        self,
        client_data: List[Dict],
        server_round: int
    ):
        """Update historical tracking for each client."""
        for client in client_data:
            client_id = client['client_id']

            if client_id not in self.client_history:
                self.client_history[client_id] = {
                    'loss_history': [],
                    'accuracy_history': [],
                    'participation_rounds': []
                }

            history = self.client_history[client_id]

            # Update metrics history
            metrics = client['metrics']
            if 'train_loss' in metrics:
                history['loss_history'].append(metrics['train_loss'])
                # Keep only recent history
                if len(history['loss_history']) > 10:
                    history['loss_history'] = history['loss_history'][-10:]

            if 'accuracy' in metrics:
                history['accuracy_history'].append(metrics['accuracy'])
                if len(history['accuracy_history']) > 10:
                    history['accuracy_history'] = history['accuracy_history'][-10:]

            history['participation_rounds'].append(server_round)
            if len(history['participation_rounds']) > 20:
                history['participation_rounds'] = history['participation_rounds'][-20:]