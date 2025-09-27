"""
Real-Time Federated Learning for Social Media Recommendations
Handles live updates and real-time learning from user interactions
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict, deque
import threading
import time
import logging
from datetime import datetime, timedelta
import queue

logger = logging.getLogger(__name__)


class RealTimeFederatedLearning:
    """
    Handles real-time federated learning updates from live social media interactions.
    Processes user interactions as they happen and updates local LoRA adapters.
    """
    
    def __init__(self,
                 model_update_callback: Callable,
                 aggregation_callback: Callable,
                 client_id: str,
                 max_buffer_size: int = 1000,
                 update_frequency_minutes: int = 5,
                 batch_size: int = 32):
        """
        Initialize real-time federated learning system.

        Args:
            model_update_callback: Function to update local model parameters
            aggregation_callback: Function to aggregate parameters with server
            client_id: Unique identifier for this client
            max_buffer_size: Maximum number of interactions to buffer
            update_frequency_minutes: How often to perform local updates
            batch_size: Size of batches for training
        """
        self.model_update_callback = model_update_callback
        self.aggregation_callback = aggregation_callback
        self.client_id = client_id
        self.max_buffer_size = max_buffer_size
        self.update_frequency_minutes = update_frequency_minutes
        self.batch_size = batch_size
        
        # Interaction buffer to accumulate interactions
        self.interaction_buffer = deque(maxlen=max_buffer_size)
        
        # Timing control
        self.last_update_time = datetime.now()
        self.update_interval = timedelta(minutes=update_frequency_minutes)
        
        # Thread control
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()
        
        # Stats tracking
        self.total_interactions_processed = 0
        self.total_updates_performed = 0
        
        logger.info(f"RealTimeFederatedLearning initialized for client {client_id}")

    def on_new_interaction(self, 
                          user_id: str, 
                          post_id: str, 
                          interaction_type: str,
                          timestamp: datetime = None,
                          content_data: Dict = None) -> bool:
        """
        Handle new user interaction in real-time.

        Args:
            user_id: ID of the user who interacted
            post_id: ID of the post that was interacted with
            interaction_type: Type of interaction ('like', 'boost', 'reply', 'view', etc.)
            timestamp: When the interaction occurred (defaults to now)
            content_data: Additional content data for the post

        Returns:
            True if interaction was successfully queued
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        interaction = {
            'user_id': user_id,
            'post_id': post_id,
            'interaction_type': interaction_type,
            'timestamp': timestamp,
            'content_data': content_data or {},
            'processed': False
        }
        
        with self.lock:
            self.interaction_buffer.append(interaction)
            self.total_interactions_processed += 1
        
        logger.debug(f"Queued interaction: {interaction_type} by {user_id} on {post_id}")
        
        # Check if we should perform an update
        self._maybe_perform_update()
        
        return True

    def _maybe_perform_update(self):
        """Check if it's time to perform a local model update."""
        current_time = datetime.now()
        
        # Update if enough time has passed and we have enough interactions
        time_for_update = current_time - self.last_update_time >= self.update_interval
        has_interactions = len(self.interaction_buffer) > 0
        
        if time_for_update and has_interactions:
            self._perform_local_update()

    def _perform_local_update(self):
        """Perform a local model update using buffered interactions."""
        with self.lock:
            if len(self.interaction_buffer) == 0:
                return
            
            # Get interactions to process
            interactions_to_process = list(self.interaction_buffer)
            self.interaction_buffer.clear()
        
        try:
            # Process interactions and update local model
            processed_count = self._process_interactions_batch(interactions_to_process)
            
            # Update timing
            self.last_update_time = datetime.now()
            self.total_updates_performed += 1
            
            logger.info(f"Performed local update: processed {processed_count} interactions")
            
        except Exception as e:
            logger.error(f"Error during local update: {e}")

    def _process_interactions_batch(self, interactions: List[Dict]) -> int:
        """
        Process a batch of interactions and update the local model.

        Args:
            interactions: List of interaction dictionaries

        Returns:
            Number of interactions processed
        """
        # Convert interactions to training format
        training_data = []
        for interaction in interactions:
            # Create training example from interaction
            example = self._convert_interaction_to_training_example(interaction)
            if example is not None:
                training_data.append(example)
        
        if not training_data:
            return 0
        
        # Perform local training on the batch
        try:
            # Update local model using the training data
            self._local_training_step(training_data)
        except Exception as e:
            logger.error(f"Error during local training: {e}")
            return 0
        
        # Mark interactions as processed
        for interaction in interactions:
            interaction['processed'] = True
        
        return len(training_data)

    def _convert_interaction_to_training_example(self, interaction: Dict) -> Optional[Dict]:
        """
        Convert an interaction to a training example format.

        Args:
            interaction: Interaction dictionary

        Returns:
            Training example or None if conversion fails
        """
        try:
            # Determine interaction label based on type
            interaction_labels = {
                'view': 0.3,  # Low engagement
                'like': 0.7,  # Medium engagement
                'boost': 0.9,  # High engagement
                'reply': 0.8,  # High engagement
                'favorite': 0.7,  # Medium engagement
                'mention': 0.8,  # High engagement
            }
            
            label = interaction_labels.get(interaction['interaction_type'], 0.5)
            
            # Create training example
            example = {
                'user_id': interaction['user_id'],
                'post_id': interaction['post_id'],
                'content_features': interaction['content_data'].get('features', {}),
                'interaction_type': interaction['interaction_type'],
                'timestamp': interaction['timestamp'],
                'label': label,  # Engagement score (0-1)
                'content_embedding': interaction['content_data'].get('embedding', None),
            }
            
            return example
            
        except Exception as e:
            logger.error(f"Error converting interaction to training example: {e}")
            return None

    def _local_training_step(self, training_data: List[Dict]):
        """
        Perform one step of local training on the provided data.

        Args:
            training_data: List of training examples
        """
        if not training_data:
            return
        
        # Group training data by batch
        for i in range(0, len(training_data), self.batch_size):
            batch = training_data[i:i + self.batch_size]
            
            # Prepare batch for training
            batch_data = self._prepare_batch_for_training(batch)
            
            # Call the model update callback
            self.model_update_callback(batch_data)

    def _prepare_batch_for_training(self, batch: List[Dict]) -> Dict:
        """
        Prepare a batch of training data for model training.

        Args:
            batch: List of training examples

        Returns:
            Prepared batch data
        """
        # Extract relevant features
        user_ids = [example['user_id'] for example in batch]
        post_embeddings = [example['content_embedding'] for example in batch if example['content_embedding'] is not None]
        labels = [example['label'] for example in batch]
        
        # Handle missing embeddings
        if not post_embeddings:
            # Create random embeddings as fallback
            dummy_embedding = np.random.randn(1024).astype(np.float32)
            post_embeddings = [dummy_embedding] * len(batch)
        
        # Pad/truncate to ensure consistent sizes
        max_len = max(len(emb) for emb in post_embeddings) if post_embeddings else 1024
        padded_embeddings = []
        for emb in post_embeddings:
            if len(emb) < max_len:
                padded_emb = np.pad(emb, (0, max_len - len(emb)), 'constant')
            else:
                padded_emb = emb[:max_len]
            padded_embeddings.append(padded_emb)
        
        return {
            'user_ids': user_ids,
            'post_embeddings': np.array(padded_embeddings),
            'labels': np.array(labels, dtype=np.float32),
            'interaction_types': [example['interaction_type'] for example in batch]
        }

    def start_background_processing(self):
        """Start background thread for continuous processing."""
        if self.running:
            logger.warning("Background processing already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("Started background real-time processing")

    def stop_background_processing(self):
        """Stop background processing."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        
        logger.info("Stopped background real-time processing")

    def _background_worker(self):
        """Background worker thread that periodically checks for updates."""
        while self.running:
            try:
                time.sleep(60)  # Check every minute
                self._maybe_perform_update()
            except Exception as e:
                logger.error(f"Error in background worker: {e}")
                time.sleep(60)  # Wait before trying again

    def trigger_federated_sync(self) -> Dict:
        """
        Trigger a federated synchronization with the server.

        Returns:
            Dictionary with sync results and metrics
        """
        logger.info(f"Triggering federated sync for client {self.client_id}")
        
        try:
            # Perform local update before sync (if needed)
            self._maybe_perform_update()
            
            # Get current model parameters to sync
            sync_result = self.aggregation_callback()
            
            sync_metrics = {
                'client_id': self.client_id,
                'interactions_since_last_sync': len(self.interaction_buffer),
                'total_interactions_processed': self.total_interactions_processed,
                'total_updates_performed': self.total_updates_performed,
                'sync_timestamp': datetime.now().isoformat(),
                'sync_successful': True,
                'sync_result': sync_result
            }
            
            logger.info(f"Federated sync completed for client {self.client_id}")
            return sync_metrics
            
        except Exception as e:
            logger.error(f"Error during federated sync: {e}")
            return {
                'client_id': self.client_id,
                'sync_successful': False,
                'error': str(e),
                'sync_timestamp': datetime.now().isoformat()
            }

    def get_status(self) -> Dict:
        """
        Get current status of the real-time learning system.

        Returns:
            Status dictionary
        """
        with self.lock:
            buffer_size = len(self.interaction_buffer)
        
        time_since_last_update = datetime.now() - self.last_update_time
        
        return {
            'client_id': self.client_id,
            'running': self.running,
            'buffer_size': buffer_size,
            'max_buffer_size': self.max_buffer_size,
            'interactions_processed': self.total_interactions_processed,
            'local_updates_performed': self.total_updates_performed,
            'time_since_last_update_minutes': time_since_last_update.total_seconds() / 60,
            'update_frequency_minutes': self.update_frequency_minutes,
            'batch_size': self.batch_size
        }

    def reset_statistics(self):
        """Reset all statistics."""
        with self.lock:
            self.total_interactions_processed = 0
            self.total_updates_performed = 0
            self.interaction_buffer.clear()
        
        self.last_update_time = datetime.now()
        
        logger.info(f"Reset statistics for client {self.client_id}")


class RealTimeFederatedTrainer:
    """
    Trainer class that integrates with the real-time federated learning system.
    """
    
    def __init__(self, model, learning_rate: float = 1e-4):
        """
        Initialize real-time trainer.

        Args:
            model: The model to train
            learning_rate: Learning rate for updates
        """
        self.model = model
        self.learning_rate = learning_rate
        
        # Use a smaller learning rate for real-time updates to avoid overfitting
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.criterion = torch.nn.BCELoss()

    def update_model(self, batch_data: Dict):
        """
        Update model with a batch of real-time data.

        Args:
            batch_data: Batch data from real-time processing
        """
        self.model.train()
        
        try:
            # Convert numpy arrays to tensors
            post_embeddings = torch.tensor(batch_data['post_embeddings'], dtype=torch.float32)
            labels = torch.tensor(batch_data['labels'], dtype=torch.float32)
            
            # Create dummy user IDs if not provided (for compatibility)
            user_ids = batch_data.get('user_ids', [0] * len(labels))
            user_tensor = torch.tensor([int(uid[-8:], 16) % self.model.user_embedding.num_embeddings 
                                        if isinstance(uid, str) else uid % self.model.user_embedding.num_embeddings
                                        for uid in user_ids], dtype=torch.long)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Prepare content tokens (for compatibility with existing model)
            content_tokens = {
                'input_ids': post_embeddings if len(post_embeddings.shape) == 2 else post_embeddings.unsqueeze(1)
            }
            
            predictions = self.model(content_tokens, user_tensor)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent instability in real-time updates
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            logger.debug(f"Real-time update: loss={loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"Error during real-time model update: {e}")
            raise


def test_realtime_federated_learning():
    """Test real-time federated learning system."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock callbacks for testing
    def mock_model_update(batch_data):
        print(f"Updating model with batch of {len(batch_data['labels'])} examples")
    
    def mock_aggregation():
        print("Aggregating with server...")
        return {"status": "success", "round": 1}
    
    # Create real-time federated learning system
    rtfl = RealTimeFederatedLearning(
        model_update_callback=mock_model_update,
        aggregation_callback=mock_aggregation,
        client_id="test_client_1"
    )
    
    print("Testing real-time federated learning...")
    
    # Simulate some interactions
    for i in range(10):
        rtfl.on_new_interaction(
            user_id=f"user_{i}",
            post_id=f"post_{i}",
            interaction_type="like" if i % 3 == 0 else "view",
            content_data={
                "features": {"text": f"Sample content {i}"},
                "embedding": np.random.randn(1024).astype(np.float32)
            }
        )
    
    print(f"Status: {rtfl.get_status()}")
    
    # Perform sync
    sync_result = rtfl.trigger_federated_sync()
    print(f"Sync result: {sync_result}")
    
    return rtfl


if __name__ == "__main__":
    test_realtime_federated_learning()