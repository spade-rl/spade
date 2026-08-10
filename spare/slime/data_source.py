"""SPARE data source for Slime rollout.

This module provides a custom data source for SPARE that doesn't require
pre-existing training/eval data files, since SPARE generates games on-the-fly.
"""

import logging
from typing import List

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class SpareDataSource(DataSource):
    """Custom data source for SPARE that generates data on-the-fly.

    SPARE doesn't use pre-existing datasets - it generates games during training.
    This data source provides minimal implementation to satisfy Slime's requirements.
    """

    def __init__(self, args):
        """Initialize the SPARE data source.

        Args:
            args: Slime arguments (we only use SPARE-specific attributes)
        """
        self.args = args

        # SPARE-specific attributes (all registered in add_spare_arguments)
        self.games_dir = args.spare_games_dir
        self.game_regeneration_interval = args.spare_game_regeneration_interval
        self.num_games_per_rollout = args.spare_num_games_per_rollout

        # Initialize tracking attributes (required by RolloutDataSource)
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        logger.info(
            f"[SPARE] Initialized SpareDataSource: "
            f"games_dir={self.games_dir}, "
            f"regeneration_interval={self.game_regeneration_interval}, "
            f"num_games={self.num_games_per_rollout}"
        )

    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        """Return samples for rollout.

        For SPARE, we return empty lists since the rollout function
        generates samples on-the-fly.

        Args:
            num_samples: Number of samples requested (ignored for SPARE)

        Returns:
            Empty list of sample groups (rollout function generates actual data)
        """
        logger.debug(f"[SPARE] get_samples called with num_samples={num_samples}")
        # Return empty groups - the rollout function generates actual data
        return []

    def add_samples(self, samples: List[List[Sample]]):
        """Add samples to the data source.

        For SPARE, samples are generated during rollout and don't need
        to be stored in the data source.

        Args:
            samples: List of sample groups to add
        """
        logger.debug(f"[SPARE] add_samples called with {len(samples)} sample groups")
        # Samples are handled by the rollout function directly
        pass

    def save(self, rollout_id):
        """Save the state of the data source.

        For SPARE, we don't need to save any state since games
        are generated on-the-fly.

        Args:
            rollout_id: Current rollout ID
        """
        logger.debug(f"[SPARE] save called with rollout_id={rollout_id}")
        # No state to save for SPARE
        pass

    def load(self, rollout_id=None):
        """Load the state of the data source.

        For SPARE, we don't need to load any state since games
        are generated on-the-fly.

        Args:
            rollout_id: Optional rollout ID to load
        """
        logger.debug(f"[SPARE] load called with rollout_id={rollout_id}")
        # No state to load for SPARE
        pass

    def __len__(self) -> int:
        """Return the length of the data source.

        For SPARE, we return 0 since there's no pre-existing dataset.
        Data is generated on-the-fly during rollout.

        Returns:
            0 (no pre-existing data)
        """
        return 0
