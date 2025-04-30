from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Roboracer2ConesVla(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='onboard camera RGB observation',
                        ),
                        'depth_image': tfds.features.Image(
                            shape=(360, 640, 1),
                            dtype=np.float32,
                            encoding_format='png',
                            doc='onboard camera depth observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x position, 4x quaternion].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.float32,
                        doc='Robot action, consists of Vx and steering angle.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode*.npz')
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        for episode_path in glob.glob(path):
            # Load all arrays from .npz
            data = np.load(episode_path)
            timestamps = data['timestamps']          # (n,)
            positions = data['positions']            # (n,3)
            orientations = data['orientations']       # (n,4)
            rgb_images = data['rgb_images']          # (n,360,640,3)
            depth_images = data['depth_images']      # (n,360,640)
            actions = data['actions']                 # (n,2)

            n_steps = timestamps.shape[0]
            instruction = 'Drive the car through the gap between the two cones directly ahead, staying centered in the opening and holding a steady speed.'
            instruction_embedding = self._embed([instruction])[0].numpy()

            episode = []
            for i in range(n_steps):
                # Combine position + orientation into state vector
                state = np.concatenate([positions[i], orientations[i]], axis=0)
                episode.append({
                    'observation': {
                        'image':rgb_images[i],
                        'depth_image': depth_images[i].reshape(360, 640, 1),
                        'state': state,
                    },
                    'action': actions[i],
                    'discount': 1.0,
                    'reward': float(i == n_steps - 1),
                    'is_first': i == 0,
                    'is_last': i == n_steps - 1,
                    'is_terminal': i == n_steps - 1,
                    'language_instruction': instruction,
                    'language_embedding': instruction_embedding,
                })

            # Assemble the final sample
            sample = {
                'steps': episode,
                'episode_metadata': {'file_path': episode_path},
            }
            yield episode_path, sample


        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

