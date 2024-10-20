from dataset.preprocess_embedding import *

class LoadEmbeddingSingle:
    def __init__(
        self,
        upstream_model_type=None,
        frame_pooling_type=None,
        device=None
    ):
        self.upstream_model_type = upstream_model_type
        self.frame_pooling_type = frame_pooling_type
        self.device = device

    def load_embedding(self, dataset_id, root, root_emb, label_mapping, subset_percentage=None):
        # Define a mapping from dataset_id to the corresponding loading function
        dataset_mapping = {
            "speechcommand": self._load_speechcommand,
            "voxceleb": self._load_voxceleb,
            "iemocap": self._load_iemocap
        }

        # Check if the dataset_id is supported
        if dataset_id not in dataset_mapping:
            raise ValueError(f"Unsupported dataset_id: {dataset_id}")

        # Get the appropriate loading function based on dataset_id
        load_function = dataset_mapping[dataset_id]

        # Load the dataset using the appropriate function, passing the required arguments
        train_data, val_data, test_data = load_function(root, root_emb, label_mapping)

        # Apply subset_percentage if provided
        if subset_percentage is not None:
            train_data = PercentageSubset(train_data, subset_percentage)
            val_data = PercentageSubset(val_data, subset_percentage)
            test_data = PercentageSubset(test_data, subset_percentage)

        return train_data, val_data, test_data

    def _load_speechcommand(self, root, root_emb, label_mapping):
        speechcommand_train_data = SPEECHCOMMANDSEmbedding(
            root=root,
            root_embedding=root_emb,
            frame_pooling_type=self.frame_pooling_type,
            url="speech_commands_v0.01",
            subset="training",
            download=False,
            label_mapping=label_mapping,
            upstream_model_type=self.upstream_model_type,
            device=self.device,
        )
        speechcommand_val_data = SPEECHCOMMANDSEmbedding(
            root=root,
            root_embedding=root_emb,
            frame_pooling_type=self.frame_pooling_type,
            url="speech_commands_v0.01",
            subset="validation",
            download=False,
            label_mapping=label_mapping,
            upstream_model_type=self.upstream_model_type,
            device=self.device,
        )
        speechcommand_test_data = SPEECHCOMMANDSEmbedding(
            root=root,
            root_embedding=root_emb,
            frame_pooling_type=self.frame_pooling_type,
            url="speech_commands_v0.01",
            subset="testing",
            download=False,
            label_mapping=label_mapping,
            upstream_model_type=self.upstream_model_type,
            device=self.device,
        )

        return speechcommand_train_data, speechcommand_val_data, speechcommand_test_data

    def _load_voxceleb(self, root, root_emb, label_mapping):
        voxceleb_train_data = VoxCeleb1Embedding(
            root=root,
            root_embedding=root_emb,
            frame_pooling_type=self.frame_pooling_type,
            subset='train',
            download=False,
            label_mapping=label_mapping,
            upstream_model_type=self.upstream_model_type,
            device=self.device,
        )
        voxceleb_val_data = VoxCeleb1Embedding(
            root=root,
            root_embedding=root_emb,
            frame_pooling_type=self.frame_pooling_type,
            subset='dev',
            download=False,
            label_mapping=label_mapping,
            upstream_model_type=self.upstream_model_type,
            device=self.device,
        )
        voxceleb_test_data = VoxCeleb1Embedding(
            root=root,
            root_embedding=root_emb,
            frame_pooling_type=self.frame_pooling_type,
            subset='test',
            download=False,
            label_mapping=label_mapping,
            upstream_model_type=self.upstream_model_type,
            device=self.device,
        )

        return voxceleb_train_data, voxceleb_val_data, voxceleb_test_data

    def _load_iemocap(self, root, root_emb, label_mapping):
        iemocap_total_data = IEMOCAPEmbedding(
            root=root,
            root_embedding=root_emb,
            frame_pooling_type=self.frame_pooling_type,
            sessions=("1", "2", "3", "4", "5"),
            label_mapping=label_mapping,
            upstream_model_type=self.upstream_model_type,
            device=self.device,
        )

        iemocap_total_samples = len(iemocap_total_data)

        # Create indices for the split
        indices = list(range(iemocap_total_samples))
        val_indices = indices[4::10]
        test_indices = indices[9::10]
        train_indices = [idx for idx in indices if idx not in val_indices and idx not in test_indices]

        iemocap_train_data = torch.utils.data.Subset(iemocap_total_data, train_indices)
        iemocap_val_data = torch.utils.data.Subset(iemocap_total_data, val_indices)
        iemocap_test_data = torch.utils.data.Subset(iemocap_total_data, test_indices)

        return iemocap_train_data, iemocap_val_data, iemocap_test_data