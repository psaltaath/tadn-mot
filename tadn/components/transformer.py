from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

# Local imports
from ..mot.metrics import pairwise_ulbr1_metric


class SingleBranchTransformer(nn.Module):
    """Single Branch TADN configuration"""

    def __init__(self, d_model, nhead=2, encoder_num_layers=2, decoder_num_layers=2):
        """Constructor for Single Branch TADN Transformer model

        Args:
            d_model (int): Number of expected features in the transformer inputs
            nhead (int, optional): Number of heads. Defaults to 2.
            encoder_num_layers (int, optional): Number of encoder layers. Defaults to 2.
            decoder_num_layers (int, optional): Number of decoder layers. Defaults to 2.
        """

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_num_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_num_layers
        )

        self.d_model = d_model

    def forward(
        self,
        targets,
        detections,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            targets (torch.Tensor): (num_targets+1, batch_size, d_model) Targets input stream. Includes null-target.
            detections (torch.Tensor): (num_detections, batch_size, d_model) Detections input stream.
            src_key_padding_mask (Optional[torch.Tensor], optional): Mask for src keys per batch. Defaults to None.
            tgt_key_padding_mask (Optional[torch.Tensor], optional): Mask for tgt keys per batch. Defaults to None.
            memory_mask (Optional[torch.Tensor], optional): Additive mask for the encoder output. Defaults to None.

        Returns:
            torch.Tensor: (num_targets+1, d_model) targets output stream.
            torch.Tensor: (num_detections, d_model) detections output stream.
        """
        transformed_targets = self.encoder(
            targets, src_key_padding_mask=src_key_padding_mask
        )
        transformed_detections = self.decoder(
            detections, transformed_targets, tgt_key_padding_mask=tgt_key_padding_mask
        )

        return transformed_detections, transformed_targets


class DualBranchTransformer(nn.Module):
    """Dual Branch TADN configuration"""

    def __init__(
        self,
        d_model: int,
        nhead: int = 2,
        encoder_num_layers: int = 2,
        decoder_num_layers: int = 2,
    ):
        """Constructor for Dual Branch TADN Transformer model

        Args:
            d_model (int): Number of expected features in the transformer inputs
            nhead (int, optional): Number of heads. Defaults to 2.
            encoder_num_layers (int, optional): Number of encoder layers. Defaults to 2.
            decoder_num_layers (int, optional): Number of decoder layers. Defaults to 2.
        """
        super().__init__()
        self.d_model = d_model
        self.target_stream = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=encoder_num_layers,
            num_decoder_layers=decoder_num_layers,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )

        self.detection_stream = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=encoder_num_layers,
            num_decoder_layers=decoder_num_layers,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )

    def forward(
        self,
        targets: torch.Tensor,
        detections: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            targets (torch.Tensor): (num_targets+1, batch_size, d_model) Targets input stream. Includes null-target.
            detections (torch.Tensor): (num_detections, batch_size, d_model) Detections input stream.
            src_key_padding_mask (Optional[torch.Tensor], optional): Mask for src keys per batch. Defaults to None.
            tgt_key_padding_mask (Optional[torch.Tensor], optional): Mask for tgt keys per batch. Defaults to None.
            memory_mask (Optional[torch.Tensor], optional): Additive mask for the encoder output. Defaults to None.

        Returns:
            torch.Tensor: (num_targets+1, d_model) targets output stream.
            torch.Tensor: (num_detections, d_model) detections output stream.
        """
        # src -> input to encoder, tgt: input to decoder. "detection" stream is the classic approach

        transformed_targets = self.target_stream(
            src=detections,
            tgt=targets,
            src_mask=None,
            tgt_mask=None,
            memory_mask=memory_mask,
            src_key_padding_mask=tgt_key_padding_mask,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=None,
        )

        if memory_mask is not None:
            memory_mask = memory_mask.T

        transformed_detections = self.detection_stream(
            src=targets,
            tgt=detections,
            src_mask=None,
            tgt_mask=None,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )

        return transformed_detections, transformed_targets


class TADN(nn.Module):
    """Transformer-based Assignment Decision Network Module Class"""

    def __init__(
        self,
        transformer_model: Union[SingleBranchTransformer, DualBranchTransformer],
        embedding_params: dict,
        null_target_params: dict,
        normalize_transformer_outputs: bool = False,
    ):
        """Constructor for TADN

        Args:
            transformer_model (Union[SingleBranchTransformer, DualBranchTransformer]): An instance of the TADN Transformer model to be used.
            embedding_params (dict): Parameters regarding embeddings.
                Keys: ('app_embedding_dim', 'app_dim', 'spatial_embedding_dim', [['spatial_memory_mask_weight']])
            null_target_params (dict): Parameters regarding null-target. Keys: ('null_target_idx')
            normalize_transformer_outputs (bool, optional): Perform batch-norm on TADN outputs. Defaults to False.
        """

        super().__init__()
        # self._assert_transformer_inputs(**transformer_params)
        self._assert_embedding_inputs(**embedding_params)
        self._assert_null_target_inputs(**null_target_params)

        self.init_hyperparameters = {
            "embedding_params": embedding_params,
            "null_target_params": null_target_params,
        }

        # d_model = app_embedding_dim + spatial_embedding_dim
        self.transformer = transformer_model

        self.app_embedding = nn.Linear(
            embedding_params["app_dim"], embedding_params["app_embedding_dim"]
        )

        # TODO: Parameterize more the following
        if embedding_params["spatial_embedding_dim"] != 0:
            self.spatial_embedding = nn.Sequential(
                nn.Linear(4, embedding_params["spatial_embedding_dim"] // 4),
                nn.Tanh(),
                nn.Linear(
                    embedding_params["spatial_embedding_dim"] // 4,
                    embedding_params["spatial_embedding_dim"] // 2,
                ),
                nn.Tanh(),
                nn.Linear(
                    embedding_params["spatial_embedding_dim"] // 2,
                    embedding_params["spatial_embedding_dim"],
                ),
            )
        else:
            self.spatial_embedding = None
            self.memory_weight = embedding_params["spatial_memory_mask_weight"]

        self.null_target = nn.Embedding(
            num_embeddings=1, embedding_dim=self.transformer.d_model
        )
        self.null_target_idx = null_target_params["null_target_idx"]

        self.normalize_transformer_outputs = normalize_transformer_outputs
        if self.normalize_transformer_outputs:
            self.targets_normalization = nn.BatchNorm1d(
                self.transformer.d_model, affine=True
            )
            self.detections_normalization = nn.BatchNorm1d(
                self.transformer.d_model, affine=True
            )

    def _assert_transformer_inputs(self, **kwargs):
        assert "nhead" in kwargs.keys()
        assert "encoder_num_layers" in kwargs.keys()
        assert "decoder_num_layers" in kwargs.keys()

    def _assert_embedding_inputs(self, **kwargs):
        assert "app_dim" in kwargs.keys()
        assert "app_embedding_dim" in kwargs.keys()
        assert "spatial_embedding_dim" in kwargs.keys()
        assert (
            kwargs["spatial_embedding_dim"] == 0
            or kwargs["spatial_embedding_dim"] >= 16
        )  # 4(xyxy) * 2**2 (spatial emb)

        if kwargs["spatial_embedding_dim"] == 0:
            assert "spatial_memory_mask_weight" in kwargs.keys()
            assert kwargs["spatial_memory_mask_weight"] is not None
        else:
            if "spatial_memory_mask_weight" in kwargs.keys():
                assert kwargs["spatial_memory_mask_weight"] is None

    def _assert_null_target_inputs(self, **kwargs):
        assert "null_target_idx" in kwargs.keys()

    def _compute_sdp_similarity(
        self,
        targets_output: torch.Tensor,
        detections_output: torch.Tensor,
        apply_softmax: bool = True,
    ) -> torch.Tensor:
        """Computes a similarity matrix using the scaled dot product operator

        Args:
            targets_output (torch.Tensor): (num_targets+1, d_model) TADN Transformer targets output stream. Include null-target
            detections_output (torch.Tensor): (num_detections, d_model) TADN Transformer detections output stream.
            apply_softmax (bool, optional): Perform row-wise softmax to output. Defaults to True.

        Returns:
            torch.Tensor: (num_detections, num_targets+1) Similarity matrix
        """
        K = targets_output.contiguous()
        Q = detections_output.contiguous()

        sdp = torch.matmul(K, Q.transpose(0, 1)) / torch.sqrt(
            torch.tensor(K.size()[-1])
        )

        similarity_matrix = sdp.transpose(0, 1)

        if apply_softmax:
            similarity_matrix = F.softmax(similarity_matrix, dim=-1)

        return similarity_matrix

    def forward(self, targets, track_app_vec, detections, det_app_vectors):
        """Forward pass for the TADN module

        Args:
            targets (torch.Tensor): (num_targets, 4) bbox locations of active targets
            track_app_vec (torch.Tensor): (num_targets, app_dim) appearance features of active targets
            detections (torch.Tensor): (num_detections, 4) bbox locations of detections
            det_app_vectors (torch.Tensor): (num_detections, app_dim) appearance features of detections

        Returns:
            torch.Tensor: (num_detections, num_targets+1) Similarity matrix between active targets and detections. Includes null target
        """

        targets_emb_list = []
        if self.spatial_embedding is not None:
            targets_emb_list.append(self.spatial_embedding(targets))
        targets_emb_list.append(self.app_embedding(track_app_vec))

        targets_emb = torch.cat(targets_emb_list, dim=-1)

        #  Add null target
        nt_emb = self.null_target(torch.Tensor([0]).type(torch.long).to(targets.device))
        targets_emb_nt = torch.cat([targets_emb, nt_emb.view(1, -1)], dim=0)

        detections_emb_list = []
        if self.spatial_embedding is not None:
            detections_emb_list.append(self.spatial_embedding(detections))
        detections_emb_list.append(self.app_embedding(det_app_vectors))
        detections_emb = torch.cat(detections_emb_list, dim=-1)

        memory_mask = None
        if self.spatial_embedding is None:
            memory_mask = self.memory_weight * pairwise_ulbr1_metric(
                targets, detections
            ).to(targets.device)

            memory_mask = torch.cat(
                [
                    memory_mask,
                    torch.zeros(1, len(detections))
                    .type(memory_mask.type())
                    .to(memory_mask.device),  # type: ignore
                ],
                dim=0,
            )

        detections_output, targets_output = self.transformer(
            targets_emb_nt,
            detections_emb,
            memory_mask=memory_mask,
        )  # Output shapes: (Num_tgt / Num_det x E)

        # Perform batch-norm on transformer outputs (default: disabled)
        if self.normalize_transformer_outputs:
            targets_output = self.targets_normalization(targets_output)
            detections_output = self.detections_normalization(detections_output)

        similarity_matrix = self._compute_sdp_similarity(
            targets_output, detections_output
        )

        return similarity_matrix
