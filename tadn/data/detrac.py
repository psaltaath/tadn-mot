# Local imports
import os
from glob import glob
from typing import Iterable, List, Tuple
from xml.etree import ElementTree as ET
import cv2
import numpy as np

from .base import MOTDataset
from .mot_challenge import MOTChallengeDetections


class DetracDataset(MOTDataset):
    """MOT Dataset for UA-Detrac dataset

    Inherits from:
        .base.MOTDataset
    """

    def __init__(
        self,
        *args,
        detector: str = "EB",
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            *args: MOTDataset non keyword args.
            detector (str, optional): Selected object detector. Defaults to "EB".
            **kwargs: MOTDataset keyword args.

        """

        assert detector in ["EB", "frcnn"]

        self.detector = detector
        self.detections_provider = None
        self.include_motc: bool

        super().__init__(*args, **kwargs)

    def _retrieve_sequences(self) -> Iterable[str]:
        """Retrieve available sequences

        Returns:
            Iterable[str]: List of available sequences
        """
        return sorted(
            list(
                map(
                    os.path.basename,
                    glob(
                        os.path.join(
                            self.data_root,
                            f"Insight-MVT_Annotation_{self.mode.capitalize()}",
                            "*",
                        )
                    ),
                )
            )
        )

    def _build_db(self) -> None:
        """Private method to build dataset"""

        self.has_gt_annotations = True

        self.detections_provider = MOTChallengeDetections(
            rtv_fun=lambda seq: os.path.join(
                self.data_root, self.detector, f"{seq}_Det_{self.detector}.txt"
            ),
        )

        self.include_motc = os.path.exists(os.path.join(self.data_root, "motc_gt"))

        self.db: List[dict] = []

        for seq in self._retrieve_sequences():
            # Compute how many frames
            seq_data_dir = os.path.join(
                self.data_root, f"Insight-MVT_Annotation_{self.mode.capitalize()}", seq
            )

            annotations_xml = os.path.join(
                self.data_root,
                f"DETRAC-{self.mode.capitalize()}-Annotations-XML",
                seq + ".xml",
            )
            assert os.path.exists(annotations_xml), f"Annotations not found for {seq}"
            annotations_xml_root = ET.parse(
                f"/data/DETRAC/DETRAC-{self.mode.capitalize()}-Annotations-XML/{seq}.xml"
            ).getroot()

            frame_annotation_elements = sorted(
                list(filter(lambda c: c.tag == "frame", annotations_xml_root)),
                key=lambda el: int(el.get("num", default=-1)),
            )

            for frame_id, frame_el in enumerate(
                frame_annotation_elements
            ):  # 0-based frame indexing
                img_url = os.path.join(seq_data_dir, f"img{frame_id+1:05d}.jpg")
                dets = self.detections_provider.get(frame_id=frame_id + 1, seq=seq)
                track_ids, gt = self._parse_frame_xml_element(frame_el)

                self.db.append(
                    {
                        "frame_height": 540,
                        "frame_width": 960,
                        "frame_id": frame_id,
                        "gt": gt,
                        "track_ids": track_ids,
                        "seq": seq,
                        "img_url": img_url,
                        "detections": dets,
                        "is_last_frame_in_seq": False,
                        "seq_first_frame": 0,
                    }
                )

                if self.include_motc:
                    self.db[-1].update(
                        {
                            "MOTC_gt_file": os.path.join(
                                self.data_root, "motc_gt", f"{seq}", "gt", "gt.txt"
                            )
                        }
                    )
            self.db[-1]["is_last_frame_in_seq"] = True

    def _parse_box(self, tgt_el: ET.Element) -> np.ndarray:
        """Parse bbox location from xml element

        Args:
            tgt_el (ET.Element): XML element to parse.

        Returns:
            (np.ndarray): (4, ) bbox location
        """
        box = tgt_el.find("box")
        assert box is not None

        return np.array(
            [
                float(box.get("left", default="-1")),
                float(box.get("top", default="-1")),
                float(box.get("width", default="-1")),
                float(box.get("height", default="-1")),
            ]
        )

    def _parse_frame_xml_element(self, fr_el: ET.Element) -> Tuple[List, np.ndarray]:
        """Parse MOT groundtruth data from XML.

        Args:
            fr_el (ET.Element): Element to parse

        Returns:
            List: Target ids.
            np.ndarray]: (num_tagets, 4) target bbox locations.
        """
        target_list_element: ET.Element = fr_el[0]
        assert target_list_element.tag == "target_list"

        target_ids = [int(t.get("id", default=-1)) for t in target_list_element]
        assert -1 not in target_ids, "Found gt target with no id!"
        target_locations = np.array(
            [self._parse_box(t) for t in target_list_element]
        ).astype(np.float32)

        return target_ids, target_locations

    def _load_frame_data(self, sample: dict) -> dict:
        """Load frame data

        Args:
            sample (dict): Sample without image data

        Returns:
            dict: Updated sample with image data
        """
        assert os.path.exists(sample["img_url"])
        sample.update({"frame_data": cv2.imread(sample["img_url"])})

        return sample
