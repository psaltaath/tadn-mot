import os
import random
import shutil
import string
from typing import Dict, Optional, Tuple

import pandas as pd
import trackeval


class MOTInference:
    """Class to handle MOT inference in MOTChallenge format"""

    def __init__(
        self,
        benchmark: str = "MOT17",
        val_tmp_root: str = "./test_results",
    ) -> None:
        """Constructor

        Args:
            benchmark (str, optional): Name of benchmark. Defaults to "MOT17".
            val_tmp_root (str, optional): Path to output location. Defaults to "./test_results".
        """
        self.benchmark = benchmark
        assert self.benchmark in ["MOT15", "MOT17"]
        self.split_to_eval = "test"
        self.val_tmp_root = val_tmp_root

    def reset(self, tmp_dir: Optional[str] = None) -> None:
        """Reset class to start over

        Args:
            tmp_dir (Optional[str], optional): Temp dir to save results in. Defaults to None.
        """

        if tmp_dir is None:
            self.tmp_dir = self.val_tmp_root
        else:
            self.tmp_dir = tmp_dir

        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        self.res_dir = os.path.join(self.tmp_dir, "tracker", "under_eval")

        os.makedirs(os.path.join(self.res_dir, "data"))

    def register_sequence(self, seq_name: str) -> str:
        """Register a new sequence

        Args:
            seq_name (str): sequence name

        Returns:
            str: path to results file for the sequence
        """

        res_file = os.path.join(self.res_dir, "data", seq_name + ".txt")
        return res_file


class MOTEvaluator(MOTInference):
    """Class to handle MOT evaluation in MOTChallenge format

    Inherits from:
        MOTInference
    """

    def __init__(
        self,
        benchmark: str = "MOT17",
        split_to_eval: str = "train",
        val_tmp_root: str = ".",
        metric_name_prepend: str = "val",
    ) -> None:
        """Constructor

        Args:
            benchmark (str, optional): Name of benchmark. Defaults to "MOT17".
            split_to_eval (str, optional): Which dataset mode/split to eval. Defaults to "train".
            val_tmp_root (str, optional): Path to output location. Defaults to ".".
            metric_name_prepend (str, optional): str identifier to prepend each metric name. Defaults to "val".
        """

        super().__init__(benchmark=benchmark, val_tmp_root=val_tmp_root)

        self.split_to_eval = split_to_eval
        assert self.split_to_eval in ["train", "test", "all"]

        self.metric_name_prepend = metric_name_prepend
        self.reset()

    def reset(self) -> None:
        """Reset class to start over"""

        tmp_dir = os.path.join(
            self.val_tmp_root,
            "tmp__{}".format(
                "".join([random.choice(string.ascii_uppercase) for _ in range(5)])
            ),
        )

        super().reset(tmp_dir=tmp_dir)

        self.gt_dir = os.path.join(self.tmp_dir, "gt")

        os.makedirs(os.path.join(self.gt_dir, "seqmaps"))

        self.seqmap_file = os.path.join(
            self.tmp_dir,
            "gt",
            "seqmaps",
            "-".join([self.benchmark, self.split_to_eval]) + ".txt",
        )
        with open(self.seqmap_file, "w") as f:
            f.write("name" + "\n")

    def _write_seqmap(self, content: str):
        """Write data to seqmap

        Args:
            content (str): Content to write
        """
        with open(self.seqmap_file, "a") as f:
            f.write(content + "\n")

    def register_file(self, gt_file: str) -> str:
        """Register a new gt file

        Args:
            gt_file (str): register a new gt file for evaluation

        Returns:
            str: path to results file for the sequence corresponding to gt file
        """

        seq_dir = os.path.dirname(os.path.dirname(gt_file))
        seq_name = os.path.basename(seq_dir)

        ini_file = os.path.join(seq_dir, "seqinfo.ini")
        assert os.path.exists(ini_file), f"Ini file {ini_file} doesn't exist"

        tgt_dir_final = os.path.join(self.gt_dir, seq_name, "gt")
        if not os.path.exists(tgt_dir_final):
            os.makedirs(tgt_dir_final)

        shutil.copyfile(gt_file, os.path.join(tgt_dir_final, os.path.basename(gt_file)))
        shutil.copyfile(
            ini_file,
            os.path.join(os.path.dirname(tgt_dir_final), os.path.basename(ini_file)),
        )

        self._write_seqmap(seq_name)

        return self.register_sequence(seq_name)

    def _generate_configs(self) -> Tuple[dict, dict, dict]:
        """Generate configs compatible with trackeval repo

        Returns:
            dict: Evaluation config
            dict: Dataset config
            dict: Metrics config
        """
        # Command line interface:
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config["DISPLAY_LESS_PROGRESS"] = False
        eval_config["PRINT_RESULTS"] = False
        eval_config["PRINT_ONLY_COMBINED"] = False
        eval_config["PRINT_CONFIG"] = False
        eval_config["TIME_PROGRESS"] = False
        eval_config["DISPLAY_LESS_PROGRESS"] = False

        eval_config["OUTPUT_SUMMARY"] = False
        eval_config["OUTPUT_EMPTY_CLASSES"] = True
        eval_config["OUTPUT_DETAILED"] = True
        eval_config["PLOT_CURVES"] = True

        dataset_config = (
            trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        )
        dataset_config["GT_FOLDER"] = self.gt_dir
        dataset_config["BENCHMARK"] = self.benchmark
        dataset_config["TRACKERS_FOLDER"] = os.path.dirname(self.res_dir)
        dataset_config["TRACKERS_TO_EVAL"] = [os.path.basename(self.res_dir)]
        dataset_config["SKIP_SPLIT_FOL"] = True
        dataset_config["PRINT_CONFIG"] = False

        metrics_config = {"METRICS": ["CLEAR"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
        return eval_config, dataset_config, metrics_config

    def eval(self) -> Dict[str, float]:
        """Perform evaluation

        Returns:
            Dict[str, float]: Metrics report wit items: {metric_name:metric_value}
        """
        eval_config, dataset_config, metrics_config = self._generate_configs()

        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric in [
            trackeval.metrics.HOTA,
            trackeval.metrics.CLEAR,
            trackeval.metrics.Identity,
            trackeval.metrics.VACE,
        ]:
            if metric.get_name() in metrics_config["METRICS"]:
                metrics_list.append(metric(metrics_config))
        if len(metrics_list) == 0:
            raise Exception("No metrics selected for evaluation")

        evaluator.evaluate(dataset_list, metrics_list)

        metrics_report = self.parse_results()

        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        self.reset()

        return metrics_report

    def parse_results(self) -> Dict[str, float]:
        """Parse results from trackeval evaluation

        Returns:
            Dict[str, float]: Metrics report wit items: {metric_name:metric_value}
        """
        results_csv = os.path.join(self.res_dir, "pedestrian_detailed.csv")
        assert os.path.exists(results_csv)

        R = pd.read_csv(results_csv, header=0, index_col=0)

        metrics_report = {}

        for seq_name, seq_results in R.iterrows():
            metrics_report.update(
                {
                    "_".join(
                        [self.metric_name_prepend, str(seq_name), metric_name]
                    ): seq_results[metric_name]
                    for metric_name in seq_results.keys()
                }
            )

        return metrics_report

    def __del__(self):
        """Destructor. Clean-up"""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
