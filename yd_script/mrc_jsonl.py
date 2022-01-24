import datasets
import jsonlines


class MrcJsonlConfig(datasets.BuilderConfig):
    """BuilderConfig for mrc_for_ner jsonl."""

    def __init__(self, **kwargs):
        """BuilderConfig for mrc_for_ner jsonl file.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MrcJsonlConfig, self).__init__(**kwargs)


class MrcJsonl(datasets.GeneratorBasedBuilder):
    """mrc_for_ner jsonl input data setting"""

    BUILDER_CONFIGS = [
        MrcJsonlConfig(
            name="", version=datasets.Version("0.0.1"),
            description=""
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "asin": datasets.Value("string"),
                    "product_type": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "entity_label": datasets.Value("string"),
                    "qas_id": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "start_position": datasets.Sequence(datasets.Value("int32")),
                    "end_position": datasets.Sequence(datasets.Value("int32")),
                    "span_position": datasets.Sequence(datasets.Value("string")),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(squad_v2): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs

        if self.config.data_files:
            urls_to_download = {
                "train": self.config.data_files["train"],
                # "validation": self.config.data_files["validation"],
            }
        else:
            raise ValueError("must assign data_files")
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            # datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with jsonlines.open(filepath) as reader:
            for index, instance in enumerate(reader):
                # dic = {(k,v) for k,v in instance.items() if k not in ["start_position", "end_position", "span_position"]}
                id_ = str(index)
                yield id_, instance
