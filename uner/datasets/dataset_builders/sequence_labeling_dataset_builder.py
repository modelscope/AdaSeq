import datasets


class SequenceLabelingDatasetBuilder(datasets.GeneratorBasedBuilder):

    def _info(self):
        info = datasets.DatasetInfo(
            features=datasets.Features(
                {
                    'id': datasets.Value('string'),
                    'tokens': datasets.Sequence(datasets.Value('string')),
                    'labels': datasets.Sequence(datasets.Value('string'))
                }
            )
        )
        return info

    def _split_generators(self, dl_manager):
        assert self.config.data_files is not None and len(self.config.data_files) > 0
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            if isinstance(data_files, str):
                data_files = [data_files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                            gen_kwargs={'filepath': data_files[0]})]
        return [datasets.SplitGenerator(name=split_name,
                                        gen_kwargs={'filepath': data_files[split_name][0]})
                for split_name in data_files.keys()]

    def _generate_examples(self, filepath):
        return load_data_file(filepath)


def load_data_file(filepath):
    if filepath.endswith('.json'):
        raise NotImplementedError
    else: 
        return load_column_data_file(filepath)


def load_column_data_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        guid = 0
        tokens = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "labels": labels,
                        }
                    guid += 1
                    tokens = []
                    labels = []
            else:
                splits = line.split()
                tokens.append(splits[0])
                labels.append(splits[-1].rstrip())
        if tokens:
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "labels": labels,
                }

